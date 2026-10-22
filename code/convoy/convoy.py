"""Two-vehicle convoy simulation (Fig. 8 architecture).

Leader: NMPC tracking a known reference path and speed profile.
Follower: NMPC tracking the leader's recorded path from a FIFO buffer
(nearest-point lookup), speed reference from a PI loop on the gap error.

Baseline mode (defaults in ConvoyConfig) follows the paper:
  * Eq. 40: v_ref = Kp e_d + Ki * int(e_d)   (no leader-speed feedforward)
  * curvature and speed held constant over the horizon (no preview)
Phase 1 (follower drives straight until it reaches the leader's start) is
implemented by pre-filling the buffer with the straight segment between the
two starting positions, which is equivalent.
"""
import time

import numpy as np

from .estimation import Sensors, make_estimator
from .nmpc import NMPC
from .params import ConvoyConfig, MPCConfig, Vehicle
from .paths import Track, wrap
from . import plant


def _horizon(track, s0, Z, T, preview, rho0, v0, Np):
    if not preview:
        return np.full(Np, rho0), np.full(Np, v0)
    # arc-length offsets from the previous predicted speeds
    v_pred = np.maximum(Z[2, 1:], 0.1) if Z is not None else np.full(Np, 1.0)
    s_h = s0 + np.cumsum(T * v_pred)
    return track.at_s(s_h)


def _arc_gap(buf: Track, X, Y, d0):
    """Arc length behind the buffer's end at which the chord to (X, Y) = d0.

    The physical link holds the *chord* at d0. On a curve the follower must
    therefore sit slightly more than d0 of path behind the leader; this
    returns that arc-length lag from the recorded geometry.
    """
    a = buf.arr
    dist = np.hypot(a[:, 1] - X, a[:, 2] - Y)
    far = np.nonzero(dist >= d0)[0]
    if far.size == 0:
        return d0
    j = far[-1]                         # last point still at least d0 away
    if j + 1 >= buf.n:
        return a[-1, 0] - a[j, 0]
    lam = (dist[j] - d0) / max(dist[j] - dist[j + 1], 1e-9)
    s_star = a[j, 0] + np.clip(lam, 0, 1) * (a[j + 1, 0] - a[j, 0])
    return a[-1, 0] - s_star


def _arc_gap_hitch(buf: Track, X, Y, course, L0, hL, hF):
    """Like _arc_gap, but for the hitch-to-hitch distance (what the link sees).

    The follower's hitch at each buffer point is placed hF ahead along the
    recorded course (course ~ yaw; sideslip is small at these speeds).
    """
    a = buf.arr
    pL = np.array([X - hL * np.cos(course), Y - hL * np.sin(course)])
    xF = a[:, 1] + hF * np.cos(a[:, 3])
    yF = a[:, 2] + hF * np.sin(a[:, 3])
    dist = np.hypot(pL[0] - xF, pL[1] - yF)
    far = np.nonzero(dist >= L0)[0]
    if far.size == 0:
        return L0 + hL + hF
    j = far[-1]
    if j + 1 >= buf.n:
        return a[-1, 0] - a[j, 0]
    lam = (dist[j] - L0) / max(dist[j] - dist[j + 1], 1e-9)
    s_star = a[j, 0] + np.clip(lam, 0, 1) * (a[j + 1, 0] - a[j, 0])
    return a[-1, 0] - s_star


def _rho_lookup(buf: Track, s_query, plan_s=None, plan_rho=None):
    """Curvature at arc lengths s_query along the buffer.

    Beyond the buffer's end the leader has not driven yet; there, use the
    leader's shared plan (its own horizon curvature) when available.
    """
    rho, _ = buf.at_s(s_query)
    if plan_s is not None:
        ahead = s_query > buf.length
        if np.any(ahead):
            rho = rho.copy()
            rho[ahead] = np.interp(s_query[ahead], plan_s, plan_rho)
    return rho


def _shift(arr, age):
    """A horizon array (speed or curvature) as seen `age` steps later."""
    if age <= 0:
        return arr
    idx = np.minimum(np.arange(age, age + len(arr)), len(arr) - 1)
    return arr[idx]


def _shift_progress(s_inc, v, age, T):
    """Leader progress since the plan was sent, and its remaining increments.

    s_inc[k] = predicted progress after k+1 steps. Beyond the horizon the
    leader is assumed to hold its last planned speed.
    """
    Np = len(s_inc)

    def s_at(j):                       # progress after j steps, j >= 0
        j = np.asarray(j)
        inside = s_inc[np.clip(j - 1, 0, Np - 1)]
        beyond = s_inc[-1] + (j - Np) * T * v[-1]
        return np.where(j <= 0, 0.0, np.where(j <= Np, inside, beyond))

    since = float(s_at(age))
    inc = s_at(np.arange(age + 1, age + Np + 1)) - since
    return since, inc


def _with_plan_extension(buf: Track, msg):
    """Buffer plus the leader's planned path from its last received pose.

    The plan's curvature is integrated along its predicted progress, giving
    the follower a path to track when the buffer stops growing (lost or
    delayed messages). The recorded part is unchanged.
    """
    s_inc, rho_p, v_p = msg["s_inc"], msg["rho_plan"], msg["v_plan"]
    if "course" not in msg:            # initial placeholder: nothing to extend
        return buf
    ext = Track(buf.n + len(s_inc) + 2)
    ext.data[: buf.n] = buf.arr
    ext.n = buf.n
    x, y, c, s0 = msg["X"], msg["Y"], msg["course"], buf.length
    prev = 0.0
    for k in range(len(s_inc)):
        ds = s_inc[k] - prev
        if ds <= 1e-3:
            continue
        cm = c + 0.5 * rho_p[k] * ds
        x, y, c = x + ds * np.cos(cm), y + ds * np.sin(cm), c + rho_p[k] * ds
        ext.data[ext.n] = (s0 + s_inc[k], x, y, c, rho_p[k], v_p[k])
        ext.n += 1
        prev = s_inc[k]
    return ext


class _Channel:
    """V2V link from leader to follower: fixed delay, random loss, blackout."""

    def __init__(self, K, T):
        self.d = int(round(K.v2v_delay / T))
        self.loss = K.v2v_loss
        self.black = K.v2v_blackout
        self.rng = np.random.default_rng(K.v2v_seed)
        self.T = T
        self.queue = []

    def send(self, n, msg):
        t = n * self.T
        lost = self.rng.random() < self.loss
        if self.black is not None and self.black[0] <= t < self.black[1]:
            lost = True
        if not lost:
            self.queue.append((n + self.d, msg))

    def receive(self, n):
        out = [m for (a, m) in self.queue if a <= n]
        self.queue = [(a, m) for (a, m) in self.queue if a > n]
        return out


def simulate(path: Track, P=Vehicle(), C=MPCConfig(), K=ConvoyConfig(),
             v0=1.0, t_max=120.0, verbose=False, follower_policy=None):
    """Run the convoy. Control uses measured / received data; the logged
    errors are always computed from the true states.

    follower_policy: optional callable obs -> action in [-1, 1]^2 (e.g. a
    trained RL policy, see convoy/rl.py) that replaces the follower's NMPC.
    It receives the same information channels as the robust gap NMPC, so
    run it with leader_plan, plan_path_ext (and plan_compensation) set."""
    T, Np = C.T, C.Np
    mpcL, mpcF = NMPC(P, C), NMPC(P, C, gap_mode=K.gap_nmpc)
    rng_noise = np.random.default_rng(K.noise_seed)
    chan = _Channel(K, T)

    # initial states: leader at path start, follower d0 behind on its heading
    d = path.arr
    x0, y0, psi0 = d[0, 1], d[0, 2], d[0, 3]
    xL = np.array([x0, y0, psi0, v0, 0, 0, 0], float)
    xF = np.array([x0 - K.d0 * np.cos(psi0), y0 - K.d0 * np.sin(psi0),
                   psi0, v0, 0, 0, 0], float)

    # physical link (C3): rest length so that the initial straight-line
    # spacing d0 leaves it unstressed
    L0 = K.d0 - K.hitch_L - K.hitch_F
    c_link = 2 * K.link_zeta * np.sqrt(K.link_k * P.m / 2) if K.link_k > 0 else 0.0

    buf, tbuf = Track(), Track()       # follower's buffer / ground-truth trace
    for lam in np.linspace(0, 1, int(K.d0 / 0.1), endpoint=False):
        for b in (buf, tbuf):
            b.append(xF[0] + lam * (x0 - xF[0]), xF[1] + lam * (y0 - xF[1]),
                     psi0, 0.0, v0)
    # what the follower knows before the first message arrives
    latest = dict(n=-1, X=x0, Y=y0, v=v0, s_inc=np.cumsum(np.full(Np, T * v0)),
                  v_plan=np.full(Np, v0), rho_plan=np.zeros(Np))

    # sensing: 'truth' (optionally with pose noise, as in the first C5 runs),
    # or a sensor suite read by a raw or EKF estimator (estimation.py)
    if K.sensing != "truth":
        rng_s = np.random.default_rng(K.noise_seed + 7)
        sensL, sensF = Sensors(K.gnss_sigma, rng_s), Sensors(K.gnss_sigma, rng_s)
        estL = make_estimator(K.sensing, xL, P, T, K.gnss_sigma)
        estF = make_estimator(K.sensing, xF, P, T, K.gnss_sigma)

    def noisy_pose(x):
        if K.pos_noise <= 0:
            return x[0], x[1], x[2]
        e = rng_noise.normal(0, K.pos_noise, 2)
        return x[0] + e[0], x[1] + e[1], x[2] + rng_noise.normal(0, K.heading_noise)

    history = []                       # leader's own pose log (for redundancy)
    last_pose_n = -1                   # newest pose already in the buffer
    uL = np.zeros(2)
    uF = np.zeros(2)
    hL = hF = hLt = hFt = 0
    integ = 0.0
    ZL = ZF = None
    log = []
    solve_ms = []
    n = 0
    while n * T < t_max:
        t = n * T
        # ---------------- leader (measures its own state)
        if K.sensing == "truth":
            XLm, YLm, psiLm = noisy_pose(xL)
            vL_est = xL[3:7]
        else:
            eL = estL.step(xL, uL, sensL.measure(xL))
            XLm, YLm, psiLm = eL[:3]
            vL_est = eL[3:7]
        hL, sL, eyL, psirL, rhoL, vrefL = path.project(XLm, YLm, hL)
        if sL >= path.length - 1.0:
            break
        epsiL = wrap(psiLm - psirL)
        rho_h, v_h = _horizon(path, sL, ZL, T, K.leader_preview,
                              rhoL, vrefL, Np)
        zL = np.array([eyL, epsiL, *vL_est])
        t0 = time.perf_counter()
        uL, ZL = mpcL.solve(zL, rho_h, v_h, uL)
        solve_ms.append(1e3 * (time.perf_counter() - t0))

        # ---------------- leader broadcasts pose + plan over V2V.
        # Path tangent = course angle (yaw + sideslip), not yaw: with sideslip
        # the two differ, and a follower told to match the leader's yaw *and*
        # position gets contradictory targets in fast corners.
        slip = np.arctan2(vL_est[1], max(vL_est[0], 0.5))
        msg = dict(n=n, X=XLm, Y=YLm, course=psiLm + slip,
                   rho=vL_est[2] / max(vL_est[0], 0.5), v=vL_est[0],
                   s_inc=np.cumsum(T * ZL[2, 1:] * np.cos(ZL[1, 1:])),
                   v_plan=ZL[2, 1:].copy(), rho_plan=rho_h.copy())
        history.append((n, msg["X"], msg["Y"], msg["course"], msg["rho"], msg["v"]))
        if K.pose_history > 0:
            # redundancy: each message also carries the last few seconds of
            # poses, so later packets refill the buffer after losses
            msg["hist"] = history[-K.pose_history:]
        chan.send(n, msg)
        slip_t = np.arctan2(xL[4], max(xL[3], 0.5))
        tbuf.append(xL[0], xL[1], xL[2] + slip_t, xL[5] / max(xL[3], 0.5), xL[3])

        # ---------------- follower: take in whatever has arrived
        for m in chan.receive(n):
            poses = m.get("hist", [(m["n"], m["X"], m["Y"], m["course"], m["rho"], m["v"])])
            for (k, X, Y, c, rh, v) in poses:
                if k > last_pose_n:        # only poses newer than the buffer's end
                    buf.append(X, Y, c, rh, v)
                    last_pose_n = k
            latest = m
        age = n - latest["n"] if K.plan_compensation else 0
        since, sL_inc = _shift_progress(latest["s_inc"], latest["v_plan"], age, T)
        vL_plan = _shift(latest["v_plan"], age)
        rhoL_plan = _shift(latest["rho_plan"], age)
        plan_s = buf.length + since + sL_inc if K.leader_plan else None

        # path the follower tracks: the recorded buffer, optionally extended
        # with the leader's planned path from its last received pose
        fpath = _with_plan_extension(buf, latest) if K.plan_path_ext else buf

        if K.sensing == "truth":
            XFm, YFm, psiFm = noisy_pose(xF)
            vF_est = xF[3:7]
        else:
            eF = estF.step(xF, uF, sensF.measure(xF))
            XFm, YFm, psiFm = eF[:3]
            vF_est = eF[3:7]
        hF, sF, eyF, psirF, rhoF, _ = fpath.project(XFm, YFm, hF)
        epsiF = wrap(psiFm - psirF)
        gap = np.hypot(xL[0] - xF[0], xL[1] - xF[1])   # CoM gap (metric)
        e_d = gap - K.d0
        f_link, _, pLh, pFh = plant.link_force(xL, xF, K.link_k, c_link, L0,
                                               K.hitch_L, K.hitch_F)
        e_link = np.hypot(*(pLh - pFh)) - L0           # link sensor: local, true
        e_meas = e_link if K.hitch_aware else e_d
        sref = None
        if K.gap_nmpc:
            # spacing in the NMPC: target progress = leader's predicted
            # progress minus the arc lag that keeps the chord at d0
            if K.gap_anchor:
                # current offset from the link sensor (local, never stale);
                # the plan only supplies how the gap should evolve
                offset = e_meas
            else:
                if K.hitch_aware:
                    d_arc = _arc_gap_hitch(buf, latest["X"], latest["Y"],
                                           latest.get("course", 0.0), L0,
                                           K.hitch_L, K.hitch_F)
                else:
                    d_arc = _arc_gap(buf, latest["X"], latest["Y"], K.d0)
                offset = buf.length + since - d_arc - sF
            sref = offset + sL_inc
            v_h_F = vL_plan
            vrefF = float(vL_plan[0])
            s_pred = sF + (ZF[6, 1:] if ZF is not None else np.cumsum(T * vL_plan))
        else:
            v_ff = (vL_plan[0] if age > 0 else latest["v"]) if K.feedforward else 0.0
            v_raw = v_ff + K.Kp * e_meas + K.Ki * (integ + e_meas * T)
            vrefF = float(np.clip(v_raw, 0.0, P.vmax))
            if vrefF == v_raw:             # conditional integration (anti-windup)
                integ += e_meas * T
            v_h_F = np.full(Np, vrefF)
            v_pred = np.maximum(ZF[2, 1:], 0.1) if ZF is not None else np.full(Np, 1.0)
            s_pred = sF + np.cumsum(T * v_pred)
        if K.follower_preview:
            rho_hF = _rho_lookup(fpath, s_pred, plan_s,
                                 rhoL_plan if K.leader_plan else None)
        else:
            rho_hF = np.full(Np, rhoF)
        zF = np.array([eyF, epsiF, *vF_est] + ([0.0] if K.gap_nmpc else []))
        t0 = time.perf_counter()
        if follower_policy is not None:
            from . import rl
            rho_prev = _rho_lookup(fpath, sF + rl.PREVIEW_M, plan_s,
                                   rhoL_plan if K.leader_plan else None)
            vL_now, aL_now = rl.leader_terms(vL_plan, T)
            obs = rl.build_obs(eyF, epsiF, vF_est, e_meas, vL_now, aL_now,
                               rho_prev, uF, C)
            uF = rl.u_from_action(np.asarray(follower_policy(obs)), C)
        else:
            uF, ZF = mpcF.solve(zF, rho_hF, v_h_F, uF, sref)
        solve_ms.append(1e3 * (time.perf_counter() - t0))

        # ---------------- true errors for the log
        hLt, _, eyLt, psirLt, _, _ = path.project(xL[0], xL[1], hLt)
        hFt, _, eyFt, psirFt, _, _ = tbuf.project(xF[0], xF[1], hFt)
        log.append(dict(t=t,
                        xL=xL[0], yL=xL[1], vL=xL[3], eyL=eyLt,
                        epsiL=wrap(xL[2] - psirLt), evL=vrefL - xL[3],
                        aL=uL[0], dL=uL[1],
                        xF=xF[0], yF=xF[1], vF=xF[3], eyF=eyFt,
                        epsiF=wrap(xF[2] - psirFt), evF=xL[3] - xF[3],
                        ed=e_d, e_link=e_link, f_link=f_link,
                        aF=uF[0], dF=uF[1], vrefF=vrefF, age=age,
                        est_posF=np.hypot(XFm - xF[0], YFm - xF[1]),
                        est_psiF=wrap(psiFm - xF[2]), est_vyF=vF_est[1] - xF[4]))

        # ---------------- advance plants
        if K.link_rigid:
            xL, xF, lam = plant.step_pair_rigid(xL, uL, xF, uF, P, T, K.dt_plant,
                                                L0, K.hitch_L, K.hitch_F)
            log[-1]["f_link"] = lam        # mean bar tension over this period
        elif K.link_k > 0:
            xL, xF = plant.step_pair(xL, uL, xF, uF, P, T, K.dt_plant,
                                     K.link_k, c_link, L0, K.hitch_L, K.hitch_F)
        else:
            xL = plant.step(xL, uL, P, T, K.dt_plant)
            xF = plant.step(xF, uF, P, T, K.dt_plant)
        n += 1
        if verbose and len(log) % 100 == 0:
            print(f"t={t:5.1f}s sL={sL:6.1f}/{path.length:.1f} "
                  f"eyL={eyL:+.3f} eyF={eyF:+.3f} ed={e_d:+.3f}")

    out = {k: np.array([r[k] for r in log]) for k in log[0]}
    out["solve_ms"] = np.array(solve_ms)
    out["path"] = path.arr.copy()
    return out
