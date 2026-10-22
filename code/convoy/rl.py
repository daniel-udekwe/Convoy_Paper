"""Reinforcement-learning follower (model-free baseline, SAC).

The RL policy replaces only the follower's NMPC and sees the same
information the robust gap-aware NMPC uses: its own path errors and
speeds, the gap error, the leader's speed and acceleration from the shared
plan, and curvature along the buffer (extended with the leader's planned
path). The observation is built by `build_obs`, which both the training
environment and `convoy.simulate(..., follower_policy=...)` call, so a
trained policy runs unchanged through every evaluation condition.

Training replays leader recordings on random paths (never the two test
scenarios), so no NMPC is solved while training.
"""
import numpy as np

from . import plant
from .nmpc import NMPC
from .params import ConvoyConfig, MPCConfig, Vehicle
from .paths import Track, wrap

PREVIEW_M = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])   # curvature look-ahead
OBS_DIM = 9 + len(PREVIEW_M) + 2


def build_obs(eyF, epsiF, vF_est, e_d, vL, aL, rho_prev, u_prev, C: MPCConfig):
    """Normalised observation vector (shared by training and evaluation)."""
    return np.concatenate([
        [eyF / 0.5, epsiF / 0.3, vF_est[0] / 5.0, vF_est[1] / 0.5,
         vF_est[2] / 0.5, vF_est[3] / 1.0, e_d / 1.0, (vL - vF_est[0]) / 1.0,
         aL / 1.0],
        np.clip(rho_prev / 0.2, -5, 5),
        action_from_u(u_prev, C),
    ]).astype(np.float32)


def u_from_action(act, C: MPCConfig):
    """[-1, 1]^2 -> (a_cmd, delta) inside the same actuator limits as the NMPC."""
    act = np.clip(act, -1, 1)
    a = C.a_min + (act[0] + 1) * 0.5 * (C.a_max - C.a_min)
    return np.array([a, act[1] * C.d_max])


def action_from_u(u, C: MPCConfig):
    return np.array([2 * (u[0] - C.a_min) / (C.a_max - C.a_min) - 1,
                     u[1] / C.d_max])


def leader_terms(vL_plan, T):
    """Leader speed now and acceleration, from the (age-shifted) plan."""
    return float(vL_plan[0]), float((vL_plan[1] - vL_plan[0]) / T)


# ---------------------------------------------------------------- recording
def record_leader(path: Track, P=Vehicle(), C=MPCConfig(), v0=1.0, t_max=120.0):
    """Drive the leader with its NMPC and record what it would broadcast."""
    from .convoy import _horizon        # local import avoids a cycle
    T, Np = C.T, C.Np
    mpc = NMPC(P, C)
    d = path.arr
    x = np.array([d[0, 1], d[0, 2], d[0, 3], v0, 0, 0, 0], float)
    u = np.zeros(2)
    Z, h = None, 0
    rec = {k: [] for k in ("x", "course", "rho", "s_inc", "v_plan", "rho_plan")}
    n = 0
    while n * T < t_max:
        h, s, ey, psir, rho, vref = path.project(x[0], x[1], h)
        if s >= path.length - 1.0:
            break
        rho_h, v_h = _horizon(path, s, Z, T, True, rho, vref, Np)
        u, Z = mpc.solve(np.array([ey, wrap(x[2] - psir), *x[3:7]]), rho_h, v_h, u)
        slip = np.arctan2(x[4], max(x[3], 0.5))
        rec["x"].append(x.copy())
        rec["course"].append(x[2] + slip)
        rec["rho"].append(x[5] / max(x[3], 0.5))
        rec["s_inc"].append(np.cumsum(T * Z[2, 1:] * np.cos(Z[1, 1:])))
        rec["v_plan"].append(Z[2, 1:].copy())
        rec["rho_plan"].append(rho_h.copy())
        x = plant.step(x, u, P, T, 0.005)
        n += 1
    return {k: np.array(v) for k, v in rec.items()}


# ---------------------------------------------------------------- environment
def make_env_class():
    import gymnasium as gym

    class FollowerEnv(gym.Env):
        """Follower driving behind a replayed leader recording (ideal V2V)."""

        metadata = {"render_modes": []}

        def __init__(self, recordings, seed=0, P=Vehicle(), C=MPCConfig(),
                     K=ConvoyConfig(), dt=0.01):
            from .convoy import _rho_lookup, _with_plan_extension
            self._lookup, self._extend = _rho_lookup, _with_plan_extension
            self.recs, self.P, self.C, self.K, self.dt = recordings, P, C, K, dt
            self.rng = np.random.default_rng(seed)
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
            self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

        # -- helpers
        def _msg(self, n):
            r = self.rec
            x = r["x"][n]
            return dict(n=n, X=x[0], Y=x[1], course=r["course"][n], rho=r["rho"][n],
                        v=x[3], s_inc=r["s_inc"][n], v_plan=r["v_plan"][n],
                        rho_plan=r["rho_plan"][n])

        def _observe(self):
            msg = self._msg(self.n)
            fpath = self._extend(self.buf, msg)
            self.hF, sF, ey, psir, _, _ = fpath.project(self.xF[0], self.xF[1], self.hF)
            epsi = wrap(self.xF[2] - psir)
            xL = self.rec["x"][self.n]
            e_d = np.hypot(xL[0] - self.xF[0], xL[1] - self.xF[1]) - self.K.d0
            plan_s = self.buf.length + msg["s_inc"]
            rho_prev = self._lookup(fpath, sF + PREVIEW_M, plan_s, msg["rho_plan"])
            vL, aL = leader_terms(msg["v_plan"], self.C.T)
            self.err = (ey, epsi, e_d)
            return build_obs(ey, epsi, self.xF[3:7], e_d, vL, aL, rho_prev,
                             self.u, self.C)

        # -- gym API
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            if seed is not None:               # reproducible episodes
                self.rng = np.random.default_rng(seed)
            self.rec =self.recs[self.rng.integers(len(self.recs))]
            x0 = self.rec["x"][0]
            psi0, d0 = x0[2], self.K.d0
            self.xF = np.array([x0[0] - d0 * np.cos(psi0), x0[1] - d0 * np.sin(psi0),
                                psi0, x0[3], 0, 0, 0], float)
            # small random start perturbations, for robustness
            lat = self.rng.uniform(-0.2, 0.2)
            self.xF[0] += -lat * np.sin(psi0)
            self.xF[1] += lat * np.cos(psi0)
            self.xF[2] += self.rng.uniform(-0.05, 0.05)
            self.xF[3] += self.rng.uniform(-0.3, 0.3)
            self.buf = Track()
            for lam in np.linspace(0, 1, int(d0 / 0.1), endpoint=False):
                self.buf.append(x0[0] - d0 * np.cos(psi0) * (1 - lam),
                                x0[1] - d0 * np.sin(psi0) * (1 - lam), psi0, 0.0, x0[3])
            self.n, self.hF, self.u = 0, 0, np.zeros(2)
            self._append_leader()
            return self._observe(), {}

        def _append_leader(self):
            m = self._msg(self.n)
            self.buf.append(m["X"], m["Y"], m["course"], m["rho"], m["v"])

        def step(self, action):
            u_prev = self.u
            self.u = u_from_action(action, self.C)
            self.xF = plant.step(self.xF, self.u, self.P, self.C.T, self.dt)
            self.n += 1
            self._append_leader()
            obs = self._observe()
            ey, epsi, e_d = self.err
            du = (self.u - u_prev) / np.array([0.5, 0.05])
            cost = (0.5 * (ey / 0.1) ** 2 + 0.2 * (epsi / 0.05) ** 2
                    + 0.5 * (e_d / 0.2) ** 2 + 0.05 * du @ du)
            reward = 1.0 - 0.1 * min(cost, 10.0)
            terminated = bool(abs(ey) > 2.0 or abs(e_d) > 4.0)
            if terminated:
                reward = -20.0
            truncated = self.n >= len(self.rec["x"]) - 1
            return obs, float(reward), terminated, bool(truncated), {}

    return FollowerEnv
