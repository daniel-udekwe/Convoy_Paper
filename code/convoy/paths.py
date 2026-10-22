"""Reference paths and the growable path buffer.

A Track stores samples (s, x, y, psi, rho, v) ordered by arc length s and
supports projecting a point onto it. The leader's reference path and the
follower's FIFO buffer are both Tracks; the buffer simply grows as the
leader drives.
"""
import numpy as np
from scipy.interpolate import CubicSpline


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Track:
    def __init__(self, capacity=200000):
        self.cap = capacity
        self.n = 0
        self.data = np.zeros((capacity, 6))  # s, x, y, psi, rho, v

    @classmethod
    def from_arrays(cls, s, x, y, psi, rho, v):
        t = cls(len(s) + 10)
        t.data[: len(s)] = np.column_stack([s, x, y, psi, rho, v])
        t.n = len(s)
        return t

    def append(self, x, y, psi, rho, v):
        if self.n == 0:
            s = 0.0
        else:
            px, py = self.data[self.n - 1, 1:3]
            ds = np.hypot(x - px, y - py)
            if ds < 1e-3:          # standing still: do not add a duplicate
                return
            s = self.data[self.n - 1, 0] + ds
        self.data[self.n] = (s, x, y, psi, rho, v)
        self.n += 1

    @property
    def arr(self):
        return self.data[: self.n]

    @property
    def length(self):
        return self.data[self.n - 1, 0]

    def project(self, X, Y, hint=0, window=200):
        """Nearest point on the polyline near index `hint`.

        Returns (idx, s, e_y, psi_ref, rho, v) with e_y positive to the left.
        """
        d = self.arr
        lo, hi = max(0, hint - window), min(self.n, hint + window)
        seg = d[lo:hi]
        i = lo + int(np.argmin((seg[:, 1] - X) ** 2 + (seg[:, 2] - Y) ** 2))
        # refine on the adjacent segment
        best = None
        for j in (i - 1, i):
            if j < 0 or j + 1 >= self.n:
                continue
            p0, p1 = d[j, 1:3], d[j + 1, 1:3]
            t_vec = p1 - p0
            L2 = t_vec @ t_vec
            if L2 < 1e-12:
                continue
            lam = np.clip(((X, Y) - p0) @ t_vec / L2, 0.0, 1.0)
            q = p0 + lam * t_vec
            dist = np.hypot(X - q[0], Y - q[1])
            if best is None or dist < best[0]:
                best = (dist, j, lam, q)
        if best is None:
            j, lam, q = i, 0.0, d[i, 1:3]
        else:
            _, j, lam, q = best
        k = min(j + 1, self.n - 1)
        s = d[j, 0] + lam * (d[k, 0] - d[j, 0])
        psi = d[j, 3] + lam * wrap(d[k, 3] - d[j, 3])
        rho = d[j, 4] + lam * (d[k, 4] - d[j, 4])
        v = d[j, 5] + lam * (d[k, 5] - d[j, 5])
        e_y = -(X - q[0]) * np.sin(psi) + (Y - q[1]) * np.cos(psi)
        return j, s, e_y, psi, rho, v

    def at_s(self, s_query):
        """Interpolate rho and v at arc lengths s_query (clamped to the ends)."""
        d = self.arr
        s = np.clip(s_query, d[0, 0], d[-1, 0])
        return np.interp(s, d[:, 0], d[:, 4]), np.interp(s, d[:, 0], d[:, 5])


# ---------------------------------------------------------------- speed profile
def speed_profile(s, rho, vmax, a_lat, a_acc, a_dec, v_start):
    v = np.minimum(vmax, np.sqrt(a_lat / np.maximum(np.abs(rho), 1e-6)))
    v[0] = min(v[0], v_start)
    for i in range(1, len(s)):          # forward pass: acceleration limit
        ds = s[i] - s[i - 1]
        v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2 * a_acc * ds))
    for i in range(len(s) - 2, -1, -1):  # backward pass: braking limit
        ds = s[i + 1] - s[i]
        v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2 * a_dec * ds))
    return v


# ---------------------------------------------------------------- scenario 1
def ideal_path(ds=0.05, v_start=1.0):
    """Parametric path of straights and arcs, in the spirit of Fig. 7a.

    Curvature jumps at each join, reproducing the 'discontinuities' the paper
    discusses. Segments: (length or angle, curvature).
    """
    segs = [("line", 6.0, 0.0),
            ("arc", np.pi, 1 / 20.0),        # left, R = 20 m
            ("arc", np.pi / 2, -1 / 10.0),   # right, R = 10 m
            ("arc", 3 * np.pi / 4, -1 / 12.0),  # right, R = 12 m
            ("line", 40.0, 0.0)]
    x, y, psi = [0.0], [-6.0], [np.pi / 2]
    rho = [0.0]
    for kind, val, k in segs:
        L = val if kind == "line" else abs(val / k)
        n = int(np.ceil(L / ds))
        h = L / n
        for _ in range(n):
            p = psi[-1] + k * h
            pm = psi[-1] + 0.5 * k * h
            x.append(x[-1] + h * np.cos(pm))
            y.append(y[-1] + h * np.sin(pm))
            psi.append(p)
            rho.append(k)
    x, y, psi, rho = map(np.array, (x, y, psi, rho))
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    v = speed_profile(s, rho, 5.0, 1.5, 0.6, 1.0, v_start)
    return Track.from_arrays(s, x, y, psi, rho, v)


# ---------------------------------------------------------------- scenario 2
# minimum radius of the clean spline is 7.5 m, inside the cart's ~3.8 m limit
# at the 0.4 rad steering bound
_SCENARIO2_WP = [(0, 0), (0, -10), (-3, -21), (-10, -28), (-18, -29),
                 (-28, -32), (-41, -30), (-56, -17), (-68, -2), (-74, 10),
                 (-68, 19), (-56, 17), (-42, 2), (-27, -12), (-12, -12)]


def min_radius(wp):
    """Minimum radius of the cubic spline through waypoints wp."""
    wp = np.asarray(wp, float)
    t = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(wp, axis=0).T))])
    sx, sy = CubicSpline(t, wp[:, 0]), CubicSpline(t, wp[:, 1])
    u = np.linspace(0, t[-1], 8000)
    dx, dy, ddx, ddy = sx(u, 1), sy(u, 1), sx(u, 2), sy(u, 2)
    rho = (dx * ddy - ddx * dy) / (dx ** 2 + dy ** 2) ** 1.5
    return 1 / np.max(np.abs(rho))


def random_waypoints(seed, n=12, r_min=7.5):
    """Random feasible waypoints for RL training paths (never the test paths).

    A heading random walk with 8-16 m steps; rejected until the spline's
    minimum radius is at least r_min.
    """
    rng = np.random.default_rng(10_000 + seed)
    while True:
        psi = rng.uniform(-np.pi, np.pi)
        pts = [(0.0, 0.0)]
        for _ in range(n - 1):
            psi += rng.uniform(-1.0, 1.0)
            L = rng.uniform(8, 16)
            pts.append((pts[-1][0] + L * np.cos(psi), pts[-1][1] + L * np.sin(psi)))
        if min_radius(pts) >= r_min:
            return pts


def gnss_like_path(seed=0, sigma=0.02, rate_hz=25.0, v_start=1.0, wp=None,
                   vcap=6.0):
    """Synthetic stand-in for the paper's GNSS-recorded path (Fig. 7b).

    A spline through hand-placed waypoints (default: scenario 2), sampled at
    25 Hz along its speed profile, with sigma = 2 cm position noise
    (RTK-grade). Heading and curvature are then re-estimated from the noisy
    samples, as they would be from a real log. Replace with a real log when
    available. vcap leaves headroom below vmax so a lagging follower can
    catch up.
    """
    if wp is None:
        wp = _SCENARIO2_WP
    wp = np.asarray(wp, float)
    t =np.concatenate([[0], np.cumsum(np.hypot(*np.diff(wp, axis=0).T))])
    sx, sy = CubicSpline(t, wp[:, 0]), CubicSpline(t, wp[:, 1])
    # dense clean geometry -> speed profile -> 25 Hz time sampling
    u = np.linspace(0, t[-1], 20000)
    xd, yd = sx(u), sy(u)
    dx, dy, ddx, ddy = sx(u, 1), sy(u, 1), sx(u, 2), sy(u, 2)
    rho_d = (dx * ddy - ddx * dy) / (dx ** 2 + dy ** 2) ** 1.5
    s_d = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xd), np.diff(yd)))])
    # cap at 6 m/s: 1 m/s headroom below vmax so a lagging follower can catch up
    v_d = speed_profile(s_d, rho_d, vcap, 2.0, 0.8, 1.2, v_start)
    # time along the path
    dt_d = np.diff(s_d) / np.maximum(0.5 * (v_d[1:] + v_d[:-1]), 0.05)
    t_d = np.concatenate([[0], np.cumsum(dt_d)])
    tt = np.arange(0, t_d[-1], 1 / rate_hz)
    rng = np.random.default_rng(seed)
    x = np.interp(tt, t_d, xd) + rng.normal(0, sigma, tt.size)
    y = np.interp(tt, t_d, yd) + rng.normal(0, sigma, tt.size)
    v = np.interp(tt, t_d, v_d)
    # re-derive geometry from the noisy log (light smoothing, as one would)
    k = 5
    ker = np.ones(k) / k
    xs = np.convolve(np.pad(x, k // 2, mode="edge"), ker, "valid")
    ys = np.convolve(np.pad(y, k // 2, mode="edge"), ker, "valid")
    s = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
    psi = np.unwrap(np.arctan2(np.gradient(ys), np.gradient(xs)))
    rho = np.gradient(psi) / np.maximum(np.gradient(s), 1e-3)
    rho = np.convolve(np.pad(rho, k // 2, mode="edge"), ker, "valid")
    keep = np.concatenate([[True], np.diff(s) > 1e-3])
    return Track.from_arrays(s[keep], xs[keep], ys[keep], psi[keep],
                             rho[keep], v[keep])
