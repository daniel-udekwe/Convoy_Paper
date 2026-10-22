"""Dynamic bicycle model (Eqs. 1-5) used as the simulated plant.

State  x = [X, Y, psi, vx, vy, r, a]   (global position of the CoM, yaw,
           body velocities, yaw rate, delivered longitudinal acceleration)
Input  u = [a_cmd, delta]
Linear tyres, two tyres per axle (hence the factor 2 in Eqs. 1-3).
"""
import numpy as np

from .params import Vehicle


def derivatives(x, u, P: Vehicle):
    X, Y, psi, vx, vy, r, a = x
    a_cmd, delta = u
    v = max(vx, P.v_floor)
    alpha_f = delta - (vy + P.lf * r) / v
    alpha_r = -(vy - P.lr * r) / v
    Ff = 2.0 * P.Cf * alpha_f
    Fr = 2.0 * P.Cr * alpha_r
    return np.array([
        vx * np.cos(psi) - vy * np.sin(psi),
        vx * np.sin(psi) + vy * np.cos(psi),
        r,
        a + vy * r,
        (Ff * np.cos(delta) + Fr) / P.m - vx * r,
        (P.lf * Ff * np.cos(delta) - P.lr * Fr) / P.Iz,
        (a_cmd - a) / P.tau,
    ])


def step(x, u, P: Vehicle, T: float, dt: float):
    """Integrate the plant over one control period with RK4 sub-steps."""
    n = max(1, int(round(T / dt)))
    h = T / n
    for _ in range(n):
        k1 = derivatives(x, u, P)
        k2 = derivatives(x + 0.5 * h * k1, u, P)
        k3 = derivatives(x + 0.5 * h * k2, u, P)
        k4 = derivatives(x + h * k3, u, P)
        x = x + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


# ------------------------------------------------------------- coupled pair
def hitch_points(xL, xF, hL, hF):
    """Hitch positions: hL behind the leader's CoM, hF ahead of the follower's."""
    pL = np.array([xL[0] - hL * np.cos(xL[2]), xL[1] - hL * np.sin(xL[2])])
    pF = np.array([xF[0] + hF * np.cos(xF[2]), xF[1] + hF * np.sin(xF[2])])
    return pL, pF


def link_force(xL, xF, k, c, L0, hL, hF):
    """Spring-damper link between the hitches (contribution C3).

    Returns (f, e, pL, pF): f > 0 is tension (link stretched), e the unit
    vector from the follower's hitch to the leader's.
    """
    pL, pF = hitch_points(xL, xF, hL, hF)
    d = pL - pF
    L = np.hypot(*d)
    e = d / max(L, 1e-9)

    def hitch_vel(x, off):             # off = hitch offset along body x
        vx_g = x[3] * np.cos(x[2]) - x[4] * np.sin(x[2])
        vy_g = x[3] * np.sin(x[2]) + x[4] * np.cos(x[2])
        rx, ry = off * np.cos(x[2]), off * np.sin(x[2])
        return np.array([vx_g - x[5] * ry, vy_g + x[5] * rx])

    Ldot = e @ (hitch_vel(xL, -hL) - hitch_vel(xF, hF))
    return k * (L - L0) + c * Ldot, e, pL, pF


def _apply(x, F, off, P):
    """Accelerations on one vehicle from a global force F at body offset off."""
    c, s = np.cos(x[2]), np.sin(x[2])
    fx, fy = F @ (c, s), F @ (-s, c)        # body frame
    moment = off * fy                       # hitch on the body x axis
    out = np.zeros(7)
    out[3], out[4], out[5] = fx / P.m, fy / P.m, moment / P.Iz
    return out


def step_pair(xL, uL, xF, uF, P: Vehicle, T, dt, k, c, L0, hL, hF):
    """Integrate leader and follower joined by the link over one period."""
    def f(z):
        a, b = z[:7], z[7:]
        fk, e, _, _ = link_force(a, b, k, c, L0, hL, hF)
        F_on_F = fk * e                     # tension pulls the follower forward
        da = derivatives(a, uL, P) + _apply(a, -F_on_F, -hL, P)
        db = derivatives(b, uF, P) + _apply(b, F_on_F, hF, P)
        return np.concatenate([da, db])

    z = np.concatenate([xL, xF])
    n = max(1, int(round(T / dt)))
    h = T / n
    for _ in range(n):
        k1 = f(z); k2 = f(z + 0.5 * h * k1); k3 = f(z + 0.5 * h * k2); k4 = f(z + h * k3)
        z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z[:7], z[7:]


# ------------------------------------------------------------- rigid drawbar
def _hitch_kin(x, xd, off):
    """Global position, velocity and acceleration of a body-fixed point.

    x: state, xd: its derivative, off: offset along the body x axis.
    """
    c, s = np.cos(x[2]), np.sin(x[2])
    r, rd = x[5], xd[5]
    R = np.array([[c, -s], [s, c]])
    rh = off * np.array([c, s])                       # global offset vector
    v_cm = R @ x[3:5]
    a_cm = R @ (xd[3:5] + r * np.array([-x[4], x[3]]))
    perp = np.array([-rh[1], rh[0]])
    p = x[:2] + rh
    v = v_cm + r * perp
    a = a_cm + rd * perp - r * r * rh
    return p, v, a


def step_pair_rigid(xL, uL, xF, uF, P: Vehicle, T, dt, L0, hL, hF,
                    alpha=20.0, beta=20.0):
    """Leader and follower joined by a rigid drawbar with pin joints.

    The bar holds the hitch-to-hitch distance at L0 through a constraint
    force lam along the bar (lam > 0 = tension). lam is solved at every
    derivative evaluation so that the constraint acceleration obeys
    g'' + 2 alpha g' + beta^2 g = 0 (Baumgarte stabilisation), where
    g = |pL - pF| - L0. Returns the new states and the mean bar tension.
    """
    def accel_terms(z, lam):
        a, b = z[:7], z[7:]
        pL, pF = hitch_points(a, b, hL, hF)
        d = pL - pF
        L = np.hypot(*d)
        e = d / max(L, 1e-9)
        F_on_F = lam * e
        da = derivatives(a, uL, P) + _apply(a, -F_on_F, -hL, P)
        db = derivatives(b, uF, P) + _apply(b, F_on_F, hF, P)
        _, vLh, aLh = _hitch_kin(a, da, -hL)
        _, vFh, aFh = _hitch_kin(b, db, hF)
        vr, ar = vLh - vFh, aLh - aFh
        g, gd = L - L0, e @ vr
        gdd = e @ ar + (vr @ vr - gd ** 2) / max(L, 1e-9)
        return np.concatenate([da, db]), g, gd, gdd

    tensions = []

    def f(z):
        _, g, gd, gdd0 = accel_terms(z, 0.0)
        _, _, _, gdd1 = accel_terms(z, 1.0)
        target = -2 * alpha * gd - beta ** 2 * g
        lam = (target - gdd0) / (gdd1 - gdd0)     # g'' is affine in lam
        zd, *_ = accel_terms(z, lam)
        tensions.append(lam)
        return zd

    z = np.concatenate([xL, xF])
    n = max(1, int(round(T / dt)))
    h = T / n
    for _ in range(n):
        k1 = f(z); k2 = f(z + 0.5 * h * k1); k3 = f(z + 0.5 * h * k2); k4 = f(z + h * k3)
        z = z + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return z[:7], z[7:], float(np.mean(tensions))
