"""NMPC on the nonlinear path-error model (cf. Eqs. 6-17).

Prediction state z = [e_y, e_psi, vx, vy, r, a]; input u = [a_cmd, delta].
Error kinematics are the exact Frenet form; Eq. 11 of the paper is its
small-angle linearisation (e_y' = vx e_psi + vy). Curvature is signed
(left turns positive); note Eq. 15 takes an absolute value, which would
lose the turn direction.

Per-step parameters: curvature rho_k and speed reference v_k over the
horizon, plus the previously applied input (for the rate penalty).
Passing constant arrays reproduces the baseline (current values only);
passing varying arrays gives curvature/speed preview (contribution C2).

gap_mode=True (contribution C1, full version): the state gains a 7th entry,
the follower's arc-length progress s along the buffered path (0 now). An
extra parameter s_ref_k gives where the follower *should* be at step k,
i.e. the leader's shared predicted progress minus the arc-length gap that
corresponds to the link length. The cost then penalises spacing directly,
and v_ref (the leader's predicted speed) is kept only as a weak secondary
term. Speed and spacing are optimised together, with no PI cascade.
"""
import casadi as ca
import numpy as np

from .params import MPCConfig, Vehicle


def _dynamics(P: Vehicle, with_s: bool):
    nz = 7 if with_s else 6
    z = ca.SX.sym("z", nz)
    u = ca.SX.sym("u", 2)
    rho = ca.SX.sym("rho")
    ey, epsi, vx, vy, r, a = [z[i] for i in range(6)]
    a_cmd, delta = u[0], u[1]
    v = ca.fmax(vx, P.v_floor)
    Ff = 2 * P.Cf * (delta - (vy + P.lf * r) / v)
    Fr = 2 * P.Cr * (-(vy - P.lr * r) / v)
    s_dot = (vx * ca.cos(epsi) - vy * ca.sin(epsi)) / (1 - rho * ey)
    rows = [
        vx * ca.sin(epsi) + vy * ca.cos(epsi),
        r - rho * s_dot,
        a + vy * r,
        (Ff * ca.cos(delta) + Fr) / P.m - vx * r,
        (P.lf * Ff * ca.cos(delta) - P.lr * Fr) / P.Iz,
        (a_cmd - a) / P.tau,
    ]
    if with_s:
        rows.append(s_dot)
    return ca.Function("f", [z, u, rho], [ca.vertcat(*rows)])


class NMPC:
    def __init__(self, P: Vehicle, C: MPCConfig, gap_mode: bool = False):
        self.P, self.C, self.gap_mode = P, C, gap_mode
        self.nz = nz = 7 if gap_mode else 6
        f = _dynamics(P, gap_mode)
        Np, h = C.Np, C.T / C.n_sub
        opti = ca.Opti()
        Z = opti.variable(nz, Np + 1)
        U = opti.variable(2, Np)
        z0 = opti.parameter(nz)
        rho = opti.parameter(Np)
        vref = opti.parameter(Np)
        sref = opti.parameter(Np)
        uprev = opti.parameter(2)
        opti.subject_to(Z[:, 0] == z0)
        q_v = C.q_v_gap if gap_mode else C.q_v
        J = 0
        for k in range(Np):
            zk = Z[:, k]
            for _ in range(C.n_sub):
                k1 = f(zk, U[:, k], rho[k])
                k2 = f(zk + h / 2 * k1, U[:, k], rho[k])
                k3 = f(zk + h / 2 * k2, U[:, k], rho[k])
                k4 = f(zk + h * k3, U[:, k], rho[k])
                zk = zk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(Z[:, k + 1] == zk)
            J += (C.q_ey * Z[0, k + 1] ** 2 + C.q_epsi * Z[1, k + 1] ** 2
                  + q_v * (vref[k] - Z[2, k + 1]) ** 2)
            if gap_mode:
                J += C.q_gap * (sref[k] - Z[6, k + 1]) ** 2
            J += C.r_a * U[0, k] ** 2 + C.r_d * U[1, k] ** 2
            du = U[:, k] - (uprev if k == 0 else U[:, k - 1])
            J += C.s_a * du[0] ** 2 + C.s_d * du[1] ** 2
        opti.subject_to(opti.bounded(C.a_min, U[0, :], C.a_max))
        opti.subject_to(opti.bounded(-C.d_max, U[1, :], C.d_max))
        opti.minimize(J)
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"print_level": 0, "sb": "yes", "max_iter": 200,
                     "tol": 1e-6, "acceptable_tol": 1e-4})
        self._solve = opti.to_function(
            "mpc", [z0, rho, vref, sref, uprev, Z, U], [U, Z])
        self.U_guess = np.zeros((2, Np))
        self.Z_guess = None

    def solve(self, z0, rho, vref, uprev, sref=None):
        Np = self.C.Np
        z0 = np.asarray(z0, float)
        if sref is None:
            sref = np.zeros(Np)
        if self.Z_guess is None:
            self.Z_guess = np.tile(z0.reshape(-1, 1), (1, Np + 1))
        Zg = self.Z_guess.copy()
        Zg[:, 0] = z0
        U, Z = self._solve(z0, rho, vref, sref, uprev, Zg, self.U_guess)
        U, Z = np.array(U), np.array(Z)
        # shift warm start (progress state is re-zeroed every step)
        self.U_guess = np.hstack([U[:, 1:], U[:, -1:]])
        Zs = np.hstack([Z[:, 1:], Z[:, -1:]])
        if self.gap_mode:
            Zs[6] -= Z[6, 1]
        self.Z_guess = Zs
        return U[:, 0], Z
