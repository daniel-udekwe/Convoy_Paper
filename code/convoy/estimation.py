"""Sensors and state estimators (contribution C5).

Sensor suite per vehicle, sampled every control period (10 Hz):
  GNSS      : X, Y position                      sigma = gnss_sigma
  odometry  : longitudinal speed vx               sigma = 0.03 m/s
  gyro      : yaw rate r + constant bias          sigma = 0.005 rad/s, bias ~ N(0, 0.003)
  accel     : longitudinal specific force (= a)   sigma = 0.05 m/s^2
There is no heading sensor (no dual-antenna GNSS), so heading must be
inferred.

Estimators
  truth : exact state (reference)
  raw   : sensors used directly; heading from integrating the (biased) gyro,
          lateral speed assumed zero
  ekf   : extended Kalman filter on [X, Y, psi, vx, vy, r, a, gyro_bias] with
          the bicycle model as process model. Its model parameters are
          deliberately mismatched (Cf, Cr x0.8, m and Iz x1.1) so it does not
          share the simulator's exact model.
"""
from dataclasses import replace

import numpy as np

from . import plant
from .params import Vehicle

SIG_ODO, SIG_GYRO, SIG_ACC, SIG_BIAS0 = 0.03, 0.005, 0.05, 0.003


class Sensors:
    def __init__(self, gnss_sigma, rng):
        self.sg = gnss_sigma
        self.rng = rng
        self.bias = rng.normal(0.0, SIG_BIAS0)

    def measure(self, x):
        n = self.rng.normal
        return np.array([x[0] + n(0, self.sg), x[1] + n(0, self.sg),
                         x[3] + n(0, SIG_ODO), x[5] + self.bias + n(0, SIG_GYRO),
                         x[6] + n(0, SIG_ACC)])


class TruthEstimator:
    def __init__(self, x0, *a, **k):
        pass

    def step(self, x_true, u_prev, z):
        return np.array(x_true, float)


class RawEstimator:
    """No fusion: GNSS position, dead-reckoned heading, sensor speeds."""

    def __init__(self, x0, P: Vehicle, T):
        self.psi = x0[2]
        self.T = T

    def step(self, x_true, u_prev, z):
        X, Y, vx, gyro, acc = z
        self.psi += gyro * self.T          # bias integrates into heading drift
        return np.array([X, Y, self.psi, vx, 0.0, gyro, acc])


class EKF:
    def __init__(self, x0, P: Vehicle, T, gnss_sigma):
        self.Pm = replace(P, Cf=0.8 * P.Cf, Cr=0.8 * P.Cr, m=1.1 * P.m,
                          Iz=1.1 * P.Iz)
        self.T = T
        self.x = np.concatenate([np.asarray(x0, float), [0.0]])
        self.P = np.diag([0.05, 0.05, 0.02, 0.1, 0.1, 0.05, 0.1, SIG_BIAS0]) ** 2
        # process noise per step: model error on the dynamic states
        self.Q = np.diag([1e-3, 1e-3, 2e-3, 0.02, 0.05, 0.03, 0.1, 1e-4]) ** 2
        self.R = np.diag([gnss_sigma, gnss_sigma, SIG_ODO, SIG_GYRO, SIG_ACC]) ** 2
        self.H = np.zeros((5, 8))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 3] = 1.0
        self.H[3, 5] = self.H[3, 7] = 1.0   # gyro = r + bias
        self.H[4, 6] = 1.0
        self.first = True

    def _f(self, x, u):
        xn = plant.step(x[:7], u, self.Pm, self.T, 0.01)
        return np.concatenate([xn, [x[7]]])

    def step(self, x_true, u_prev, z):
        if not self.first:                 # predict with the last applied input
            x = self.x
            fx = self._f(x, u_prev)
            F = np.zeros((8, 8))
            eps = np.array([1e-4, 1e-4, 1e-5, 1e-4, 1e-4, 1e-5, 1e-4, 1e-6])
            for i in range(8):
                dx = np.zeros(8)
                dx[i] = eps[i]
                F[:, i] = (self._f(x + dx, u_prev) - fx) / eps[i]
            self.x = fx
            self.P = F @ self.P @ F.T + self.Q
        self.first = False
        # update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        Kg = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + Kg @ y
        self.P = (np.eye(8) - Kg @ self.H) @ self.P
        return self.x[:7].copy()


def make_estimator(kind, x0, P, T, gnss_sigma):
    if kind == "raw":
        return RawEstimator(x0, P, T)
    if kind == "ekf":
        return EKF(x0, P, T, gnss_sigma)
    return TruthEstimator(x0)
