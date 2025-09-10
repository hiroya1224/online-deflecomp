
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pinocchio as pin

from ..utils.robot import RobotArm
from ..controller.equilibrium import EquilibriumSolver

def skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3,)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=float)

@dataclass
class LagEKFConfig:
    n_tau: int = 1              # 1: shared tau, n: per-joint tau
    p0: float = 0.0             # log(kappa) initial (kappa = 1/tau)
    u0: Optional[np.ndarray] = None
    Q_u: float = 1e-6           # process noise for u (per joint, diag)
    Q_p: float = 1e-6           # process noise for p (per each tau param, diag)
    R_meas: float = 5e-2        # measurement std (rad) for gravity dir components (~ small-angle approx)
    kp_clip: Tuple[float, float] = (1e-3, 5e3)

class LagEKF:
    """
    EKF for first-order lag (gain 1) between theta_cmd and realized angles u,
    using IMU gravity directions as measurements. No finite diff; analytic Jacobians.
    State: x = [u (n), p (n_tau)]  where kappa = exp(p), tau = 1/kappa.
    """
    def __init__(self, n: int, cfg: LagEKFConfig) -> None:
        self.n = int(n)
        self.n_tau = int(cfg.n_tau) if cfg.n_tau in [1, n] else 1
        self.p = np.full((self.n_tau,), float(cfg.p0), dtype=float)
        self.u = np.zeros((self.n,), dtype=float) if cfg.u0 is None else np.asarray(cfg.u0, dtype=float).copy()

        # Covariances
        Pu = np.eye(self.n) * 1e-3
        Pp = np.eye(self.n_tau) * 1e-2
        self.P = np.block([[Pu, np.zeros((self.n, self.n_tau))],
                           [np.zeros((self.n_tau, self.n)), Pp]]).astype(float)
        self.Q = np.block([[np.eye(self.n) * float(cfg.Q_u), np.zeros((self.n, self.n_tau))],
                           [np.zeros((self.n_tau, self.n)), np.eye(self.n_tau) * float(cfg.Q_p)]]).astype(float)
        self.R_std = float(cfg.R_meas)
        self.kp_clip = tuple(cfg.kp_clip)

        # cache
        self.theta_eq_last: Optional[np.ndarray] = None

    # --- helpers ---
    def _kappa_vec(self) -> np.ndarray:
        if self.n_tau == 1:
            return np.full((self.n,), float(np.exp(self.p[0])), dtype=float)
        else:
            return np.exp(self.p)

    def _alpha_beta(self, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # alpha = exp(-kappa*dt), beta = 1 - alpha, and d_alpha/dp for Jacobian
        kappa = self._kappa_vec()
        a = np.exp(-kappa * float(max(0.0, dt)))
        b = 1.0 - a
        # dp -> dkappa = exp(p) * dp, so d_alpha/dp = (d_alpha/dkappa) * dkappa/dp = (-dt * a) * kappa
        # If shared p (n_tau==1), broadcast the same scalar derivative to all joints
        if self.n_tau == 1:
            da_dp = (-float(max(0.0, dt)) * a) * float(kappa[0])
            da_dp = np.full((self.n,), da_dp, dtype=float)
        else:
            da_dp = (-float(max(0.0, dt)) * a) * kappa
        return a, b, da_dp

    # --- EKF prediction ---
    def predict(self, theta_cmd: np.ndarray, dt: float) -> None:
        theta_cmd = np.asarray(theta_cmd, dtype=float).reshape(self.n,)
        a, b, da_dp = self._alpha_beta(dt)
        u_prev = self.u.copy()

        # state propagation
        self.u = a * self.u + b * theta_cmd
        # p is random walk
        # (no change here)

        # Jacobian F wrt x = [u; p]
        Fuu = np.diag(a)
        if self.n_tau == 1:
            # column is shared p
            Fup = (da_dp * (u_prev - theta_cmd)).reshape(self.n, 1)
        else:
            Fup = np.diag(da_dp * (u_prev - theta_cmd))
        F = np.block([[Fuu, Fup],
                      [np.zeros((self.n_tau, self.n)), np.eye(self.n_tau)]])
        self.P = F @ self.P @ F.T + self.Q

    # --- EKF update ---
    def update(self,
               imu_dirs: Dict[int, np.ndarray],
               frame_ids: List[int],
               robot: RobotArm,
               solver: EquilibriumSolver,
               kp_vec: np.ndarray,
               theta_init: Optional[np.ndarray] = None) -> None:
        """
        imu_dirs: fid -> 3 unit vector in frame coords (gravity dir measured)
        frame_ids: list of fids to use, order defines stacking in z/h/H
        kp_vec: stiffness (>=0) used inside equilibrium (shape n, in ABS units)
        """
        kp_vec = np.clip(np.asarray(kp_vec, dtype=float).reshape(self.n,), self.kp_clip[0], self.kp_clip[1])

        # 1) equilibrium at current u
        theta0 = theta_init if theta_init is not None else self.u
        try:
            theta_eq = solver.solve(robot, theta_cmd=self.u, kp_vec=kp_vec, theta_init=theta0)
        except Exception:
            theta_eq = theta0.copy()
        self.theta_eq_last = theta_eq.copy()

        # 2) Hessian of energy wrt theta: H = Kp + d_tau_gravity(theta_eq)
        Htheta = robot.d_tau_gravity(theta_eq).astype(float)
        H = Htheta + np.diag(kp_vec)

        # 3) Sensitivity of equilibrium to u: dtheta_eq/du = H^{-1} Kp
        #    Use symmetric solve for stability
        try:
            Hinv = np.linalg.pinv(H, rcond=1e-10)
        except Exception:
            Hinv = np.linalg.pinv(H + 1e-6*np.eye(H.shape[0]))
        Sens = Hinv @ np.diag(kp_vec)   # (n x n)

        # 4) Build z, h, and Jacobian Hx (stack per frame)
        zs = []
        hs = []
        Hu_list = []  # each is 3 x n
        for fid in frame_ids:
            g_meas = imu_dirs.get(fid, None)
            if g_meas is None:
                continue
            g_meas = g_meas / (np.linalg.norm(g_meas) + 1e-12)
            zs.append(g_meas)

            # predict gravity dir in frame
            g_pred = robot.gravity_dir_in_frame(theta_eq, robot.model.gravity.linear, fid)
            hs.append(g_pred)

            # angular Jacobian of frame in local coords
            # Compute jacobians at theta_eq
            pin.computeJointJacobians(robot.model, robot.data, theta_eq)
            pin.updateFramePlacements(robot.model, robot.data)
            # LOCAL frame: 6xnv -> [linear(0:3); angular(3:6)] in Pinocchio
            J6_local = pin.computeFrameJacobian(robot.model, robot.data, theta_eq, fid, pin.ReferenceFrame.LOCAL)
            J_ang_local = J6_local[3:6, :]
            # ∂g/∂theta = S(g_pred) * J_ang_local  (3 x n)
            Hu_theta = skew(g_pred) @ J_ang_local
            # chain rule to u
            Hu = Hu_theta @ Sens
            Hu_list.append(Hu)

        if not zs:
            return  # nothing to update

        z = np.concatenate(zs, axis=0).reshape(-1,)
        h = np.concatenate(hs, axis=0).reshape(-1,)
        H_u = np.concatenate(Hu_list, axis=0)  # (3m x n)

        # H wrt full state x = [u; p]  (measurement independent of p directly)
        Hx = np.block([ [H_u, np.zeros((H_u.shape[0], self.n_tau))] ])

        # residual
        y = z - h

        # Measurement covariance
        R = (self.R_std ** 2) * np.eye(y.shape[0])

        # Kalman gain & update
        S = Hx @ self.P @ Hx.T + R
        try:
            Sinv = np.linalg.pinv(S, rcond=1e-10)
        except Exception:
            Sinv = np.linalg.pinv(S + 1e-9*np.eye(S.shape[0]))
        K = self.P @ Hx.T @ Sinv

        dx = K @ y
        # apply update
        self.u = self.u + dx[:self.n]
        if self.n_tau == 1:
            self.p[0] = self.p[0] + float(dx[self.n])
        else:
            self.p = self.p + dx[self.n: self.n + self.n_tau]

        # Covariance Joseph form for symmetry
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ Hx) @ self.P @ (I - K @ Hx).T + K @ R @ K.T

    # --- accessors ---
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        # returns (u, p) with kappa=exp(p)
        return self.u.copy(), self.p.copy()

    def get_tau(self) -> np.ndarray:
        # tau = 1/kappa
        kappa = self._kappa_vec()
        return 1.0 / (kappa + 1e-12)