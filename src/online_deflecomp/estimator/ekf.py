from typing import Dict, Optional, Tuple
import numpy as np
from ..utils.robot import RobotArm
from ..utils.bingham import BinghamUtils
from ..controller.equilibrium import EquilibriumSolver

class MultiFrameWeirdEKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, eps_def: float = 1e-6) -> None:
        self.x = x0.copy()
        self.P = P0.copy()
        self.Q = Q.copy()
        self.eps_def = float(eps_def)
        self.last_theta_eq: Optional[np.ndarray] = None

    def predict(self) -> None:
        self.P = self.P + self.Q

    @staticmethod
    def _common_terms(robot: RobotArm, theta_eq: np.ndarray, theta_cmd: np.ndarray, k_diag: np.ndarray):
        dG = robot.d_tau_gravity(theta_eq)
        K = np.diag(k_diag)
        J_q = dG + K
        J_x = np.diag(k_diag * (theta_eq - theta_cmd))
        return J_q, J_x

    def _accumulate_frame_terms(self, robot: RobotArm, theta_eq: np.ndarray, fid: int, A_f: np.ndarray, J_q: np.ndarray, J_x: np.ndarray):
        z_f = robot.frame_quaternion_wxyz_base(theta_eq, fid)
        Qz_f = BinghamUtils.qmat_from_quat_wxyz(z_f)
        # NOTE: z_f represents frame->WORLD (body-to-world) rotation.
        # Qz is G(z) for WORLD angular velocity; keep A_f built as simple_bingham_unit(g_in_frame, g_in_world)
        # so v_f = Qz^T (A_f z_f) and H0_f = 0.5 M_f^T A_f M_f remain consistent.
        J_w_f = robot.frame_angular_jacobian_world(theta_eq, fid)

        v_f = Qz_f.T @ (A_f @ z_f)
        u_f = J_w_f.T @ v_f

        X = np.linalg.pinv(J_q, rcond=1e-12) @ J_x
        M_f = Qz_f @ (J_w_f @ X)
        H0_f = 0.5 * (M_f.T @ (A_f @ M_f))
        MtM_f = M_f.T @ M_f
        return u_f, H0_f, MtM_f

    def _stabilize_hessian(self, H0_total: np.ndarray, MtM_total: np.ndarray) -> np.ndarray:
        H0s = 0.5 * (H0_total + H0_total.T)
        wH = np.linalg.eigvalsh(H0s)
        lam_max_H = float(np.max(wH)) if wH.size > 0 else 0.0
        if lam_max_H <= -self.eps_def:
            return H0s

        Bs = 0.5 * (MtM_total + MtM_total.T)
        wB = np.linalg.eigvalsh(Bs)
        lam_max_B = float(np.max(wB)) if wB.size > 0 else 0.0

        if lam_max_B <= 1e-12:
            return H0s - (lam_max_H + self.eps_def) * np.eye(H0s.shape[0])

        c = -2.0 * (lam_max_H + self.eps_def) / lam_max_B
        return H0s + 0.5 * c * MtM_total, c

    def _grad_hess_multi(self, solver: EquilibriumSolver, x0: np.ndarray, theta_cmd: np.ndarray, A_map: Dict[int, np.ndarray], robot_est: RobotArm, theta_init: Optional[np.ndarray]):
        n = x0.size
        k_diag = np.exp(x0)

        theta_eq = solver.solve(robot=robot_est, theta_cmd=theta_cmd, kp_vec=k_diag, theta_init=theta_init)
        J_q, J_x = self._common_terms(robot_est, theta_eq, theta_cmd, k_diag)

        u_total = np.zeros(n, dtype=float)
        H0_total = np.zeros((n, n), dtype=float)
        MtM_total = np.zeros((n, n), dtype=float)

        for fid, A_f in A_map.items():
            u_f, H0_f, MtM_f = self._accumulate_frame_terms(robot_est, theta_eq, fid, A_f, J_q, J_x)
            u_total += u_f
            H0_total += H0_f
            MtM_total += MtM_f

        y = np.linalg.pinv(J_q.T, rcond=1e-12) @ u_total
        g = -(J_x.T @ y)
        H, c_bingham = self._stabilize_hessian(H0_total, MtM_total)
        return g, H, theta_eq, c_bingham

    def update_with_multi(
        self,
        solver: EquilibriumSolver,
        theta_cmd: np.ndarray,
        A_map: Dict[int, np.ndarray],
        robot_est: RobotArm,
        theta_init_eq_pred: Optional[np.ndarray],
        kp_lim: Tuple[float],
        theta_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Multi-frame EKF update with Bingham observations.
        + Command-sensitivity regularization:
            add  lambda * || J_cmd(x) * dx ||^2  to the LS objective,
            where  J_cmd = d/dx [ g(Kp) ]  with x = log diag(Kp).
        Here, g(Kp) is the "update term" in theta_cmd = theta_ref + g(Kp).
        With the current command rule  g(Kp) = tau_g(theta_ref) / Kp  (elementwise),
        we have  d g / d x = - diag( g(Kp) ).

        Args:
            theta_ref: (optional) reference posture used to compute tau_g(theta_ref).
                    If not provided, we approximate tau_g at theta_init_eq_pred
                    to form g(Kp) ≈ tau_g(theta_init)/Kp.
        """
        # Time update
        self.predict()

        # Measurement (Bingham) gradient/Hessian, equilibrium theta_eq
        g, H, theta_eq, _ = self._grad_hess_multi(
            x0=self.x,
            solver=solver,
            theta_cmd=theta_cmd,
            A_map=A_map,
            robot_est=robot_est,
            theta_init=theta_init_eq_pred,
        )

        # Information matrix from Bingham block (positive semidefinite)
        Sinv = -H

        # ===== Command-sensitivity regularization =====
        # State is x = log(kp).  kp_vec = exp(x).
        kp_vec = np.exp(self.x)
        kp_safe = np.maximum(kp_vec, 1e-12)

        if theta_ref is not None:
            # exact: g(Kp) = theta_cmd - theta_ref
            d_cmd = (theta_cmd - theta_ref).astype(float)
        else:
            # fallback (no API change needed at call site):
            # approximate tau_g at theta_init_eq_pred (or theta_cmd as last resort)
            theta_for_tau = theta_init_eq_pred if theta_init_eq_pred is not None else theta_cmd
            tau_g_approx = robot_est.tau_gravity(theta_for_tau)
            d_cmd = (tau_g_approx / kp_safe).astype(float)

        # J_cmd = d g / d x = -diag(d_cmd)  because d/dx (1/exp(x)) = -1
        J_cmd = -np.diag(d_cmd)

        # Scale-aware weight (dimensionless). Keep small by default.
        # You can tune alpha if you want stronger damping on the update term.
        alpha = 1e+1
        scale = float(np.linalg.norm(d_cmd))
        lambda_cmd = alpha / max(1.0, scale)

        # Add lambda * J_cmd^T J_cmd  (PSD) to the information matrix.
        # This damps the part of dx that would amplify the command update term.
        Sinv = Sinv + lambda_cmd * (J_cmd.T @ J_cmd)
        # ===== End of regularization =====

        # Symmetrize & minimum-eigenvalue floor for numerical safety
        w = np.linalg.eigvalsh(0.5 * (Sinv + Sinv.T))
        lam_min = float(np.min(w))
        if lam_min <= self.eps_def:
            Sinv = Sinv + ((self.eps_def - lam_min) + 1e-12) * np.eye(Sinv.shape[0])

        # Prior fusion (information form) with small Tikhonov (same as before)
        Pinv = np.linalg.pinv(self.P, rcond=1e-12)
        lam = 1e-6
        P_post = np.linalg.pinv(Pinv + Sinv + lam * np.eye(Sinv.shape[0]), rcond=1e-12)

        # State update
        x_post = self.x + P_post @ g

        # Clip to kp limits in log-domain (same as before)
        x_post = np.clip(x_post, np.log(kp_lim[0]), np.log(kp_lim[1]))

        # Commit
        self.P = 0.5 * (P_post + P_post.T)
        self.x = x_post
        self.last_theta_eq = theta_eq.copy()
        return theta_eq
