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

    def update_with_multi(self, solver: EquilibriumSolver, theta_cmd: np.ndarray, A_map: Dict[int, np.ndarray], robot_est: RobotArm, theta_init_eq_pred: Optional[np.ndarray], kp_lim: Tuple[float]) -> np.ndarray:
        self.predict()
        g, H, theta_eq, c_bingham = self._grad_hess_multi(solver=solver, x0=self.x, theta_cmd=theta_cmd, A_map=A_map, robot_est=robot_est, theta_init=theta_init_eq_pred)
        Sinv = -H
        w = np.linalg.eigvalsh(0.5 * (Sinv + Sinv.T))
        lam_min = float(np.min(w))
        if lam_min <= self.eps_def:
            Sinv = Sinv + ((self.eps_def - lam_min) + 1e-12) * np.eye(Sinv.shape[0])

        # S = np.linalg.pinv(Sinv, rcond=1e-12)
        # m = self.x + S @ g

        Pinv = np.linalg.pinv(self.P, rcond=1e-12)
        # J_post = Pinv + Sinv
        # P_post = np.linalg.pinv(J_post, rcond=1e-12)
        # h_post = Pinv @ self.x + Sinv @ m
        # x_post = P_post @ h_post

        # 等価な一行更新（丸め誤差的にも綺麗）
        # lam = ||x - x_prev|| を小さくする regularize factor
        lam = 1e-6
        P_post = np.linalg.pinv(Pinv + Sinv + lam*np.eye(Sinv.shape[0]), rcond=1e-12)
        x_post = self.x + P_post @ g

        ## clip
        x_post = np.clip(x_post, np.log(kp_lim[0]), np.log(kp_lim[1]))

        self.P = 0.5 * (P_post + P_post.T)
        self.x = x_post
        self.last_theta_eq = theta_eq.copy()
        return theta_eq
