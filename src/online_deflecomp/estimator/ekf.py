from typing import Dict, Optional, Tuple
import numpy as np
from ..utils.robot import RobotArm
from ..utils.bingham import BinghamUtils
from ..controller.equilibrium import EquilibriumSolver

# --- helpers for square-root (QR) EKF ---
def _sym(A: np.ndarray) -> np.ndarray:
    return 0.5 * (A + A.T)

def _chol_psd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Return upper-triangular R such that R.T @ R ≈ A (PSD with small floor).
    """
    A = _sym(A.astype(float))
    try:
        L = np.linalg.cholesky(A)  # A = L @ L.T
        return L.T                 # R = upper
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(A)
        w = np.maximum(w, eps)
        A_spd = (V * w) @ V.T
        L = np.linalg.cholesky(A_spd)
        return L.T

def _qrr_upper(A: np.ndarray) -> np.ndarray:
    """
    Return the R from reduced QR of A (upper-triangular).
    """
    _, R = np.linalg.qr(A, mode="reduced")
    return R

def _solve_upper(R: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve R x = b for upper-triangular R.
    """
    return np.linalg.solve(R, b)


class MultiFrameWeirdEKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, eps_def: float = 1e-6) -> None:
        self.x = x0.copy()
        # keep square-root covariance
        self.R = _chol_psd(P0)              # P0 = R.T @ R
        self.P = self.R.T @ self.R          # maintain for compatibility
        self.Q = Q.copy()
        self.eps_def = float(eps_def)
        self.last_theta_eq: Optional[np.ndarray] = None

    def predict(self) -> None:
        # Square-root prediction for random-walk: P+Q via qrr([R; chol(Q)])
        Uq = _chol_psd(self.Q)
        A = np.vstack([self.R, Uq])
        self.R = _qrr_upper(A)
        self.P = self.R.T @ self.R  # keep compatibility

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
            return H0s, 0.

        Bs = 0.5 * (MtM_total + MtM_total.T)
        wB = np.linalg.eigvalsh(Bs)
        lam_max_B = float(np.max(wB)) if wB.size > 0 else 0.0

        if lam_max_B <= 1e-12:
            return H0s - (lam_max_H + self.eps_def) * np.eye(H0s.shape[0]), 0.

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

    def update_with_multi(self, solver: EquilibriumSolver, theta_cmd: np.ndarray, A_map: Dict[int, np.ndarray], robot_est: RobotArm, theta_init_eq_pred: Optional[np.ndarray], kp_lim: Optional[Tuple[float]] = None) -> np.ndarray:
        if kp_lim is None:
            kp_lim = (1e-10, 2000)
        
        self.predict()
        g, H, theta_eq, c_bingham = self._grad_hess_multi(solver=solver, x0=self.x, theta_cmd=theta_cmd, A_map=A_map, robot_est=robot_est, theta_init=theta_init_eq_pred)
        Sinv = -H
        w = np.linalg.eigvalsh(0.5 * (Sinv + Sinv.T))
        print("eigval of Sinv = ", w)
        lam_min = float(np.min(w))
        if lam_min <= self.eps_def:
            Sinv = Sinv + ((self.eps_def - lam_min) + 1e-12) * np.eye(Sinv.shape[0])

        print("eigval of Sinv shifted = ", np.linalg.eigh(Sinv)[0])
        # --- Square-Root (QR) update (information form) ---
        # Sinv = S.T @ S (SPD by stabilization)
        S = _chol_psd(Sinv, eps=1e-12)
        y = np.linalg.solve(S.T, g.reshape(-1, 1))  # (n x 1), S.T @ y = g
        n = self.R.shape[0]
        # information square-root of Pinv: W = R^{-T}
        Rinv = _solve_upper(self.R, np.eye(n))      # R @ Rinv = I  -> Rinv = R^{-1}
        W = Rinv.T
        # small Tikhonov regularization (consistent with previous impl)
        # lam = 0.
        # if lam > 0.0:
        #     L = np.sqrt(lam) * np.eye(n)
        #     A = np.vstack([W, S, L])
        #     b = np.vstack([np.zeros((n, 1)), y, np.zeros((n, 1))])
        # else:
        A = np.vstack([W, S])
        b = np.vstack([np.zeros((n, 1)), y])
        # QR least-squares
        Q, Rbar = np.linalg.qr(A, mode="reduced")   # A = Q Rbar
        bbar = Q.T @ b
        dx = _solve_upper(Rbar, bbar)               # solves (Pinv + Sinv + lam I) dx = g
        x_post = (self.x.reshape(-1, 1) + dx).reshape(-1)
        # posterior covariance square-root: R_post = Rbar^{-1}
        R_post = _solve_upper(Rbar, np.eye(n))
        self.R = R_post
        self.P = self.R.T @ self.R  # keep compatibility

        ## clip
        x_post = np.clip(x_post, np.log(kp_lim[0]), np.log(kp_lim[1]))

        print("x_post = ", x_post)

        self.x = x_post
        self.last_theta_eq = theta_eq.copy()
        return theta_eq
