from typing import Dict, Optional, Tuple
import time
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
        # filled each update_with_multi() call with ms breakdowns
        self.last_timing: Optional[Dict[str, float]] = None

    def predict(self) -> None:
        # Square-root prediction for random-walk: P+Q via qrr([R; chol(Q)])
        print("self.Q: ", np.linalg.eigh(self.Q)[0])
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

    def _stabilize_hessian(self, H0_total: np.ndarray, MtM_total: np.ndarray) -> Tuple[np.ndarray, float]:
        H0s = 0.5 * (H0_total + H0_total.T)
        wH = np.linalg.eigvalsh(H0s)
        lam_max_H = float(np.max(wH)) if wH.size > 0 else 0.0
        if lam_max_H <= -self.eps_def:
            return H0s, 0.0

        Bs = 0.5 * (MtM_total + MtM_total.T)
        wB = np.linalg.eigvalsh(Bs)
        lam_max_B = float(np.max(wB)) if wB.size > 0 else 0.0

        if lam_max_B <= 1e-12:
            return H0s - (lam_max_H + self.eps_def) * np.eye(H0s.shape[0]), 0.0

        c = -2.0 * (lam_max_H + self.eps_def) / lam_max_B
        return H0s + 0.5 * c * MtM_total, c

    def _grad_hess_multi(
        self,
        solver: EquilibriumSolver,
        x0: np.ndarray,
        theta_cmd: np.ndarray,
        A_map: Dict[int, np.ndarray],
        robot_est: RobotArm,
        theta_init: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Dict[str, float]]:
        """
        Build gradient and Hessian approx (Gauss-Newton) and accumulate per-frame terms.
        Returns (g, H, theta_eq, c_bingham, timings_ms_dict).
        """
        n = x0.size
        k_diag = np.exp(x0)

        # equilibrium solve timing
        t_eq0 = time.perf_counter()
        theta_eq = solver.solve_rti(robot=robot_est, theta_cmd=theta_cmd, kp_vec=k_diag, theta_init=theta_init)
        t_eq1 = time.perf_counter()

        J_q, J_x = self._common_terms(robot_est, theta_eq, theta_cmd, k_diag)

        u_total = np.zeros(n, dtype=float)
        H0_total = np.zeros((n, n), dtype=float)
        MtM_total = np.zeros((n, n), dtype=float)

        # accumulate per-frame
        t_acc0 = time.perf_counter()
        for fid, A_f in A_map.items():
            u_f, H0_f, MtM_f = self._accumulate_frame_terms(robot_est, theta_eq, fid, A_f, J_q, J_x)
            u_total += u_f
            H0_total += H0_f
            MtM_total += MtM_f
        t_acc1 = time.perf_counter()

        # solve for adjoint y: (J_q^T) y = u_total
        t_pinv0 = time.perf_counter()
        y = np.linalg.pinv(J_q.T, rcond=1e-12) @ u_total
        t_pinv1 = time.perf_counter()

        # gradient and stabilized Hessian
        g = -(J_x.T @ y)
        t_stab0 = time.perf_counter()
        H, c_bingham = self._stabilize_hessian(H0_total, MtM_total)
        t_stab1 = time.perf_counter()

        timings = {
            "eq_solve_ms": (t_eq1 - t_eq0) * 1000.0,
            "accum_ms": (t_acc1 - t_acc0) * 1000.0,
            "pinv_JqT_ms": (t_pinv1 - t_pinv0) * 1000.0,
            "stab_eig_ms": (t_stab1 - t_stab0) * 1000.0,
        }
        return g, H, theta_eq, c_bingham, timings

    def update_with_multi(
        self,
        solver: EquilibriumSolver,
        theta_cmd: np.ndarray,
        A_map: Dict[int, np.ndarray],
        robot_est: RobotArm,
        theta_init_eq_pred: Optional[np.ndarray],
        kp_lim: Optional[Tuple[float]] = None
    ) -> np.ndarray:
        if kp_lim is None:
            kp_lim = (1e-10, 2000)

        t_all0 = time.perf_counter()

        # predict (square-root form)
        t_pred0 = time.perf_counter()
        self.predict()
        t_pred1 = time.perf_counter()

        # build gradient/Hessian (with timings from subroutine)
        g, H, theta_eq, c_bingham, t_gh = self._grad_hess_multi(
            solver=solver, x0=self.x, theta_cmd=theta_cmd, A_map=A_map, robot_est=robot_est, theta_init=theta_init_eq_pred
        )

        # information matrix from Hessian approx
        Sinv = -H
        t_eig0 = time.perf_counter()
        w = np.linalg.eigvalsh(0.5 * (Sinv + Sinv.T))
        lam_min = float(np.min(w))
        if lam_min <= self.eps_def:
            Sinv = Sinv + ((self.eps_def - lam_min) + 1e-12) * np.eye(Sinv.shape[0])
        t_eig1 = time.perf_counter()

        # Square-root (QR) update -- information form
        # Sinv = S.T @ S (SPD by stabilization)
        t_chol0 = time.perf_counter()
        print("Sinv : ", np.linalg.eigh(Sinv)[0])
        S = _chol_psd(Sinv)
        t_chol1 = time.perf_counter()

        # RHS: S.T @ y = g
        t_rhs0 = time.perf_counter()
        y = np.linalg.solve(S.T, g.reshape(-1, 1))  # (n x 1)
        t_rhs1 = time.perf_counter()

        n = self.R.shape[0]

        # information square-root of Pinv: W = R^{-T}
        t_w0 = time.perf_counter()
        Rinv = _solve_upper(self.R, np.eye(n))      # R @ Rinv = I  -> Rinv = R^{-1}
        W = Rinv.T
        t_w1 = time.perf_counter()

        # stack least-squares system and solve with QR
        A = np.vstack([W, S])
        b = np.vstack([np.zeros((n, 1)), y])

        t_qr0 = time.perf_counter()
        Qm, Rbar = np.linalg.qr(A, mode="reduced")   # A = Q Rbar
        t_qr1 = time.perf_counter()
        bbar = Qm.T @ b

        t_solve0 = time.perf_counter()
        dx = _solve_upper(Rbar, bbar)                # solves (Pinv + Sinv) dx = g
        t_solve1 = time.perf_counter()

        x_post = (self.x.reshape(-1, 1) + dx).reshape(-1)

        # posterior covariance square-root: R_post = Rbar^{-1}
        t_inv0 = time.perf_counter()
        R_post = _solve_upper(Rbar, np.eye(n))
        self.R = R_post
        self.P = self.R.T @ self.R  # keep compatibility
        t_all1 = time.perf_counter()

        # clip x (log K)
        x_post = np.clip(x_post, np.log(kp_lim[0]), np.log(kp_lim[1]))

        # pack timings (ms) for external CSV
        self.last_timing = {
            "pred_ms": (t_pred1 - t_pred0) * 1000.0,
            **t_gh,
            "eig_Sinv_ms": (t_eig1 - t_eig0) * 1000.0,
            "chol_Sinv_ms": (t_chol1 - t_chol0) * 1000.0,
            "solve_St_y_ms": (t_rhs1 - t_rhs0) * 1000.0,
            "solve_Rinv_ms": (t_w1 - t_w0) * 1000.0,
            "qr_A_ms": (t_qr1 - t_qr0) * 1000.0,
            "solve_Rbar_ms": (t_solve1 - t_solve0) * 1000.0,
            "inv_Rbar_ms": (t_all1 - t_inv0) * 1000.0,
            "wekf_update_ms": (t_all1 - t_all0) * 1000.0,
        }

        # commit posterior
        self.x = x_post
        self.last_theta_eq = theta_eq.copy()
        return theta_eq
