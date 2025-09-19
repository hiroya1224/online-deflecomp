#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from online_deflecomp.utils.robot import RobotArm
from online_deflecomp.controller.equilibrium import EquilibriumSolver

# --- QR-based linear solver: solves A X = B via QR without forming A^{-1} ---
def _qr_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: (m x m), B: (m x k). Uses reduced QR for speed/stability.
    # Assumes A is reasonably well-conditioned after ridge/lam terms.
    Q, R = np.linalg.qr(A, mode="reduced")
    return np.linalg.solve(R, Q.T @ B)

# --- PSD projection helper (eigenvalue floor/ceiling) ---
def _project_psd_bounds(M: np.ndarray, eig_min: float = 1e-12, eig_max: Optional[float] = None) -> np.ndarray:
    """
    Project symmetric M onto the PSD cone with spectral bounds:
      M_proj = V * clip(eig(M), [eig_min, eig_max]) * V^T.
    If eig_max is None, only floor is applied.
    """
    A = 0.5 * (M + M.T)
    try:
        w, V = np.linalg.eigh(A)
    except Exception:
        # Fallback: small ridge then eigh
        A = A + (abs(eig_min) + 1e-12) * np.eye(A.shape[0], dtype=float)
        w, V = np.linalg.eigh(A)
    if eig_max is not None:
        w = np.clip(w, eig_min, eig_max)
    else:
        w = np.maximum(w, eig_min)
    return (V * w) @ V.T

# ----------------------------
# Config with per-feature toggles
# ----------------------------
@dataclass
class CmdLagEKFConfig:
    # timing
    dt: float

    # tau bounds/init (per-joint)
    tau_init: float = 0.20
    tau_min: float = 0.00
    tau_max: float = 0.80
    eps_tau: float = 1e-2

    # RLS hyper-params
    rls_lambda: float = 0.99
    rls_P0: float = 1e2
    rls_ridge: float = 1e-9

    # numeric guards
    qy_diag: float = 1e-6
    qs_diag: float = 1e-6
    rk_diag: float = 1e-6
    ridge: float = 1e-6
    phi_norm_min: float = 1e-8
    e_min: float = 1e-6

    # tau_pub bounds (for public LPF suggestion)
    tau_pub_min: float = 1e-5
    tau_pub_max: float = 10.0

    # --- feature toggles (RLS stability tricks) ---
    use_forgetting: bool = True         # forgetting factor lambda
    use_normalize: bool = True          # normalize Phi and Omega
    norm_epsilon: float = 1e-12         # epsilon for normalization
    use_innov_clip: bool = True         # clip innovation
    innov_clip: float = 5.0             # |innovation| limit (rad/s)
    use_gating: bool = True             # gate update when Phi/e too small
    use_projection: bool = True         # project kappa to [1/tau_max, 1/tau_min]

    # gyro smoothing
    use_omega_ema: bool = True
    omega_alpha: float = 0.2  # in (0,1]; 1.0 = no smoothing

    # spring sensitivity floor for cos((theta - u)/2)
    spring_cos_floor: float = 1e-6


class CmdLagEKF:
    """
    Gyro-only vector RLS for kappa (1/tau) with optional stability tricks.
    Builds Phi = W_local(theta_eq) @ S @ Diag(e), with e = u - y.
    """
    def __init__(self,
                 robot: RobotArm,
                 frames: List[str],
                 frame_ids: Dict[str, int],
                 g_unit: np.ndarray,
                 cfg: CmdLagEKFConfig) -> None:
        self.robot = robot
        self.frames = list(frames)
        self.frame_ids = dict(frame_ids)
        self.g_unit = np.asarray(g_unit, dtype=float)
        self.cfg = cfg

        self.n = self.robot.nv
        self.m = 3 * len(self.frames)  # 3 per frame (local angular velocity)

        # RLS state: kappa (diag) and covariance Pt
        kappa0 = np.full((self.n,), 1.0 / max(self.cfg.tau_init, 1e-9), dtype=float)
        self.kappa = kappa0.copy()
        self.Pt = np.eye(self.n, dtype=float) * float(self.cfg.rls_P0)

        # plant-side lag state y
        self.y_prev = np.zeros((self.n,), dtype=float)

        # public tau suggestion (for telemetry)
        self.tau_pub_vec = np.full((self.n,), float(self.cfg.tau_pub_min), dtype=float)

        # EMA for stacked Omega
        self.omega_ema = np.zeros((self.m,), dtype=float)

        # book-keeping
        self.initialized = False
        self.last_theta_eq: Optional[np.ndarray] = None

    # ---------- helpers ----------
    def _propagate_y_BE(self, u: np.ndarray, kappa: np.ndarray) -> np.ndarray:
        dt = float(self.cfg.dt)
        num = self.y_prev + dt * (kappa * u)
        den = 1.0 + dt * kappa
        y = num / np.maximum(den, 1e-12)
        return y

    def _recompute_tau_pub_from_tau(self, tau_vec: np.ndarray) -> None:
        # tau_pub = dt * a / (1 - a), a = exp(-dt/tau)
        dt = float(self.cfg.dt)
        a = np.exp(-dt / np.maximum(tau_vec, 1e-9))
        denom = np.maximum(1.0 - a, 1e-12)
        self.tau_pub_vec = np.clip(dt * a / denom,
                                   float(self.cfg.tau_pub_min),
                                   float(self.cfg.tau_pub_max))

    def get_tau_pub(self) -> np.ndarray:
        return self.tau_pub_vec.copy()

    def _local_angular_jacobian(self, theta: np.ndarray, frame_id: int) -> np.ndarray:
        """
        Try several method names to obtain LOCAL angular Jacobian of the frame.
        Returns shape (3, n).
        """
        # common conventions:
        # - 6xN spatial Jacobian: first 3 = angular, last 3 = linear OR vice versa
        # - explicit angular local jacobian 3xN
        J = None
        # try common names in this codebase
        # 1) frame_jacobian_local -> 6xN
        if hasattr(self.robot, "frame_jacobian_local"):
            J6 = self.robot.frame_jacobian_local(frame_id, theta)
            if J6 is not None:
                J6 = np.asarray(J6, dtype=float)
                if J6.shape[0] == 6:
                    # heuristics: angular usually on top rows in many libs
                    # but some put angular at bottom; choose larger norm
                    Jang_top = J6[0:3, :]
                    Jang_bot = J6[3:6, :]
                    if np.linalg.norm(Jang_top) >= np.linalg.norm(Jang_bot):
                        J = Jang_top
                    else:
                        J = Jang_bot
        # 2) angular_jacobian_local -> 3xN
        if J is None and hasattr(self.robot, "angular_jacobian_local"):
            J = np.asarray(self.robot.angular_jacobian_local(frame_id, theta), dtype=float)
        # 3) generic jacobian_local -> maybe 6xN
        if J is None and hasattr(self.robot, "jacobian_local"):
            Jtmp = np.asarray(self.robot.jacobian_local(frame_id, theta), dtype=float)
            if Jtmp.shape[0] == 6:
                J = Jtmp[0:3, :]
            elif Jtmp.shape[0] == 3:
                J = Jtmp
        if J is None:
            # fallback: zeros (avoids crash; update will be gated by phi_norm)
            J = np.zeros((3, self.n), dtype=float)
        return J

    def _stack_W_local(self, theta_eq: np.ndarray) -> np.ndarray:
        rows = []
        for nm in self.frames:
            fid = self.frame_ids.get(nm, None)
            if fid is None:
                rows.append(np.zeros((3, self.n), dtype=float))
            else:
                Jloc = self._local_angular_jacobian(theta_eq, fid)
                rows.append(Jloc)
        return np.vstack(rows) if rows else np.zeros((0, self.n), dtype=float)

    def _stack_omega_vec(self, omega_obs: Optional[Dict[str, np.ndarray]]) -> Optional[np.ndarray]:
        if omega_obs is None:
            return None
        vecs = []
        for nm in self.frames:
            w = omega_obs.get(nm, None)
            if w is None:
                vecs.append(np.zeros((3,), dtype=float))
            else:
                w3 = np.asarray(w, dtype=float).reshape(3)
                vecs.append(w3)
        if not vecs:
            return None
        return np.hstack(vecs).reshape(-1)

    # ---------- main update ----------
    def update(self,
               u_k: np.ndarray,
               g_obs: Dict[str, np.ndarray],        # unused here (gyro-only RLS)
               omega_obs: Optional[Dict[str, np.ndarray]],
               kp_vec: np.ndarray,
               solver: EquilibriumSolver,
               theta_init: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run one step of BE propagation + vector RLS on kappa.
        Returns (y_post, tau_vec).
        """
        u = np.asarray(u_k, dtype=float).reshape(-1)
        assert u.size == self.n

        # 1) predict y with current kappa (BE)
        y_pred = self._propagate_y_BE(u, self.kappa)
        e = u - y_pred

        # 2) equilibrium at y_pred (time-consistent)
        try:
            theta0 = theta_init if theta_init is not None else y_pred
            theta_eq = solver.solve_rti(self.robot, theta_cmd=y_pred, kp_vec=kp_vec, theta_init=theta0)             
            self.last_theta_eq = theta_eq.copy()
        except Exception:
            theta_eq = self.last_theta_eq.copy() if (self.last_theta_eq is not None) else y_pred.copy()

        # 3) build S = H^{-1} K_eff at theta_eq (NONLINEAR spring)
        try:
            Htheta = self.robot.d_tau_gravity(theta_eq).astype(float)
        except Exception:
            Htheta = np.zeros((self.n, self.n), dtype=float)
        # effective spring slope: K_eff = Kp * cos((theta_eq - y_pred)/2)
        d_nl = (theta_eq - y_pred)  # no wrap to keep S^1 continuity
        c_half = np.cos(0.5 * d_nl)
        c_eff = np.clip(c_half, float(self.cfg.spring_cos_floor), 1.0)
        K_eff = kp_vec * c_eff
        H = Htheta + np.diag(K_eff)
        try:
            Hinv = np.linalg.pinv(H, rcond=1e-10)
        except Exception:
            Hinv = np.linalg.pinv(H + 1e-6 * np.eye(self.n))
        S = Hinv @ np.diag(K_eff)

        # 4) Phi = W_local(theta_eq) * S * Diag(e)
        W = self._stack_W_local(theta_eq)
        Phi = W @ S @ np.diag(e)

        # 5) stack Omega (LOCAL), optionally smooth by EMA
        Omega = self._stack_omega_vec(omega_obs)
        if Omega is None:
            # nothing to update; keep prediction
            y_post = y_pred.copy()
            tau_vec = np.clip(1.0 / np.maximum(self.kappa, 1e-12),
                              float(self.cfg.tau_min), float(self.cfg.tau_max))
            self._recompute_tau_pub_from_tau(tau_vec)
            self.y_prev = y_post.copy()
            return y_post, tau_vec

        if self.cfg.use_omega_ema:
            a = float(np.clip(self.cfg.omega_alpha, 1e-3, 1.0))
            self.omega_ema = (1.0 - a) * self.omega_ema + a * Omega
            Omega_use = self.omega_ema
        else:
            Omega_use = Omega

        # 6) vector RLS on kappa with optional tricks
        P = self.Pt.copy()
        theta = self.kappa.copy()

        # gating
        do_update = True
        # if self.cfg.use_gating:
        #     if np.linalg.norm(e) < float(self.cfg.e_min):
        #         do_update = False
        #     if np.linalg.norm(Phi) < float(self.cfg.phi_norm_min):
        #         do_update = False

        if do_update:
            Phi_use = Phi.copy()
            Om_use = Omega_use.copy()

            # normalization
            if self.cfg.use_normalize:
                scale = np.sqrt(float(self.cfg.norm_epsilon) + float(np.sum(Phi_use * Phi_use)))
                Phi_use = Phi_use / scale
                Om_use = Om_use / scale

            # forgetting
            lam = float(self.cfg.rls_lambda) if self.cfg.use_forgetting else 1.0

            # innovation (with optional clip)
            innov = Om_use - Phi_use @ theta
            if self.cfg.use_innov_clip and float(self.cfg.innov_clip) > 0.0:
                c = float(self.cfg.innov_clip)
                innov = np.clip(innov, -c, c)

            # RLS gain and update
            S_k = lam * np.eye(Phi_use.shape[0], dtype=float) + Phi_use @ P @ Phi_use.T \
                 + float(self.cfg.rls_ridge) * np.eye(Phi_use.shape[0], dtype=float)
            # QR-based solves (no explicit inverse)
            innov_col = innov.reshape(-1, 1)
            # z = S_k^{-1} * innov
            z = _qr_solve(S_k, innov_col)                     # (m x 1)
            # T = S_k^{-1} * Phi_use
            T = _qr_solve(S_k, Phi_use)                       # (m x n)
            # theta update: theta_new = theta + P * Phi^T * z
            theta_new = (theta + (P @ Phi_use.T @ z).reshape(-1))
            # covariance update: P_new = (1/lam) * (P - P * Phi^T * T * P)
            P_new = (1.0 / lam) * (P - (P @ Phi_use.T @ T @ P))

            # 情報形の床（情報下限の注入）
            zeta = 1e-3  # TODO: 物理幅から決める
            Pinv = np.linalg.pinv(P_new, rcond=1e-12)
            P_new = np.linalg.pinv(Pinv + zeta*np.eye(P.shape[0]), rcond=1e-12)

            # projection (keep tau within [tau_min, tau_max])
            if self.cfg.use_projection:
                tau_new = 1.0 / np.maximum(theta_new, 1e-12)
                tau_new = np.clip(tau_new, float(self.cfg.tau_min), float(self.cfg.tau_max))
                theta_new = 1.0 / np.maximum(tau_new, 1e-12)

            self.kappa = theta_new
            self.Pt = P_new

        # 7) re-propagate y with updated kappa (consistency) and export
        y_post = self._propagate_y_BE(u, self.kappa)
        tau_vec = np.clip(1.0 / np.maximum(self.kappa, 1e-12),
                          float(self.cfg.tau_min), float(self.cfg.tau_max))
        self._recompute_tau_pub_from_tau(tau_vec)

        # 8) keep last y
        self.y_prev = y_post.copy()
        return y_post, tau_vec
