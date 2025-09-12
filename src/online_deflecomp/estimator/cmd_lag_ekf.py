# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pinocchio as pin  # LOCAL frame Jacobian (matches sim IMU local)

# Model (per joint):
#   First-order lag: ydot = kappa * (u - y),  kappa = 1 / tau.
#   BE propagation:  y_k = (y_{k-1} + dt * (kappa ∘ u_k)) / (1 + dt * kappa).
#   Gyro-only regress: Omega ≈ W_local(y) * Diag(e) * kappa,  e = u - y.
#
# "Safe poles" for public LPF (per joint):
#   a = exp(-dt / tau_hat),  choose alpha_pub = 1 - a  (double pole at z=a).
#   tau_pub = dt * a / (1 - a).
#
@dataclass
class CmdLagEKFConfig:
    dt: float
    # RLS hyper-parameters
    rls_lambda: float = 0.99
    rls_P0: float = 1e2
    rls_ridge: float = 1e-9
    # placeholders / numeric guards (as requested)
    qy_diag: float = 1e-6
    qs_diag: float = 1e-6
    rk_diag: float = 1e-6
    ridge: float = 1e-6
    # tau bounds/init
    tau_init: float = 0.0
    tau_min: float = 0.0
    tau_max: float = 0.8
    eps_tau: float = 1e-2
    # gating
    phi_norm_min: float = 1e-8   # skip update if ||Phi||_F is too small
    e_min: float = 1e-6          # floor for e = u - y
    # public LPF tau bounds
    tau_pub_min: float = 1e-5
    tau_pub_max: float = 10.0
    # robustness for noisy gyro
    omega_alpha: float = 0.2   # EMA factor for Omega, in (0,1]; 1.0 = no smoothing
    innov_clip: float = 5.0    # clip magnitude for innovation (rad/s units)

class CmdLagEKF:
    def __init__(self, robot, frames: List[str], frame_ids: Dict[str, int], g_unit: np.ndarray, cfg: CmdLagEKFConfig) -> None:
        self.robot = robot
        self.frames = list(frames)
        self.frame_ids = dict(frame_ids)
        self.g_unit = np.asarray(g_unit, dtype=float).reshape(3)
        self.cfg = cfg

        self.n = robot.nv
        self.m = 3 * len(self.frames)  # stacked angular velocity (3 per frame)
        # state: [y(n), log(tau)(n)]
        self.x = np.zeros(2 * self.n, dtype=float)
        # parameter covariance for kappa (n x n)
        self.Pt = np.eye(self.n, dtype=float) * float(cfg.rls_P0)

        # bounds for kappa from tau bounds
        tau_lo = max(float(cfg.eps_tau), float(cfg.tau_min))
        tau_hi = float(cfg.tau_max) if float(cfg.tau_max) > 0.0 else np.inf
        self.kappa_min = 0.0 if np.isinf(tau_hi) else 1.0 / max(tau_hi, cfg.eps_tau)
        self.kappa_max = 1.0 / tau_lo

        # parameters: kappa only
        tau0 = max(float(cfg.eps_tau), float(cfg.tau_init))
        k0 = np.clip(1.0 / tau0, self.kappa_min, self.kappa_max)
        self.kappa = np.full((self.n,), float(k0), dtype=float)

        # RLS covariance over kappa (n x n)
        self.Pt = np.eye(self.n, dtype=float) * float(cfg.rls_P0)

        # previous command (kept for completeness)
        self.u_prev = np.zeros(self.n, dtype=float)

        # expose s = log(tau)
        tau_vec = 1.0 / np.maximum(self.kappa, 1e-12)
        self.x[self.n:] = np.log(np.maximum(cfg.eps_tau, tau_vec))

        # per-joint recommended public LPF tau (computed each step)
        self.tau_pub_vec = np.full((self.n,), float(cfg.tau_pub_min), dtype=float)

        self.initialized = False
        self.theta_eq_last: Optional[np.ndarray] = None
        # EMA of stacked Omega
        self.omega_ema = np.zeros(self.m, dtype=float)

    # ---------- Pinocchio helpers (LOCAL IMU frame) ----------
    def _fk_update(self, q: np.ndarray) -> None:
        pin.computeJointJacobians(self.robot.model, self.robot.data, q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

    def _W_local_stack(self, q: np.ndarray) -> np.ndarray:
        # Stack LOCAL angular Jacobians (3 x n) for each frame
        self._fk_update(q)
        blocks = []
        for nm in self.frames:
            fid = self.frame_ids.get(nm, None)
            if fid is None:
                blocks.append(np.zeros((3, self.n), dtype=float))
                continue
            J6 = pin.computeFrameJacobian(self.robot.model, self.robot.data, q, fid, pin.ReferenceFrame.LOCAL)
            blocks.append(J6[3:6, :])
        return np.vstack(blocks) if blocks else np.zeros((0, self.n), dtype=float)

    # ---------- lag propagation ----------
    @staticmethod
    def _be_step(y_prev: np.ndarray, u: np.ndarray, kappa: np.ndarray, dt: float) -> np.ndarray:
        denom = 1.0 + dt * kappa
        return (y_prev + dt * (kappa * u)) / denom

    def reset(self, y0: Optional[np.ndarray] = None) -> None:
        y = np.zeros(self.n, dtype=float) if y0 is None else np.asarray(y0, dtype=float).reshape(self.n)
        self.x[:self.n] = y
        self.initialized = True

    def _update_tau_pub_from_tau(self, tau: np.ndarray) -> None:
        dt = float(self.cfg.dt)
        a = np.exp(-dt / tau)
        one_minus_a = np.maximum(1.0 - a, 1e-9)
        tau_pub = dt * a / one_minus_a
        tau_pub = np.clip(tau_pub, float(self.cfg.tau_pub_min), float(self.cfg.tau_pub_max))
        self.tau_pub_vec = tau_pub

    def get_tau_pub(self) -> np.ndarray:
        return self.tau_pub_vec.copy()

    def update(self,
               u_k: np.ndarray,
               g_obs: Dict[str, np.ndarray],          # kept for signature; unused
               omega_obs: Optional[Dict[str, np.ndarray]] = None,
               # --- new optional inputs (minimal invasive) ---
               kp_vec: Optional[np.ndarray] = None,
               solver: Optional[object] = None,
               theta_init: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            y_post: time-aligned internal command state (n,)
            tau_vec: per-joint tau estimate derived from kappa
        """
        u = np.asarray(u_k, dtype=float).reshape(self.n)
        if not self.initialized:
            self.reset(y0=u)

        dt = float(self.cfg.dt)
        y_prev = self.x[:self.n].copy()

        # 1) advance y with current kappa (BE)
        y_pred = self._be_step(y_prev, u, self.kappa, dt)
        e = u - y_pred  # (n,)

        # 2) build stacked LOCAL W_i(y_pred) and Omega_obs (IMU local)
        if omega_obs is None or len(omega_obs) == 0:
            y_post = y_pred
            self.x[:self.n] = y_post
            tau_vec = 1.0 / np.maximum(self.kappa, 1e-12)
            tau_vec = np.clip(tau_vec,
                              max(self.cfg.eps_tau, self.cfg.tau_min),
                              (self.cfg.tau_max if self.cfg.tau_max > 0.0 else np.inf))
            self.x[self.n:] = np.log(tau_vec)
            self._update_tau_pub_from_tau(tau_vec)
            self.u_prev = u.copy()
            return y_post.copy(), tau_vec

        # --- build W and Phi ---
        use_sens = (kp_vec is not None) and (solver is not None)
        if use_sens:
            # 1) equilibrium at current y_pred
            kp_vec = np.asarray(kp_vec, dtype=float).reshape(self.n)
            th0 = (self.theta_eq_last if self.theta_eq_last is not None
                   else (theta_init if theta_init is not None else y_pred))
            try:
                theta_eq = solver.solve(self.robot, theta_cmd=y_pred, kp_vec=kp_vec, theta_init=th0)
            except Exception:
                theta_eq = th0.copy()
            self.theta_eq_last = theta_eq.copy()

            # 2) sensitivity S = H^{-1} Kp,  H = d_tau_gravity(theta_eq) + diag(Kp)
            try:
                Htheta = self.robot.d_tau_gravity(theta_eq).astype(float)
            except Exception:
                Htheta = np.zeros((self.n, self.n), dtype=float)
            H = Htheta + np.diag(kp_vec)
            try:
                Hinv = np.linalg.pinv(H, rcond=1e-10)
            except Exception:
                Hinv = np.linalg.pinv(H + 1e-6 * np.eye(self.n))
            S = Hinv @ np.diag(kp_vec)   # (n x n)
        else:
            S = np.eye(self.n, dtype=float)

        W = self._W_local_stack(theta_eq if use_sens else y_pred)  # LOCAL frame
        # Apply gating / floor
        e_eff = np.where(np.abs(e) < float(self.cfg.e_min), np.sign(e) * float(self.cfg.e_min), e)
        Phi = W @ (S @ np.diag(e_eff))   # (3*m) x n

        # 3) collect Omega from omega_obs
        Om_list = []
        for nm in self.frames:
            w = omega_obs.get(nm, None)
            if w is None:
                Om_list.append(np.zeros(3, dtype=float))
            else:
                Om_list.append(np.asarray(w, dtype=float).reshape(3))
        Omega = np.hstack(Om_list).reshape(-1)

        # 4) robustify Omega and innovation
        # 4) vector RLS over kappa (diag)
        lam = float(self.cfg.rls_lambda)
        P = self.Pt.copy()
        theta = self.kappa.copy()
        
        # EMA on Omega
        a_ema = float(np.clip(self.cfg.omega_alpha, 1e-3, 1.0))
        self.omega_ema = (1.0 - a_ema) * self.omega_ema + a_ema * Omega
        Omega_use = self.omega_ema

        # S_k = lam*I + Phi P Phi^T (+ ridge)
        S_k = lam * np.eye(Phi.shape[0], dtype=float) + Phi @ P @ Phi.T + float(self.cfg.rls_ridge) * np.eye(Phi.shape[0])
        try:
            Sinv = np.linalg.pinv(S_k, rcond=1e-10)
        except Exception:
            Sinv = np.linalg.pinv(S_k + 1e-6 * np.eye(S_k.shape[0]))
        K = P @ Phi.T @ Sinv
        # Innovation (Huber-like clipping): Omega_use - Phi theta
        innov = Omega_use - (Phi @ theta)
        clip_mag = float(max(self.cfg.innov_clip, 0.0))
        if clip_mag > 0.0:
            innov = np.clip(innov, -clip_mag, clip_mag)

        theta_new = theta + K @ innov

        # projection to physical bounds (tau bounds -> kappa bounds)
        kappa_new = np.clip(theta_new, self.kappa_min, self.kappa_max)

        # covariance: P = lambda^{-1} (I - K Phi) P
        I_n = np.eye(self.n, dtype=float)
        P_new = (I_n - K @ Phi) @ P
        P_new = (1.0 / lam) * P_new

        # commit parameters + cov
        self.kappa = kappa_new
        self.Pt = 0.5 * (P_new + P_new.T)  # symmetrize

        # 5) final y with updated kappa
        y_post = self._be_step(y_prev, u, self.kappa, dt)

        # 6) expose public state
        self.x[:self.n] = y_post
        tau_vec = 1.0 / np.maximum(self.kappa, 1e-12)
        tau_vec = np.clip(tau_vec,
                          max(self.cfg.eps_tau, self.cfg.tau_min),
                          (self.cfg.tau_max if self.cfg.tau_max > 0.0 else np.inf))
        self.x[self.n:] = np.log(tau_vec)

        # Note: no state covariance (self.P) is maintained in this RLS-only class; legacy update removed.

        # update per-joint pub tau suggestion
        self._update_tau_pub_from_tau(tau_vec)

        self.u_prev = u.copy()
        return y_post.copy(), tau_vec
