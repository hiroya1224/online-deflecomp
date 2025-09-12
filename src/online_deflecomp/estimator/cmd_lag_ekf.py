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
# Debug print (per step): CSV line of [Omega, Phi @ theta].

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
    e_min: float = 1e-6          # guard tiny |e| when forming Phi
    # public LPF clamps
    tau_pub_min: float = 1e-5
    tau_pub_max: float = 10.0

class CmdLagEKF:
    """
    Gyro-only lag estimator via vector RLS on:
        Omega = (W_local(y) * Diag(e)) * kappa
    Public state kept for compatibility: x = [y; s], s = log(tau).
    Also computes per-joint recommended public LPF time constants:
        tau_pub_j = dt * a_j / (1 - a_j),  a_j = exp(-dt / tau_j_hat).
    """
    def __init__(self,
                 robot,
                 frame_names: List[str],
                 frame_ids: Dict[str, int],
                 g_world_unit: np.ndarray,   # kept for interface; unused here
                 cfg: CmdLagEKFConfig) -> None:
        self.robot = robot
        self.frame_names = list(frame_names)
        self.frame_ids = dict(frame_ids)
        self.cfg = cfg

        self.n = int(robot.nv)

        # public state/cov (y part kept for external compatibility)
        self.x = np.zeros(2 * self.n, dtype=float)  # [y; s]
        self.P = np.eye(2 * self.n, dtype=float)
        self.P[:self.n, :self.n] *= float(cfg.qy_diag)
        self.P[self.n:, self.n:] *= 1e-2

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

    # ---------- Pinocchio helpers (LOCAL IMU frame) ----------
    def _fk_update(self, q: np.ndarray) -> None:
        pin.computeJointJacobians(self.robot.model, self.robot.data, q)
        pin.updateFramePlacements(self.robot.model, self.robot.data)

    def _W_stack_local(self,
                       q: np.ndarray,
                       omega_obs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stack LOCAL-frame angular Jacobians and observed omegas for available IMU frames.
        Returns (W_stack(3m,n), Omega_stack(3m,)).
        """
        self._fk_update(q)
        W_list = []
        O_list = []
        for nm in self.frame_names:
            if nm not in omega_obs:
                continue
            fid = self.frame_ids[nm]
            J6_loc = pin.computeFrameJacobian(self.robot.model, self.robot.data, q, fid, pin.ReferenceFrame.LOCAL)
            Jw_loc = J6_loc[3:6, :]  # (3, n)
            W_list.append(Jw_loc)
            omg = np.asarray(omega_obs[nm], dtype=float).reshape(3)
            O_list.append(omg)
        if not W_list:
            return np.zeros((0, self.n), dtype=float), np.zeros((0,), dtype=float)
        W = np.vstack(W_list)
        Omega = np.hstack(O_list).reshape(-1)
        return W, Omega

    # ---------- BE propagation (kappa-only) ----------
    def _be_step(self,
                 y_prev: np.ndarray,
                 u_k: np.ndarray,
                 kappa: np.ndarray,
                 dt: float) -> np.ndarray:
        d = 1.0 + dt * kappa
        return (y_prev + dt * (kappa * u_k)) / d

    # ---------- API ----------
    def reset(self,
              y0: Optional[np.ndarray] = None,
              tau_init: Optional[np.ndarray] = None) -> None:
        if y0 is None:
            y0 = np.zeros(self.n, dtype=float)
        y0 = np.asarray(y0, dtype=float).reshape(self.n)
        self.x[:self.n] = y0
        self.u_prev = y0.copy()

        if tau_init is not None:
            tau_init = np.maximum(self.cfg.eps_tau, np.asarray(tau_init, dtype=float).reshape(self.n))
            self.kappa = np.clip(1.0 / tau_init, self.kappa_min, self.kappa_max)
        else:
            tau0 = max(self.cfg.eps_tau, float(self.cfg.tau_init))
            k0 = np.clip(1.0 / tau0, self.kappa_min, self.kappa_max)
            self.kappa = np.full((self.n,), float(k0), dtype=float)

        tau_vec = 1.0 / np.maximum(self.kappa, 1e-12)
        self.x[self.n:] = np.log(np.maximum(self.cfg.eps_tau, tau_vec))

        self.P[:, :] = 0.0
        self.P[:self.n, :self.n] = np.eye(self.n) * float(self.cfg.qy_diag)
        self.P[self.n:, self.n:] = np.eye(self.n) * 1e-2
        self.P[:self.n, self.n:] = 0.0
        self.P[self.n:, :self.n] = 0.0

        self.Pt = np.eye(self.n, dtype=float) * float(self.cfg.rls_P0)

        self._update_tau_pub_from_tau(tau_vec)
        self.initialized = True

    def _update_tau_pub_from_tau(self, tau_vec: np.ndarray) -> None:
        dt = float(self.cfg.dt)
        tau = np.maximum(np.asarray(tau_vec, dtype=float).reshape(self.n), self.cfg.eps_tau)
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
               theta_init: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        One RLS step driven by stacked gyro observations, kappa-only.
        Returns:
            (y_k, tau_vec_k)
        Also updates suggested public LPF tau_pub_vec per joint.
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

            # 3) W at theta_eq (LOCAL)
            W, Omega = self._W_stack_local(theta_eq, omega_obs)
        else:
            # fallback (legacy): W at y_pred, no sensitivity
            W, Omega = self._W_stack_local(y_pred, omega_obs)

        m = W.shape[0]
        if m == 0:
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

        # 3) Phi assembly
        #    legacy: Phi = W * Diag(e)
        #    new   : Phi = W * S * Diag(e)
        e_guard = e.copy()
        tiny = np.abs(e_guard) < float(self.cfg.e_min)
        e_guard[tiny] = np.sign(e_guard[tiny]) * float(self.cfg.e_min)
        if use_sens:
            De = np.diag(e_guard)
            Phi = W @ (S @ De)   # (3m x n)
        else:
            Phi = W * e_guard.reshape((1, self.n))

        if float(np.linalg.norm(Phi, ord='fro')) < float(self.cfg.phi_norm_min):
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

        # 4) vector RLS update on kappa (theta)
        lam = float(self.cfg.rls_lambda)
        P = self.Pt
        theta = self.kappa  # (n,)

        S = lam * np.eye(m, dtype=float) + Phi @ P @ Phi.T
        S.flat[::m+1] += float(self.cfg.rls_ridge)
        try:
            Sinv = np.linalg.pinv(S, rcond=1e-10)
        except Exception:
            Sinv = np.linalg.pinv(S + 1e-8 * np.eye(m))
        K = P @ Phi.T @ Sinv

        # Innovation: Omega_obs - Phi theta
        innov = Omega - (Phi @ theta)
        theta_new = theta + K @ innov

        # ---- Debug print (CSV one-liner) ----
        try:
            vals = np.hstack([Omega, Phi @ theta]).tolist()
            fmt = ",".join(["{}"] * len(vals))
            print(fmt.format(*vals))
        except Exception:
            print("DBG Omega|Phi@theta:", Omega, Phi @ theta)

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

        # maintain y-cov (compat only; diagonal approx)
        Fy = 1.0 / (1.0 + dt * self.kappa)
        Py = np.diag(Fy) @ self.P[:self.n, :self.n] @ np.diag(Fy) + np.eye(self.n) * float(self.cfg.qy_diag)
        self.P[:self.n, :self.n] = Py
        self.P[:self.n, self.n:] = 0.0
        self.P[self.n:, :self.n] = 0.0

        # update per-joint pub tau suggestion
        self._update_tau_pub_from_tau(tau_vec)

        self.u_prev = u.copy()
        return y_post.copy(), tau_vec
