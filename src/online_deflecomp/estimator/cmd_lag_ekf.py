# -*- coding: utf-8 -*-
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# NOTE:
# - ASCII-only variable names.
# - Gyro-only lag estimation.
# - Use analytic ZOH solution for first-order lag:
#     y_k = a * y_{k-1} + (1 - a) * u_k,  a = exp(-dt * kappa),  kappa = 1/tau.
# - Build per-joint scalar measurements of kappa via log-ratio of projected gyros:
#     gamma_{j,k} = (W[:,j]/||W[:,j]||)^T * omega_k  ≈  kappa_j * e_{j,k}
#     kappa_meas = -(1/dt) * log( |gamma_k| / (|gamma_{k-1}|+eps) )
# - Each joint runs its own tiny KF on kappa_j (random walk). Gravity is not used here.

def _safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.linalg.norm(v)) + eps

@dataclass
class CmdLagEKFConfig:
    dt: float
    # y process (kept for API; y covariance is diagonal small)
    qy_diag: float = 1e-6
    # kappa random-walk process variance (per joint)
    qs_diag: float = 1e-4
    # measurement noise variance for kappa_meas (per joint, scalar used)
    rk_diag: float = 1e-2
    # tau init/bounds
    tau_init: float = 0.0
    tau_min: float = 0.0
    tau_max: float = 0.8
    eps_tau: float = 1e-2
    # gyro projection gating
    gamma_min: float = 1e-4      # [rad/s] minimum magnitude to accept a ratio
    ratio_min: float = 1e-3      # clip for |gamma_k|/|gamma_{k-1}|
    ratio_max: float = 0.999     # <1 to keep log defined and robust
    # small ridge for numerics
    ridge: float = 1e-12

class CmdLagEKF:
    """
    Lightweight lag estimator:
      - State reported as x = [y; s], s = log(tau) for API compatibility.
      - Internally estimate kappa = 1/tau per joint via scalar KF, using
        kappa_meas = -(1/dt) * log(|gamma_k| / (|gamma_{k-1}|+eps)).
      - y propagated by exact ZOH with current kappa.
    """
    def __init__(self,
                 robot,
                 frame_names: List[str],
                 frame_ids: Dict[str, int],
                 g_world_unit: np.ndarray,   # unused (kept for API)
                 cfg: CmdLagEKFConfig) -> None:
        self.robot = robot
        self.frame_names = list(frame_names)
        self.frame_ids = dict(frame_ids)
        self.cfg = cfg

        self.n = int(robot.nv)

        # public buffers
        self.x = np.zeros(2 * self.n, dtype=float)  # [y; s]
        self.P = np.eye(2 * self.n, dtype=float)

        # tau->kappa bounds
        self.kappa_min = 0.0 if float(cfg.tau_max) <= 0.0 else 1.0 / max(cfg.tau_max, cfg.eps_tau)
        self.kappa_max = 1.0 / max(cfg.tau_min, cfg.eps_tau) if float(cfg.tau_min) > 0.0 else 1.0 / cfg.eps_tau

        # init kappa from tau_init (vectorized)
        tau0 = max(float(cfg.eps_tau), float(cfg.tau_init))
        kappa0 = np.clip(1.0 / tau0, self.kappa_min, self.kappa_max)
        self.kappa = np.full((self.n,), float(kappa0), dtype=float)

        # scalar KF per joint: covariances
        self.Pk = np.ones(self.n, dtype=float) * 1e-2
        self.Qk = float(cfg.qs_diag)
        self.Rk = float(cfg.rk_diag)

        # y init and cov
        self.x[:self.n] = np.zeros(self.n, dtype=float)
        self.P[:self.n, :self.n] = np.eye(self.n) * float(cfg.qy_diag)

        # s=log(tau) from kappa (for API)
        tau_vec = 1.0 / np.maximum(self.kappa, 1e-12)
        self.x[self.n:] = np.log(np.maximum(cfg.eps_tau, tau_vec))

        # store previous gamma per frame (n,) for ratio
        self.prev_gamma: Dict[str, np.ndarray] = {}

        self.initialized = False

    # ---------- kinematics helpers ----------

    def _fk_update(self, y: np.ndarray) -> None:
        self.robot._fk_update(y)

    def _W_frames(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        W(y) = R_wl(y)^T * Jw_omega_world(y)  (3 x n), in frame coords.
        """
        self._fk_update(y)
        Wmap: Dict[str, np.ndarray] = {}
        for nm in self.frame_names:
            fid = self.frame_ids[nm]
            R_wl = self.robot.data.oMf[fid].rotation
            try:
                Jw = self.robot.frame_angular_jacobian_world(y, fid)  # (3,n)
            except TypeError:
                Jw = self.robot.frame_angular_jacobian_world(fid, y)
            Wmap[nm] = R_wl.T @ Jw
        return Wmap

    def _project_gamma(self,
                       W: np.ndarray,
                       omega: np.ndarray) -> np.ndarray:
        """
        For each joint j: gamma_j = (W[:,j]/||W[:,j]||)^T * omega.
        Returns shape (n,)
        """
        n = W.shape[1]
        gam = np.zeros(n, dtype=float)
        for j in range(n):
            col = W[:, j].reshape(3)
            norm = _safe_norm(col)
            v = col / norm
            gam[j] = float(v.dot(omega))
        return gam

    # ---------- API ----------

    def reset(self,
              y0: Optional[np.ndarray] = None,
              tau_init: Optional[np.ndarray] = None) -> None:
        if y0 is None:
            y0 = np.zeros(self.n, dtype=float)
        y0 = np.asarray(y0, dtype=float).reshape(self.n)
        self.x[:self.n] = y0

        if tau_init is not None:
            tau_init = np.maximum(self.cfg.eps_tau, np.asarray(tau_init, dtype=float).reshape(self.n))
            self.kappa = np.clip(1.0 / tau_init, self.kappa_min, self.kappa_max)
        else:
            tau0 = max(self.cfg.eps_tau, float(self.cfg.tau_init))
            kappa0 = np.clip(1.0 / tau0, self.kappa_min, self.kappa_max)
            self.kappa = np.full((self.n,), float(kappa0), dtype=float)

        self.Pk[:] = 1e-2
        self.P[:self.n, :self.n] = np.eye(self.n) * float(self.cfg.qy_diag)

        tau_vec = 1.0 / np.maximum(self.kappa, 1e-12)
        self.x[self.n:] = np.log(np.maximum(self.cfg.eps_tau, tau_vec))

        self.prev_gamma.clear()
        self.initialized = True

    def update(self,
               u_k: np.ndarray,
               g_obs: Dict[str, np.ndarray],  # kept for signature; not used
               omega_obs: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        One step using intended command u_k and gyro-only observations.
        Returns:
            (y_k, tau_vec_k)
        """
        u = np.asarray(u_k, dtype=float).reshape(self.n)
        if not self.initialized:
            self.reset(y0=u)

        y_prev = self.x[:self.n].copy()
        dt = float(self.cfg.dt)

        # ---- exact ZOH propagation with current kappa ----
        a = np.exp(-dt * np.asarray(self.kappa, dtype=float).reshape(self.n))  # (n,)
        a = np.clip(a, 1e-6, 1.0 - 1e-9)
        y_pred = a * y_prev + (1.0 - a) * u

        # y covariance (diag approx; small)
        Fy = np.diag(a)
        Py = Fy @ self.P[:self.n, :self.n] @ Fy.T + np.eye(self.n) * float(self.cfg.qy_diag)

        # ---- build kappa measurements from gyro ratios (per joint) ----
        if omega_obs is not None and len(omega_obs) > 0:
            Wmap = self._W_frames(y_pred)
            # per-joint list of candidates from all frames
            z_list = [[] for _ in range(self.n)]
            for nm in self.frame_names:
                if nm not in omega_obs:
                    continue
                omega = np.asarray(omega_obs[nm], dtype=float).reshape(3)
                Wnm = Wmap[nm]  # (3,n)
                gamma_curr = self._project_gamma(Wnm, omega)  # (n,)
                gamma_prev = self.prev_gamma.get(nm, None)

                if gamma_prev is not None:
                    # ratio and log
                    for j in range(self.n):
                        g0 = abs(float(gamma_prev[j]))
                        g1 = abs(float(gamma_curr[j]))
                        if g0 < self.cfg.gamma_min or g1 < self.cfg.gamma_min:
                            continue  # low SNR -> skip
                        r = g1 / (g0 + 1e-12)
                        r = float(np.clip(r, self.cfg.ratio_min, self.cfg.ratio_max))
                        z = -(1.0 / dt) * np.log(r)  # kappa measurement
                        # bounds guard
                        if not np.isfinite(z):
                            continue
                        z = float(np.clip(z, self.kappa_min, self.kappa_max))
                        z_list[j].append(z)

                # store current for next step
                self.prev_gamma[nm] = gamma_curr.copy()

            # ---- scalar KF per joint using aggregated measurement (median) ----
            for j in range(self.n):
                if len(z_list[j]) == 0:
                    # no update for this joint
                    self.Pk[j] = self.Pk[j] + self.Qk
                    continue
                z_med = float(np.median(z_list[j]))
                # predict
                Pp = self.Pk[j] + self.Qk
                # update (H=1)
                S = Pp + self.Rk + self.cfg.ridge
                K = Pp / S
                innov = z_med - self.kappa[j]
                self.kappa[j] = self.kappa[j] + K * innov
                # bounds
                self.kappa[j] = float(np.clip(self.kappa[j], self.kappa_min, self.kappa_max))
                # cov
                self.Pk[j] = (1.0 - K) * Pp
        else:
            # no gyro; only time update on kappa cov
            self.Pk[:] = self.Pk + self.Qk

        # ---- optional refine y with updated kappa (still analytic) ----
        a2 = np.exp(-dt * np.asarray(self.kappa, dtype=float).reshape(self.n))
        a2 = np.clip(a2, 1e-6, 1.0 - 1e-9)
        y_post = a2 * y_prev + (1.0 - a2) * u

        # ---- commit public state ----
        self.x[:self.n] = y_post
        tau_vec = 1.0 / np.maximum(self.kappa, 1e-12)
        tau_vec = np.clip(tau_vec, max(self.cfg.eps_tau, self.cfg.tau_min),
                          (self.cfg.tau_max if self.cfg.tau_max > 0.0 else np.inf))
        self.x[self.n:] = np.log(tau_vec)

        # pack P (block-diag approx)
        self.P[:self.n, :self.n] = Py
        # s variance via delta method: s = log(tau) = -log(kappa)
        var_k = self.Pk.copy()
        J = 1.0 / np.maximum(self.kappa, 1e-12)   # |ds/dkappa|
        var_s = (J * J) * var_k
        self.P[self.n:, self.n:] = np.diag(var_s)
        self.P[:self.n, self.n:] = 0.0
        self.P[self.n:, :self.n] = 0.0

        return self.x[:self.n].copy(), tau_vec
