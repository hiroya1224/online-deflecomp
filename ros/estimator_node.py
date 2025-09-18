#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This node estimates Kp (via EKF with Bingham gravity residuals) and plant lag kappa (via gyro-only RLS),
then publishes a stabilized theta_cmd. It also prints numeric-only debug lines (IDs 1..7) for analysis.

With "equi-damped (double-pole)" publish LPF:
  - Plant pole: a = exp(-dt/tau)
  - Publish LPF pole: a_p = exp(-dt/tau_pub)
  - Desired closed-loop double pole: beta_eff = sqrt(a * a_p)
  - Gain: L = diag(gamma) * S^{-1},  gamma = (a + a_p - 2*sqrt(a a_p)) / ((1 - a)*(1 - a_p))

Debug CSV line formats (first field is line-type ID):
1,k,t,dt_ms,ms_cmdlag,ms_wekf,ms_eq,ms_total,n_g,n_w,had_A_map,wekf_updated,eq_solved
2,k,condH,condS,normSinv,lfac_min,lfac_max,denom_min,beta_min,beta_max,a_min,a_max
  * here lfac_min/max := min/max(gamma), beta_* := min/max(beta_eff), denom_min := min_j ((1-a_j)*(1-a_pj))
5,k,a_vec...,a_pub_vec...,beta_eff_vec...,gamma_vec...
(others unchanged)
"""
import argparse
from bisect import bisect_left
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import threading
import time
import numpy as np
import rospy
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as Rsc

from online_deflecomp.utils.robot import RobotArm
from online_deflecomp.controller.command import theta_cmd_from_theta_ref
from online_deflecomp.controller.equilibrium import EquilibriumSolver, EquilibriumConfig
from online_deflecomp.estimator.ekf import MultiFrameWeirdEKF
from online_deflecomp.estimator.cmd_lag_ekf import CmdLagEKF, CmdLagEKFConfig

from online_deflecomp.utils.bingham import NoiseCovarianceHelper
 
# ---------------- helpers ----------------
def angle_wrap(x: np.ndarray) -> np.ndarray:
    # elementwise wrap to (-pi, pi]
    return (x + np.pi) % (2.0 * np.pi) - np.pi

def angle_unwrap_to_ref(cur: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # return cur' s.t. (cur' - ref) is the minimal (wrapped) difference
    d = angle_wrap(cur - ref)
    return ref + d


def map_jointstate_to_model(msg: JointState, model_names: List[str]) -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(msg.name)}
    q = np.zeros(len(model_names), dtype=float)
    for j, n in enumerate(model_names):
        if n in name_to_idx:
            q[j] = msg.position[name_to_idx[n]]
    return q

def make_timestamp_sec(stamp) -> float:
    try:
        return float(stamp.secs) + 1e-9 * float(stamp.nsecs)
    except Exception:
        return rospy.Time.now().to_sec()

def quat_to_rot_wxyz(w: float, x: float, y: float, z: float) -> np.ndarray:
    q = np.array([x, y, z, w], dtype=float)
    return Rsc.from_quat(q).as_matrix()

def simple_bingham_unit(frame_g_dir: np.ndarray, world_g_unit: np.ndarray, A_param: float = 100.0) -> np.ndarray:
    # Build a 4x4 symmetric matrix for Bingham residual between frame gravity and world gravity unit.
    b = np.asarray(frame_g_dir, dtype=float); a = np.asarray(world_g_unit, dtype=float)
    bn = b / (np.linalg.norm(b) + 1e-12); an = a / (np.linalg.norm(a) + 1e-12)
    vq = np.array([0.0, bn[0], bn[1], bn[2]], dtype=float)  # pure imaginary quat for frame g
    xq = np.array([0.0, an[0], an[1], an[2]], dtype=float)  # pure imaginary quat for world g
    def Lmat(q: np.ndarray) -> np.ndarray:
        a0,b0,c0,d0 = q
        return np.array([[a0,-b0,-c0,-d0],[b0,a0,-d0,c0],[c0,d0,a0,-b0],[d0,-c0,b0,a0]], dtype=float)
    def Rmat(q: np.ndarray) -> np.ndarray:
        w0,x0,y0,z0 = q
        return np.array([[w0,-x0,-y0,-z0],[x0,w0,z0,-y0],[y0,-z0,w0,x0],[z0,y0,-x0,w0]], dtype=float)
    P = Lmat(xq) - Rmat(vq)
    return float(A_param) * (-0.25 * (P.T @ P))

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


@dataclass
class ImuBuffer:
    t_list: List[float]
    g_list: List[np.ndarray]
    w_list: List[np.ndarray]
    maxlen: int
    lock: threading.RLock

class ImuStore:
    def __init__(self, maxlen: int = 4000) -> None:
        self.buf: Dict[str, ImuBuffer] = {}
        self.maxlen = int(maxlen)
        self.lock = threading.RLock()

    def ensure(self, nm: str) -> None:
        with self.lock:
            if nm not in self.buf:
                self.buf[nm] = ImuBuffer([], [], [], self.maxlen, threading.RLock())

    def push(self, nm: str, t: float, g_dir: Optional[np.ndarray], w_loc: Optional[np.ndarray]) -> None:
        self.ensure(nm)
        b = self.buf[nm]
        g = None
        if g_dir is not None:
            g = np.asarray(g_dir, dtype=float)
            g = g / (np.linalg.norm(g) + 1e-12)
        w = np.zeros(3, dtype=float) if w_loc is None else np.asarray(w_loc, dtype=float).reshape(3)
        with b.lock:
            idx = bisect_left(b.t_list, t)
            if idx < len(b.t_list) and abs(b.t_list[idx] - t) < 1e-12:
                if g is not None:
                    b.g_list[idx] = g
                b.w_list[idx] = w
            else:
                b.t_list.insert(idx, t)
                b.g_list.insert(idx, g if g is not None else np.array([np.nan, np.nan, np.nan], dtype=float))
                b.w_list.insert(idx, w)
                if len(b.t_list) > b.maxlen:
                    b.t_list.pop(0); b.g_list.pop(0); b.w_list.pop(0)

    def interp(self, nm: str, t: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if nm not in self.buf:
            return None, None
        b = self.buf[nm]
        with b.lock:
            if not b.t_list:
                return None, None
            if t <= b.t_list[0]:
                g = b.g_list[0]; w = b.w_list[0]
            elif t >= b.t_list[-1]:
                g = b.g_list[-1]; w = b.w_list[-1]
            else:
                idx = bisect_left(b.t_list, t)
                t0 = b.t_list[idx - 1]; t1 = b.t_list[idx]
                g0 = b.g_list[idx - 1]; g1 = b.g_list[idx]
                w0 = b.w_list[idx - 1]; w1 = b.w_list[idx]
                a = 0.0 if (t1 - t0) <= 1e-12 else (t - t0) / (t1 - t0)
                g = g1 if np.any(np.isnan(g0)) or np.any(np.isnan(g1)) \
                    else ((1.0 - a) * g0 + a * g1)
                w = (1.0 - a) * w0 + a * w1
            g = None if (g is None or np.any(np.isnan(g))) else (g / (np.linalg.norm(g) + 1e-12))
            return g, w

# ---------------- estimator node ----------------
class EstimatorNode:
    def __init__(self,
                 urdf_path: str,
                 frames: List[str],
                 topic_ref: str,
                 topic_imu: str,
                 topic_cmd_out: str,
                 dt: float,
                 A_param: float,
                 kp0: List[float],
                 kp_lim: Tuple[float, float],
                 q_proc: float,
                 # control
                 kp_smooth_alpha: float,
                 fb_tau_des: List[float],
                 inv_denom_min: float,
                 rate_limit: float,
                 dbg_interval: int,
                 # RLS toggles (passed to CmdLagEKFConfig)
                 rls_lambda: float,
                 rls_use_forgetting: bool,
                 rls_use_normalize: bool,
                 rls_norm_epsilon: float,
                 rls_use_innov_clip: bool,
                 rls_innov_clip: float,
                 rls_use_gating: bool,
                 rls_e_min: float,
                 rls_phi_norm_min: float,
                 rls_use_projection: bool,
                 tau_min: float,
                 tau_max: float,
                 lag_use_omega_ema: bool,
                 lag_omega_alpha: float) -> None:

        # Robot and solver
        self.robot = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
        self.solver = EquilibriumSolver(EquilibriumConfig(maxiter=80))
        self.n = self.robot.nv
        self.model_joint_names = [self.robot.model.names[j] for j in range(1, self.robot.model.njoints)]

        # Frames and gravity
        self.frames = frames
        self.frame_ids: Dict[str, int] = {nm: self.robot.get_frame_id(nm) for nm in self.frames}
        self.g_world = np.array([0.0, 0.0, -9.81], dtype=float)
        self.g_unit = self.g_world / np.linalg.norm(self.g_world)
        self.A_param = float(A_param)

        # WEKF (for Kp)
        x0 = np.log(np.resize(np.array(kp0, dtype=float), self.n))
        P0 = np.eye(self.n) * 1.0
        Q  = np.eye(self.n) * float(q_proc)
        self.wekf = MultiFrameWeirdEKF(x0, P0, Q, eps_def=1e-6)
        self.kp_lim = kp_lim
        self.kp_smooth_alpha = float(kp_smooth_alpha)
        self.kp_hat_smooth: Optional[np.ndarray] = None

        # Lag estimator (gyro-only) with toggles
        lag_cfg = CmdLagEKFConfig(
            dt=float(dt),
            tau_init=max(min(0.2, tau_max), tau_min),
            tau_min=float(tau_min),
            tau_max=float(tau_max),
            eps_tau=1e-2,
            rls_lambda=float(rls_lambda),
            rls_P0=1e2,
            rls_ridge=1e-9,
            qy_diag=1e-6, qs_diag=1e-6, rk_diag=1e-6, ridge=1e-6,
            phi_norm_min=float(rls_phi_norm_min),
            e_min=float(rls_e_min),
            tau_pub_min=1e-5, tau_pub_max=10.0,
            use_forgetting=bool(rls_use_forgetting),
            use_normalize=bool(rls_use_normalize),
            norm_epsilon=float(rls_norm_epsilon),
            use_innov_clip=bool(rls_use_innov_clip),
            innov_clip=float(rls_innov_clip),
            use_gating=bool(rls_use_gating),
            use_projection=bool(rls_use_projection),
            use_omega_ema=bool(lag_use_omega_ema),
            omega_alpha=float(lag_omega_alpha)
        )
        self.cmdlag = CmdLagEKF(self.robot, self.frames, self.frame_ids, self.g_unit, lag_cfg)

        # State
        self.q_ref = np.zeros(self.n, dtype=float)
        self.have_ref = False
        self.last_cmd: Optional[np.ndarray] = None  # published (LPF'ed) command
        self.last_cmd_t: Optional[float] = None

        # IMU buffers
        self.imu = ImuStore(maxlen=4000)

        # ROS I/O
        self.sub_ref = rospy.Subscriber(topic_ref, JointState, self.cb_ref, queue_size=50)
        self.sub_imu = rospy.Subscriber(topic_imu, Imu, self.cb_imu, queue_size=400)
        self.pub_cmd = rospy.Publisher(topic_cmd_out, JointState, queue_size=10)
        self.pub_kp  = rospy.Publisher("/online_deflecomp/kp_hat", Float64MultiArray, queue_size=10)
        self.pub_kpc = rospy.Publisher("/online_deflecomp/kp_cov_diag", Float64MultiArray, queue_size=10)
        self.pub_tau = rospy.Publisher("/online_deflecomp/tau_hat", Float64MultiArray, queue_size=10)
        self.pub_tpub = rospy.Publisher("/online_deflecomp/tau_pub", Float64MultiArray, queue_size=10)

        # Control params
        # --- TODO(kappa-noise): keep base Q for WEKF noise adaptation ---
        # We will synthesize Q_eff each cycle: Q_eff = Q_base + Q_add(kappa_cov)
        self.q_proc_base = float(q_proc)

        self.dt = float(dt)
        self.inv_denom_min = float(inv_denom_min)
        self.rate_limit = float(rate_limit)
        if fb_tau_des is None or len(fb_tau_des) == 0:
            self.fb_tau_des = np.full((self.n,), 0.08, dtype=float)
        else:
            v = np.array(fb_tau_des, dtype=float).reshape(-1)
            self.fb_tau_des = v if v.size == self.n else np.full((self.n,), float(v[0]), dtype=float)

        # Debug
        self.dbg_interval = max(1, int(dbg_interval))
        self.k_counter = 0
        self.t0 = time.perf_counter()

        # --- TODO(kappa-noise): we need previous y_hat to form dy/dkappa Jacobian ---
        self.last_y_hat: Optional[np.ndarray] = None
        # Timer
        self.timer = rospy.Timer(rospy.Duration.from_sec(self.dt), self.on_timer)
        rospy.loginfo("estimator_node: frames=%s", ", ".join(self.frames))

    # -------- callbacks --------
    def cb_ref(self, msg: JointState) -> None:
        q = map_jointstate_to_model(msg, self.model_joint_names)
        self.q_ref = q.copy()
        self.have_ref = True

    def cb_imu(self, msg: Imu) -> None:
        nm = msg.header.frame_id
        if not nm:
            return
        t = make_timestamp_sec(msg.header.stamp)
        # gravity dir in frame from orientation (world -> frame)
        g_f = None
        try:
            w = float(msg.orientation.w); x = float(msg.orientation.x)
            y = float(msg.orientation.y); z = float(msg.orientation.z)
            R_wf = quat_to_rot_wxyz(w, x, y, z)
            g_f = R_wf.T @ (self.g_unit)
        except Exception:
            g_f = None
        w_loc = None
        try:
            w_loc = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=float)
        except Exception:
            w_loc = None
        self.imu.push(nm, t, g_f, w_loc)

    # -------- main timer --------
    def on_timer(self, event) -> None:
        if not self.have_ref:
            return
        t_step0 = time.perf_counter()
        self.k_counter += 1
        k = self.k_counter

        now = rospy.Time.now().to_sec()

        # (0) collect IMU observations at 'now'
        t_imu0 = time.perf_counter()
        g_obs: Dict[str, np.ndarray] = {}
        w_obs: Dict[str, np.ndarray] = {}
        for nm in self.frames:
            g_f, w_f = self.imu.interp(nm, now)
            if g_f is not None:
                g_obs[nm] = g_f
            if w_f is not None:
                w_obs[nm] = w_f
        n_g = len(g_obs); n_w = len(w_obs)

        t_imu1 = time.perf_counter()
        # (1) current published command u_k for lag propagation
        if self.last_cmd is not None:
            u_k = self.last_cmd.copy()
        else:
            u_k = theta_cmd_from_theta_ref(self.robot, self.q_ref, np.exp(self.wekf.x))

        # (2) lag estimator -> time-aligned y_hat and tau_hat (with toggles already in cfg)
        t_cmdlag0 = time.perf_counter()
        kp_for_lag = self.kp_hat_smooth if self.kp_hat_smooth is not None else np.exp(self.wekf.x)
        y_hat, tau_vec = self.cmdlag.update(
            u_k=u_k,
            g_obs=g_obs,
            omega_obs=w_obs if w_obs else None,
            kp_vec=kp_for_lag,
            solver=self.solver,
            theta_init=(self.wekf.last_theta_eq if self.wekf.last_theta_eq is not None else None)
        )
        t_cmdlag1 = time.perf_counter()
        self.pub_tau.publish(Float64MultiArray(data=tau_vec.tolist()))

        # ---------------------------------------------------------------------
        # (2.5) kappa-progress driven noise adaptation (revived)
        # ---------------------------------------------------------------------
        # Goal: inflate WEKF observation and process noise when kappa estimate
        #       is still uncertain, and relax them as P_kappa shrinks.
        #
        # Theory (linear spring version for WEKF consistency):
        #   J_ykappa = diag( dt*(u - y_prev) / (1 + dt*kappa)^2 )
        #   S_theta  = (d tau_g/dtheta + diag(K))^{-1} * diag(K)
        #   Sigma_theta = S_theta * J_ykappa * P_kappa * J_ykappa^T * S_theta^T
        #   Q_eff = Q_base + beta * J_x^T * (J_q^{-1} * Sigma_theta * J_q^{-T}) * J_x
        #   A_f  <- A_f + c_kappa * I_4   (unit quaternion: z^T(A + cI)z = z^T A z + c)
        # Notes:
        #   * We compute all at a linearization pose 'theta_lin' available *now*.
        #     Use last equilibrium if available, else y_hat (time-aligned cmd).
        #   * If CmdLagEKF exposes P through a different name, update the 'try' section.
        # ---------------------------------------------------------------------
        # --- TODO(kappa-noise): fetch kappa covariance from CmdLagEKF ---
        try:
            P_kappa = self.cmdlag.Pt.copy()                  # expected RLS covariance (n x n)
        except Exception:
            # Fallback: conservative diagonal (kept large so that adaptation is visible)
            P_kappa = np.eye(self.n) * 1e+1                 # TODO(kappa-noise): wire to real cov if name differs

        # Build dy/dkappa (per-joint) at current step
        kappa_vec = 1.0 / np.maximum(tau_vec, 1e-6)
        y_prev = self.last_y_hat if (self.last_y_hat is not None) else y_hat
        J_yk = (self.dt * (u_k - y_prev)) / np.square(1.0 + self.dt * kappa_vec)
        J_ykappa = np.diag(J_yk.reshape(-1))

        # ---------------------------------------------------------------------
        # Linearization for Σ_theta (kappa -> theta uncertainty propagation)
        # NOTE: This block is ONLY for Sigma_theta. Controller S and WEKF Hessian
        #       remain with linear K elsewhere. Here we use physically-meaningful
        #       effective stiffness K_eff = K * cos( (theta - theta_cmd)/2 ),
        #       so Σ_theta does not blow up when the equilibrium weakens near ±pi.
        # ---------------------------------------------------------------------
        theta_lin = self.wekf.last_theta_eq if (self.wekf.last_theta_eq is not None) else y_hat
        try:
            Htheta_lin = self.robot.d_tau_gravity(theta_lin).astype(float)
        except Exception:
            Htheta_lin = np.zeros((self.n, self.n), dtype=float)
        # Base stiffness (linear spring for consistency with WEKF elsewhere)
        K_lin = self.kp_hat_smooth if (self.kp_hat_smooth is not None) else np.exp(self.wekf.x)
        # --- TODO(kappa-noise): effective stiffness ONLY for Sigma_theta ---
        delta_nl = (theta_lin - y_hat)                     # theta_cmd ~ y_hat (time-aligned)
        c_half = np.cos(0.5 * delta_nl)
        c_eff = np.clip(c_half, 1e-3, 1.0)                 # numerical floor; physical softening near ±pi
        K_eff = K_lin * c_eff
        # J_q for Sigma_theta path
        J_q_lin = Htheta_lin + np.diag(K_eff)
        # S_theta = J_q^{-1} * diag(K_eff)
        try:
            # Minimal change: keep pinv but on the effective J_q.
            # TODO(kappa-noise): consider Tikhonov regularization if needed:
            #   J_q_inv = np.linalg.solve(J_q_lin.T @ J_q_lin + rho*I, J_q_lin.T)
            J_q_inv = np.linalg.pinv(J_q_lin, rcond=1e-10)
        except Exception:
            J_q_inv = np.linalg.pinv(J_q_lin + 1e-6 * np.eye(self.n))
        S_theta = J_q_inv @ np.diag(K_eff)
        Sigma_theta = S_theta @ J_ykappa @ P_kappa @ J_ykappa.T @ S_theta.T

        # --- TODO(kappa-noise): observation inflation scalar from Sigma_theta ---
        # Use scalar proxy: tr(Sigma_theta); tune alpha_R later if needed.
        # alpha_R = 1.0   # TODO(kappa-noise): retune/CLI if necessary
        # c_kappa = float(alpha_R * np.trace(Sigma_theta))

        # --- TODO(kappa-noise): process noise augmentation for WEKF ---
        delta_lin = (theta_lin - y_hat)          # consistent with linear spring used in WEKF
        J_x_lin = np.diag(K_lin * delta_lin)
        beta_Q = 1.0   # TODO(kappa-noise): retune/CLI if necessary
        Q_add = beta_Q * (J_x_lin.T @ (J_q_inv @ Sigma_theta @ J_q_inv.T) @ J_x_lin)
        Q_eff = np.eye(self.n) * self.q_proc_base + Q_add
        Q_eff = 0.5 * (Q_eff + Q_eff.T)
        # Numerical floor on diagonal for PSD robustness
        diag_min = 1e-10
        dQ = np.clip(np.diag(Q_eff), diag_min, None)
        np.fill_diagonal(Q_eff, dQ)

        # ------------------------------------------------------------------
        # TODO(kappa-noise): Physics-based spectral ceiling for Q_eff
        # x = log(K) random walk. Assume a maximum diffusion rate:
        #   sigma_xdot_max [1/s]  => per-step std <= sigma_xdot_max * dt
        # Thus eigenvalue cap: eig(Q_eff) <= (sigma_xdot_max * dt)^2
        # ------------------------------------------------------------------
        sigma_xdot_max = 20.0  # [1/s]  TODO(kappa-noise): tune / expose via CLI
        q_eig_max = (sigma_xdot_max * self.dt) ** 2
        q_eig_min = 1e-12      # numerical floor (already applied above; keep consistent)
        try:
            Q_eff = _project_psd_bounds(Q_eff, eig_min=q_eig_min, eig_max=q_eig_max)
        except Exception:
            # In worst case, keep the floored version without ceiling.
            pass

        # (3) Build A_map from IMU gravity observations (for Bingham WEKF)
        t_amap0 = time.perf_counter()
        A_map: Dict[int, np.ndarray] = {}
        for nm, g_f in g_obs.items():
            if nm in self.frame_ids:
                fid = self.frame_ids[nm]
                # --- TODO(kappa-noise): per-frame C_theta at the linearization pose (theta_lin) ---
                # C_theta_f = Qz_f(theta_lin) @ J_w_f(theta_lin)
                #   Qz_f: 4x4 quaternion "G" matrix from frame->world quat (wxyz)
                #   J_w_f: 4x3? actually 3xN world angular-vel jacobian; robot API returns (3 x n)
                #   We use Qz_f (4x4) only to map quat derivatives; for gravity Bingham, the effective
                #   observation Jacobian wrt joint angles is (3 x n) = J_w_f. The lifted form used in
                #   WEKF derivation is C_theta_f = Qz_f @ J_w_f (4 x n), and the covariance projection
                #   for gravity direction residual lives in the 3D subspace; NoiseCovarianceHelper
                #   expects a 3x3 covariance, so we project via J_w_f only.
                # z_lin = self.robot.frame_quaternion_wxyz_base(theta_lin, fid)
                # Qz_lin = BinghamUtils.qmat_from_quat_wxyz(z_lin)
                J_w_lin = self.robot.frame_angular_jacobian_world(theta_lin, fid)  # (3 x n)
                # --- TODO(kappa-noise): strict lifted form would be (Qz_lin @ J_w_lin), but for the
                # gravity-direction residual the 3D projection via J_w_lin is sufficient and matches
                # the helper signature (3x3 covariance).
                Sigma_obs = J_w_lin @ Sigma_theta @ J_w_lin.T          # (3 x 3) = C_theta Σ_theta C_theta^T
                Sigma_add = np.eye(3, dtype=float) * 1e-9              # TODO(kappa-noise): base sensor noise
                Sigma_in = np.eye(3, dtype=float) * 2e-2 + Sigma_obs
                A_raw = NoiseCovarianceHelper.calc_bingham_A(g_f, self.g_unit, Sigma_in, Sigma_add)

                # print("A_raw")
                # print(np.linalg.eigh(A_raw))
                # --- TODO(kappa-noise): optional scalar inflation c_kappa I_4 (unit quat: z^T(A+cI)z = z^T A z + c) ---
                # A_map[fid] = A_raw  # + (c_kappa * np.eye(4, dtype=float))
                A_map[fid] = simple_bingham_unit(g_f, self.g_unit, self.A_param)

                # print("simple_bingham_unit(g_f, self.g_unit, self.A_param)")
                # print(np.linalg.eigh(simple_bingham_unit(g_f, self.g_unit, self.A_param)))

                # print(A_raw / simple_bingham_unit(g_f, self.g_unit, self.A_param))
                
                # print(-4 * A_map[fid] / self.A_param)
                # print(NoiseCovarianceHelper.get_Hmat(g_f, self.g_unit).T @ NoiseCovarianceHelper.get_Hmat(g_f, self.g_unit))

        had_A_map = 1 if len(A_map) > 0 else 0
        t_amap1 = time.perf_counter()

        # (4) EKF update for Kp using time-aligned y_hat
        t_wekf0 = time.perf_counter()
        theta_init = self.wekf.last_theta_eq if self.wekf.last_theta_eq is not None else y_hat
        wekf_updated = 0
        if A_map:
            # --- TODO(kappa-noise): push Q_eff into WEKF before predict() inside update_with_multi ---
            # MultiFrameWeirdEKF.predict() uses self.Q; set it here every cycle.
            try:
                self.wekf.Q = Q_eff.copy()
            except Exception:
                # Safety: if WEKF ever wraps Q inside a config, keep old behavior
                self.wekf.Q = np.eye(self.n) * self.q_proc_base
            theta_eq = self.wekf.update_with_multi(
                self.solver, y_hat, A_map, self.robot,
                theta_init_eq_pred=theta_init, kp_lim=self.kp_lim
            )
            wekf_updated = 1
        else:
            try:
                theta_eq = self.solver.solve_rti(self.robot, theta_cmd=y_hat, kp_vec=np.exp(self.wekf.x), theta_init=theta_init)
                self.wekf.last_theta_eq = theta_eq.copy()
            except Exception:
                theta_eq = theta_init.copy()
        t_wekf1 = time.perf_counter()

        # (5) smooth Kp (EMA)
        kp_raw = np.clip(np.exp(self.wekf.x), self.kp_lim[0], self.kp_lim[1])
        if self.kp_hat_smooth is None:
            self.kp_hat_smooth = kp_raw.copy()
        else:
            a_s = float(np.clip(self.kp_smooth_alpha, 0.0, 1.0))
            self.kp_hat_smooth = (1.0 - a_s) * self.kp_hat_smooth + a_s * kp_raw
        self.pub_kp.publish(Float64MultiArray(data=self.kp_hat_smooth.tolist()))
        self.pub_kpc.publish(Float64MultiArray(data=np.diag(self.wekf.P).tolist()))

        # (6) compute NONLINEAR S and equi-damped gain L = diag(gamma) * S^{-1}
        t_eq0 = time.perf_counter()
        try:
            Htheta = self.robot.d_tau_gravity(theta_eq).astype(float)
        except Exception:
            Htheta = np.zeros((self.n, self.n), dtype=float)
        # K_eff = Kp * cos((theta_eq - y_hat)/2), with floor for numerical stability
        d_nl = (theta_eq - y_hat)  # no wrap (S^1-consistent via cos)
        c_half = np.cos(0.5 * d_nl)
        c_eff = np.clip(c_half, 1e-3, 1.0)
        K_eff = self.kp_hat_smooth * c_eff
        H = Htheta + np.diag(K_eff)
        try:
            Hinv = np.linalg.pinv(H, rcond=1e-10)
        except Exception:
            Hinv = np.linalg.pinv(H + 1e-6 * np.eye(self.n))
        S = Hinv @ np.diag(K_eff)

        # plant pole a
        a_vec = np.exp(-self.dt / np.maximum(tau_vec, 1e-6))
        # desired speed -> beta_des
        beta_des = np.exp(-self.dt / np.maximum(self.fb_tau_des, 1e-6))
        # publish LPF pole a_p solved from beta_des^2 = a * a_p
        a_pub_vec = beta_des * beta_des / np.maximum(a_vec, 1e-12)
        a_pub_vec = np.clip(a_pub_vec, 1e-6, 1.0 - 1e-6)

        # equi-damped gamma
        sqrt_aa = np.sqrt(a_vec * a_pub_vec)
        denom_prod = np.maximum((1.0 - a_vec) * (1.0 - a_pub_vec), self.inv_denom_min)
        gamma_vec = (a_vec + a_pub_vec - 2.0 * sqrt_aa) / denom_prod

        try:
            Sinv = np.linalg.pinv(S, rcond=1e-10)
        except Exception:
            Sinv = np.linalg.pinv(S + 1e-6 * np.eye(self.n))
        L = (np.diag(gamma_vec) @ Sinv)
        t_eq1 = time.perf_counter()

        # (7) u* and raw FB
        u_star = theta_cmd_from_theta_ref(self.robot, self.q_ref, self.kp_hat_smooth)
        # use shortest-arc error on S1 to avoid 2*pi jumps near +/-pi
        theta_err = angle_wrap(theta_eq - self.q_ref)
        u_cmd_raw = u_star - (L @ theta_err)

        # (8) publish LPF: u_k = a_p * u_{k-1} + (1-a_p) * u_raw, then rate limit
        if self.last_cmd is not None:
            # unwrap raw command to be continuous w.r.t. last published value
            u_cmd_raw = angle_unwrap_to_ref(u_cmd_raw, self.last_cmd)
            u_lpf = a_pub_vec * self.last_cmd + (1.0 - a_pub_vec) * u_cmd_raw
            du = u_lpf - self.last_cmd
        else:
            u_lpf = u_cmd_raw.copy()
            du = u_lpf.copy()

        max_step = self.rate_limit * self.dt
        du_lim = np.clip(du, -max_step, max_step)
        rate_hit = 1 if np.any(np.abs(du) > max_step + 1e-12) else 0
        u_cmd = (self.last_cmd + du_lim) if (self.last_cmd is not None) else u_lpf.copy()

        # (9) publish
        out = JointState()
        out.header.stamp = rospy.Time.now()
        out.name = self.model_joint_names
        out.position = u_cmd.tolist()
        self.pub_cmd.publish(out)

        # (10) telemetry
        self.pub_tpub.publish(Float64MultiArray(data=(-self.dt / np.log(np.maximum(a_pub_vec, 1e-12))).tolist()))

        # (11) book-keeping
        self.last_cmd = u_cmd.copy()
        self.last_cmd_t = now

        # -------- debug prints (numeric-only) --------
        if (k % self.dbg_interval) == 0:
            t_step1 = time.perf_counter()
            # --- TODO(kappa-noise-dbg): extra diagnostics for root-cause isolation ---
            # cond(J_q_lin), tr(P_kappa), tr(Sigma_theta), SL consistency error, asin saturation proxy, Q_eff stats
            try:
                condJq = float(np.linalg.cond(J_q_lin))  # from (2.5) block
            except Exception:
                condJq = float("nan")
            try:
                trPkappa = float(np.trace(self.cmdlag.Pt))
            except Exception:
                trPkappa = float("nan")
            try:
                trSigma = float(np.trace(Sigma_theta))
            except Exception:
                trSigma = float("nan")
            try:
                SL_err = float(np.linalg.norm((S @ (np.diag(gamma_vec) @ np.linalg.pinv(S))), ord=2))
            except Exception:
                # Fallback: ||S@L - diag(gamma)||_F
                try:
                    SL_err = float(np.linalg.norm(S @ (np.diag(gamma_vec) @ np.linalg.pinv(S)) - np.diag(gamma_vec)))
                except Exception:
                    SL_err = float("nan")
            # asin saturation proxy (requires tau_gravity; guarded)
            try:
                tau_g_ref = self.robot.tau_gravity(self.q_ref).reshape(-1)
                asin_arg = -tau_g_ref / (2.0 * np.maximum(self.kp_hat_smooth, 1e-12))
                asin_min, asin_max = float(np.min(asin_arg)), float(np.max(asin_arg))
                asin_sat = float(np.mean(np.abs(asin_arg) > 1.0))
            except Exception:
                asin_min = asin_max = asin_sat = float("nan")

            ms_cmdlag = (t_cmdlag1 - t_cmdlag0) * 1000.0
            ms_wekf = (t_wekf1 - t_wekf0) * 1000.0
            # extra: where do we spend outside WEKF?
            ms_imu = (t_imu1 - t_imu0) * 1000.0
            ms_amap = (t_amap1 - t_amap0) * 1000.0

            ms_eq = (t_eq1 - t_eq0) * 1000.0
            ms_total = (t_step1 - t_step0) * 1000.0
            t_rel = t_step1 - self.t0
            try:
                condH = float(np.linalg.cond(H))
            except Exception:
                condH = float("nan")
            try:
                condS = float(np.linalg.cond(S))
            except Exception:
                condS = float("nan")
            try:
                normSinv = float(np.linalg.norm(Sinv, 2))
            except Exception:
                normSinv = float(np.linalg.norm(Sinv))
            # for ID=2: reuse fields with new semantics (see header)
            beta_eff_vec = np.sqrt(a_vec * a_pub_vec)
            # --- TODO(kappa-noise-dbg): Q_eff stats if available ---
            try:
                qeff_diag = np.diag(Q_eff)
                qeff_min = float(np.min(qeff_diag)); qeff_max = float(np.max(qeff_diag))
                qeff_tr  = float(np.trace(Q_eff))
            except Exception:
                qeff_min = qeff_max = qeff_tr = float("nan")
            # --- TODO(kappa-noise-dbg): extra norms ---
            try:
                du_inf = float(np.max(np.abs(u_cmd_raw - (self.last_cmd if self.last_cmd is not None else 0.0))))
            except Exception:
                du_inf = float("nan")
            beta_min = float(np.min(beta_eff_vec)); beta_max = float(np.max(beta_eff_vec))
            lfac_min = float(np.min(gamma_vec)); lfac_max = float(np.max(gamma_vec))
            a_min = float(np.min(a_vec)); a_max = float(np.max(a_vec))
            denom_min = float(np.min(denom_prod))
            norm_theta_err = float(np.linalg.norm(theta_err))  # already wrapped
            norm_du_raw = float(np.linalg.norm(u_cmd_raw - (self.last_cmd if self.last_cmd is not None else 0.0)))
            norm_du_lim = float(np.linalg.norm(du_lim))

            # 1: summary
            print(",".join(str(x) for x in [
                1, k, f"{t_rel:.6f}", f"{self.dt*1000.0:.3f}",
                f"{ms_cmdlag:.3f}", f"{ms_wekf:.3f}", f"{ms_eq:.3f}", f"{ms_total:.3f}",
                n_g, n_w, 1 if had_A_map else 0, 1 if wekf_updated else 0, 1
            ]))
            # 2: conds/gamma/beta
            print(",".join(str(x) for x in [
                2, k,
                f"{condH:.6e}", f"{condS:.6e}", f"{normSinv:.6e}",
                f"{lfac_min:.6e}", f"{lfac_max:.6e}",
                f"{denom_min:.6e}", f"{beta_min:.6e}", f"{beta_max:.6e}",
                f"{a_min:.6e}", f"{a_max:.6e}"
            ]))
            # 3: limiter hits
            print(",".join(str(x) for x in [
                3, k,
                f"{norm_theta_err:.6e}", f"{norm_du_raw:.6e}", f"{norm_du_lim:.6e}",
                rate_hit, 0, 0
            ]))
            # 8: kappa-driven noise adaptation summary
            #    fields: tr(P_kappa), tr(Sigma_theta), cond(J_q_lin), qeff_tr, qeff_min, qeff_max
            print(",".join(str(x) for x in [
                8, k,
                f"{trPkappa:.6e}", f"{trSigma:.6e}", f"{condJq:.6e}",
                f"{qeff_tr:.6e}", f"{qeff_min:.6e}", f"{qeff_max:.6e}"
            ]))
            # 9: SL consistency / asin saturation proxy / limiter
            #    fields: SL_err, asin_min, asin_max, asin_sat_frac, du_inf, rate_hit
            print(",".join(str(x) for x in [
                9, k,
                f"{SL_err:.6e}", f"{asin_min:.6e}", f"{asin_max:.6e}", f"{asin_sat:.6e}",
                f"{du_inf:.6e}", rate_hit
            ]))
            # 10: per-joint asin_arg (if available), else NaN vector length n
            try:
                asin_list = [f"{v:.6e}" for v in asin_arg.tolist()]
            except Exception:
                asin_list = [f"{float('nan'):.6e}" for _ in range(self.n)]
            print(",".join([str(10), str(k)] + asin_list))
            # 11: Q_eff diagonal (if available)
            try:
                qeff_list = [f"{v:.6e}" for v in np.diag(Q_eff).tolist()]
            except Exception:
                qeff_list = [f"{float('nan'):.6e}" for _ in range(self.n)]
            print(",".join([str(11), str(k)] + qeff_list))
            # 12: y_kappa Jacobian diag and S_theta diag proxies (scale / sanity)
            try:
                Jyk_list = [f"{v:.6e}" for v in np.diag(J_ykappa).tolist()]
            except Exception:
                Jyk_list = [f"{float('nan'):.6e}" for _ in range(self.n)]
            try:
                Stheta_diag = np.diag(S_theta) if 'S_theta' in locals() else np.full((self.n,), float("nan"))
                Stheta_list = [f"{v:.6e}" for v in Stheta_diag.tolist()]
            except Exception:
                Stheta_list = [f"{float('nan'):.6e}" for _ in range(self.n)]

            # 13: WEKF internal timing breakdown (ms)
            try:
                wt = self.wekf.last_timing if (self.wekf.last_timing is not None) else {}
                fields = [
                    wt.get("pred_ms", float("nan")),
                    wt.get("eq_solve_ms", float("nan")),
                    wt.get("accum_ms", float("nan")),
                    wt.get("pinv_JqT_ms", float("nan")),
                    wt.get("stab_eig_ms", float("nan")),
                    wt.get("eig_Sinv_ms", float("nan")),
                    wt.get("chol_Sinv_ms", float("nan")),
                    wt.get("solve_St_y_ms", float("nan")),
                    wt.get("solve_Rinv_ms", float("nan")),
                    wt.get("qr_A_ms", float("nan")),
                    wt.get("solve_Rbar_ms", float("nan")),
                    wt.get("inv_Rbar_ms", float("nan")),
                    wt.get("wekf_update_ms", float("nan")),
                ]
            except Exception:
                fields = [float("nan")] * 13
            print(",".join([str(13), str(k)] + [f"{v:.3f}" for v in fields]))
            # 14: IMU/A_map timing (ms) — helps to rule out I/O/interp vs math
            print(",".join(str(x) for x in [
                14, k,
                f"{ms_imu:.3f}", f"{ms_amap:.3f}"
            ]))

            print(",".join([str(12), str(k)] + Jyk_list + Stheta_list))
            # 4: vectors (theta_ref, theta_eq, u_cmd)
            print(",".join([str(4), str(k)] + [f"{v:.6e}" for v in self.q_ref.tolist() + theta_eq.tolist() + u_cmd.tolist()]))
            # 5: a/beta/gamma vectors -> (a_vec, a_pub_vec, beta_eff_vec, gamma_vec)
            print(",".join([str(5), str(k)] + [f"{v:.6e}" for v in a_vec.tolist() + a_pub_vec.tolist() + beta_eff_vec.tolist() + gamma_vec.tolist()]))
            # 6: y_hat and tau_vec
            print(",".join([str(6), str(k)] + [f"{v:.6e}" for v in y_hat.tolist() + tau_vec.tolist()]))
            # 7: u_star
            print(",".join([str(7), str(k)] + [f"{v:.6e}" for v in u_star.tolist()]))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, type=str)
    ap.add_argument("--frames", type=str, default="link6")
    ap.add_argument("--topic-ref", type=str, default="/ref/joint_states")
    ap.add_argument("--topic-imu", type=str, default="/imu")
    ap.add_argument("--topic-cmd-out", type=str, default="/cmd/joint_states")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--A", type=float, default=100.0)
    ap.add_argument("--kp0", type=str, default="100,100,100,100,100,100")
    ap.add_argument("--kp-min", type=float, default=5)
    ap.add_argument("--kp-max", type=float, default=100)
    ap.add_argument("--q-proc", type=float, default=1e-3)

    # control layer
    ap.add_argument("--kp-smooth-alpha", type=float, default=0.2)
    ap.add_argument("--fb-tau-des", type=str, default="0.08")  # desired closed-loop speed (used to solve a_p)
    ap.add_argument("--inv-denom-min", type=float, default=1e-3)
    ap.add_argument("--rate-limit", type=float, default=2.0)

    # debug
    ap.add_argument("--dbg-interval", type=int, default=1)

    # --- RLS feature toggles (cmd_lag_ekf) --- (defaults updated per request)
    ap.add_argument("--rls-lambda", type=float, default=0.9)
    ap.add_argument("--rls-use-forgetting", action="store_true", default=True)
    ap.add_argument("--rls-no-forgetting", dest="rls_use_forgetting", action="store_false")

    ap.add_argument("--rls-use-normalize", action="store_true", default=False)
    ap.add_argument("--rls-no-normalize", dest="rls_use_normalize", action="store_false")
    ap.add_argument("--rls-norm-epsilon", type=float, default=1e-12)

    ap.add_argument("--rls-use-innov-clip", action="store_true", default=False)
    ap.add_argument("--rls-no-innov-clip", dest="rls_use_innov_clip", action="store_false")
    ap.add_argument("--rls-innov-clip", type=float, default=5.0)

    ap.add_argument("--rls-use-gating", action="store_true", default=False)
    ap.add_argument("--rls-no-gating", dest="rls_use_gating", action="store_false")
    ap.add_argument("--rls-e-min", type=float, default=1e-6)
    ap.add_argument("--rls-phi-norm-min", type=float, default=1e-8)

    ap.add_argument("--rls-use-projection", action="store_true", default=False)
    ap.add_argument("--rls-no-projection", dest="rls_use_projection", action="store_false")
    ap.add_argument("--tau-min", type=float, default=0.00)
    ap.add_argument("--tau-max", type=float, default=0.80)

    ap.add_argument("--lag-use-omega-ema", action="store_true", default=True)
    ap.add_argument("--lag-no-omega-ema", dest="lag_use_omega_ema", action="store_false")
    ap.add_argument("--lag-omega-alpha", type=float, default=0.2)

    args = ap.parse_args()

    frames = [s.strip() for s in args.frames.split(",") if s.strip()]
    kp0 = [float(x) for x in args.kp0.split(",") if x.strip()]
    try:
        tau_list = [float(x) for x in args.fb_tau_des.split(",") if x.strip()]
    except Exception:
        tau_list = [0.08]

    rospy.init_node("online_deflecomp_estimator", anonymous=False)
    EstimatorNode(
        urdf_path=args.urdf,
        frames=frames,
        topic_ref=args.topic_ref,
        topic_imu=args.topic_imu,
        topic_cmd_out=args.topic_cmd_out,
        dt=args.dt,
        A_param=args.A,
        kp0=kp0,
        kp_lim=(args.kp_min, args.kp_max),
        q_proc=args.q_proc,
        kp_smooth_alpha=args.kp_smooth_alpha,
        fb_tau_des=tau_list,
        inv_denom_min=args.inv_denom_min,
        rate_limit=args.rate_limit,
        dbg_interval=args.dbg_interval,
        rls_lambda=args.rls_lambda,
        rls_use_forgetting=args.rls_use_forgetting,
        rls_use_normalize=args.rls_use_normalize,
        rls_norm_epsilon=args.rls_norm_epsilon,
        rls_use_innov_clip=args.rls_use_innov_clip,
        rls_innov_clip=args.rls_innov_clip,
        rls_use_gating=args.rls_use_gating,
        rls_e_min=args.rls_e_min,
        rls_phi_norm_min=args.rls_phi_norm_min,
        rls_use_projection=args.rls_use_projection,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        lag_use_omega_ema=args.lag_use_omega_ema,
        lag_omega_alpha=args.lag_omega_alpha
    )
    rospy.spin()

if __name__ == "__main__":
    main()
