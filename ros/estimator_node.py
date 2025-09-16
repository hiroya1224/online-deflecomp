#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This node estimates Kp (via EKF with Bingham gravity residuals) and plant lag kappa (via gyro-only RLS),
then publishes a stabilized theta_cmd. It also prints numeric-only debug lines (IDs 1..7) for analysis.

Command-line options include per-feature toggles for the RLS stabilizers:
  forgetting / normalization / innovation-clip / gating / projection, and gyro EMA.

Debug CSV line formats (first field is line-type ID):
1,k,t,dt_ms,ms_cmdlag,ms_wekf,ms_eq,ms_total,n_g,n_w,had_A_map,wekf_updated,eq_solved
2,k,condH,condS,normSinv,lfac_min,lfac_max,denom_min,beta_min,beta_max,a_min,a_max
3,k,norm_theta_err,norm_du_raw,norm_du_lim,rate_hit,sat_min_hit,sat_max_hit
4,k,theta_ref...,theta_eq...,u_cmd...
5,k,a_vec...,beta_speed...,beta_vec...,lfac...
6,k,y_hat...,tau_vec...
7,k,u_star...
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

# ---------------- helpers ----------------
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
        self.last_cmd: Optional[np.ndarray] = None
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
        g_obs: Dict[str, np.ndarray] = {}
        w_obs: Dict[str, np.ndarray] = {}
        for nm in self.frames:
            g_f, w_f = self.imu.interp(nm, now)
            if g_f is not None:
                g_obs[nm] = g_f
            if w_f is not None:
                w_obs[nm] = w_f
        n_g = len(g_obs); n_w = len(w_obs)

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
        self.pub_tpub.publish(Float64MultiArray(data=self.cmdlag.get_tau_pub().tolist()))

        # (3) Build A_map from IMU gravity observations (for Bingham WEKF)
        A_map: Dict[int, np.ndarray] = {}
        for nm, g_f in g_obs.items():
            if nm in self.frame_ids:
                A_map[self.frame_ids[nm]] = simple_bingham_unit(g_f, self.g_unit, self.A_param)
        had_A_map = 1 if len(A_map) > 0 else 0

        # (4) EKF update for Kp using time-aligned y_hat
        t_wekf0 = time.perf_counter()
        theta_init = self.wekf.last_theta_eq if self.wekf.last_theta_eq is not None else y_hat
        wekf_updated = 0
        if A_map:
            theta_eq = self.wekf.update_with_multi(
                self.solver, y_hat, A_map, self.robot,
                theta_init_eq_pred=theta_init, kp_lim=self.kp_lim
            )
            wekf_updated = 1
        else:
            try:
                theta_eq = self.solver.solve(self.robot, theta_cmd=y_hat, kp_vec=np.exp(self.wekf.x), theta_init=theta_init)
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

        # (6) compute S and control gain L (certainty equivalence)
        t_eq0 = time.perf_counter()
        try:
            Htheta = self.robot.d_tau_gravity(theta_eq).astype(float)
        except Exception:
            Htheta = np.zeros((self.n, self.n), dtype=float)
        H = Htheta + np.diag(self.kp_hat_smooth)
        try:
            Hinv = np.linalg.pinv(H, rcond=1e-10)
        except Exception:
            Hinv = np.linalg.pinv(H + 1e-6 * np.eye(self.n))
        S = Hinv @ np.diag(self.kp_hat_smooth)

        a_vec = np.exp(-self.dt / np.maximum(tau_vec, 1e-6))
        beta_speed = np.exp(-self.dt / np.maximum(self.fb_tau_des, 1e-6))
        beta_vec = beta_speed.copy()  # (ここでノイズ上限制約等を入れる場合は、この行で調整)
        denom = np.maximum(1.0 - a_vec, self.inv_denom_min)
        lfac = (a_vec - beta_vec) / denom

        try:
            Sinv = np.linalg.pinv(S, rcond=1e-10)
        except Exception:
            Sinv = np.linalg.pinv(S + 1e-6 * np.eye(self.n))
        L = (np.diag(lfac) @ Sinv)
        t_eq1 = time.perf_counter()

        # (7) compute u* and feedback
        u_star = theta_cmd_from_theta_ref(self.robot, self.q_ref, self.kp_hat_smooth)
        theta_err = (theta_eq - self.q_ref)
        u_cmd_raw = u_star - (L @ theta_err)

        # (8) safety: rate limit
        rate_hit = 0
        if self.last_cmd is not None:
            du = u_cmd_raw - self.last_cmd
            max_step = self.rate_limit * self.dt
            du_lim = np.clip(du, -max_step, max_step)
            if np.any(np.abs(du) > max_step + 1e-12):
                rate_hit = 1
            u_cmd = self.last_cmd + du_lim
        else:
            du = u_cmd_raw.copy()
            du_lim = du.copy()
            u_cmd = u_cmd_raw.copy()

        # (9) publish
        out = JointState()
        out.header.stamp = rospy.Time.now()
        out.name = self.model_joint_names
        out.position = u_cmd.tolist()
        self.pub_cmd.publish(out)

        # (10) book-keeping
        self.last_cmd = u_cmd.copy()
        self.last_cmd_t = now

        # -------- debug prints (numeric-only) --------
        if (k % self.dbg_interval) == 0:
            t_step1 = time.perf_counter()
            ms_cmdlag = (t_cmdlag1 - t_cmdlag0) * 1000.0
            ms_wekf = (t_wekf1 - t_wekf0) * 1000.0
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
            lfac_min = float(np.min(lfac)); lfac_max = float(np.max(lfac))
            denom_min = float(np.min(denom))
            beta_min = float(np.min(beta_vec)); beta_max = float(np.max(beta_vec))
            a_min = float(np.min(a_vec)); a_max = float(np.max(a_vec))
            norm_theta_err = float(np.linalg.norm(theta_err))
            norm_du_raw = float(np.linalg.norm(du))
            norm_du_lim = float(np.linalg.norm(du_lim))

            # 1: summary
            print(",".join(str(x) for x in [
                1, k, f"{t_rel:.6f}", f"{self.dt*1000.0:.3f}",
                f"{ms_cmdlag:.3f}", f"{ms_wekf:.3f}", f"{ms_eq:.3f}", f"{ms_total:.3f}",
                n_g, n_w, 1 if had_A_map else 0, 1 if wekf_updated else 0, 1
            ]))
            # 2: gains/conds
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
            # 4: vectors
            print(",".join([str(4), str(k)] + [f"{v:.6e}" for v in self.q_ref.tolist() + theta_eq.tolist() + u_cmd.tolist()]))
            # 5: a/beta/lfac vectors
            print(",".join([str(5), str(k)] + [f"{v:.6e}" for v in a_vec.tolist() + beta_speed.tolist() + beta_vec.tolist() + lfac.tolist()]))
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
    ap.add_argument("--kp0", type=str, default="50,50,50,50,50,50")
    ap.add_argument("--kp-min", type=float, default=5)
    ap.add_argument("--kp-max", type=float, default=500)
    ap.add_argument("--q-proc", type=float, default=1e-3)

    # control layer
    ap.add_argument("--kp-smooth-alpha", type=float, default=0.2)
    ap.add_argument("--fb-tau-des", type=str, default="0.08")
    ap.add_argument("--inv-denom-min", type=float, default=1e-3)
    ap.add_argument("--rate-limit", type=float, default=2.0)

    # debug
    ap.add_argument("--dbg-interval", type=int, default=1)

    # --- RLS feature toggles (cmd_lag_ekf) ---
    ap.add_argument("--rls-lambda", type=float, default=0.99)
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
