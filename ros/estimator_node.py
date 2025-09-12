#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from bisect import bisect_left
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple
import threading
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
        a,b,c,d = q
        return np.array([[a,-b,-c,-d],[b,a,-d,c],[c,d,a,-b],[d,-c,b,a]], dtype=float)
    def Rmat(q: np.ndarray) -> np.ndarray:
        w,x,y,z = q
        return np.array([[w,-x,-y,-z],[x,w,z,-y],[y,-z,w,x],[z,y,-x,w]], dtype=float)
    P = Lmat(xq) - Rmat(vq)
    return float(A_param) * (-0.25 * (P.T @ P))

@dataclass
class ImuSample:
    t: float
    g: np.ndarray  # unit 3-vector (gravity in that frame)
    w: np.ndarray  # angular velocity in LOCAL frame

class ImuBuffer:
    def __init__(self, maxlen: int = 4000) -> None:
        self.t_list: List[float] = []
        self.g_list: List[np.ndarray] = []
        self.w_list: List[np.ndarray] = []
        self.maxlen = int(maxlen)
        self.lock = threading.RLock()

    def push(self, t: float, g_dir: np.ndarray, w_loc: Optional[np.ndarray]) -> None:
        g = np.asarray(g_dir, dtype=float)
        g = g / (np.linalg.norm(g) + 1e-12)
        w = np.zeros(3, dtype=float) if w_loc is None else np.asarray(w_loc, dtype=float).reshape(3)
        with self.lock:
            idx = bisect_left(self.t_list, t)
            if idx < len(self.t_list) and abs(self.t_list[idx] - t) < 1e-12:
                self.t_list[idx] = t; self.g_list[idx] = g; self.w_list[idx] = w
            else:
                self.t_list.insert(idx, t); self.g_list.insert(idx, g); self.w_list.insert(idx, w)
                if len(self.t_list) > self.maxlen:
                    self.t_list.pop(0); self.g_list.pop(0); self.w_list.pop(0)

    def interpolate_g(self, t: float) -> Optional[np.ndarray]:
        with self.lock:
            if not self.t_list:
                return None
            if t <= self.t_list[0]:
                return self.g_list[0].copy()
            if t >= self.t_list[-1]:
                return self.g_list[-1].copy()
            idx = bisect_left(self.t_list, t)
            t0 = self.t_list[idx - 1]; t1 = self.t_list[idx]
            g0 = self.g_list[idx - 1]; g1 = self.g_list[idx]
            if t1 - t0 <= 1e-12:
                g = g1.copy()
            else:
                a = (t - t0) / (t1 - t0)
                g = (1.0 - a) * g0 + a * g1
            return g / (np.linalg.norm(g) + 1e-12)

    def interpolate_w(self, t: float) -> Optional[np.ndarray]:
        with self.lock:
            if not self.t_list:
                return None
            if t <= self.t_list[0]:
                return self.w_list[0].copy()
            if t >= self.t_list[-1]:
                return self.w_list[-1].copy()
            idx = bisect_left(self.t_list, t)
            t0 = self.t_list[idx - 1]; t1 = self.t_list[idx]
            w0 = self.w_list[idx - 1]; w1 = self.w_list[idx]
            if t1 - t0 <= 1e-12:
                return w1.copy()
            a = (t - t0) / (t1 - t0)
            return (1.0 - a) * w0 + a * w1

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
                 kp_smooth_alpha: float = 0.1,
                 fb_tau_des: Optional[List[float]] = None,
                 inv_denom_min: float = 1e-3,
                 noise_gain_max: float = 0.3,
                 mu_margin: float = 0.2,
                 rate_limit: float = 2.0,
                 u_min: Optional[List[float]] = None,
                 u_max: Optional[List[float]] = None) -> None:
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

        # Lag estimator (gyro-only)
        lag_cfg = CmdLagEKFConfig(
            dt=float(dt),
            tau_init=0.2, tau_min=0.0, tau_max=0.8, eps_tau=1e-2,
            rls_lambda=0.99, rls_P0=1e2, rls_ridge=1e-9,
            qy_diag=1e-6, qs_diag=1e-6, rk_diag=1e-6, ridge=1e-6,
            phi_norm_min=1e-8, e_min=1e-6, tau_pub_min=1e-5, tau_pub_max=10.0
        )
        self.cmdlag = CmdLagEKF(self.robot, self.frames, self.frame_ids, self.g_unit, lag_cfg)

        # State
        self.q_ref = np.zeros(self.n, dtype=float)
        self.have_ref = False
        self.last_cmd: Optional[np.ndarray] = None
        self.last_cmd_t: Optional[float] = None

        # IMU buffers
        self.imu_bufs: Dict[str, ImuBuffer] = {nm: ImuBuffer(maxlen=4000) for nm in self.frames}

        # ROS I/O
        self.sub_ref = rospy.Subscriber(topic_ref, JointState, self.cb_ref, queue_size=50)
        self.sub_imu = rospy.Subscriber(topic_imu, Imu, self.cb_imu, queue_size=400)
        self.pub_cmd = rospy.Publisher(topic_cmd_out, JointState, queue_size=10)
        self.pub_kp  = rospy.Publisher("/online_deflecomp/kp_hat", Float64MultiArray, queue_size=10)
        self.pub_kpc = rospy.Publisher("/online_deflecomp/kp_cov_diag", Float64MultiArray, queue_size=10)
        self.pub_tau = rospy.Publisher("/online_deflecomp/tau_hat", Float64MultiArray, queue_size=10)
        self.pub_tpub = rospy.Publisher("/online_deflecomp/tau_pub", Float64MultiArray, queue_size=10)

        # Control params
        self.noise_gain_max = float(noise_gain_max)
        self.mu_margin = float(mu_margin)
        self.dt = float(dt)
        self.inv_denom_min = float(inv_denom_min)
        self.rate_limit = float(rate_limit)
        self.u_min = np.array(u_min, dtype=float) if u_min is not None else None
        self.u_max = np.array(u_max, dtype=float) if u_max is not None else None

        # Desired time constants (for beta)
        if fb_tau_des is None:
            self.fb_tau_des = np.full((self.n,), 0.08, dtype=float)
        else:
            v = np.array(fb_tau_des, dtype=float).reshape(-1)
            self.fb_tau_des = v if v.size == self.n else np.full((self.n,), float(v[0]), dtype=float)

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
        if nm not in self.frames:
            return
        t = make_timestamp_sec(msg.header.stamp)
        # build gravity dir in frame from orientation (world -> frame)
        if msg.orientation is not None:
            w = float(msg.orientation.w); x = float(msg.orientation.x)
            y = float(msg.orientation.y); z = float(msg.orientation.z)
            R_wf = quat_to_rot_wxyz(w, x, y, z)
            g_f = R_wf.T @ (self.g_unit)
        else:
            g_f = None
        w_loc = None
        if msg.angular_velocity is not None:
            w_loc = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=float)
        if g_f is not None:
            self.imu_bufs[nm].push(t, g_f, w_loc)

    # -------- main timer --------
    def on_timer(self, event) -> None:
        if not self.have_ref:
            return
        now = rospy.Time.now().to_sec()

        # (0) collect IMU observations at 'now'
        g_obs: Dict[str, np.ndarray] = {}
        w_obs: Dict[str, np.ndarray] = {}
        for nm, buf in self.imu_bufs.items():
            g_f = buf.interpolate_g(now)
            if g_f is not None:
                g_obs[nm] = g_f
            w_f = buf.interpolate_w(now)
            if w_f is not None:
                w_obs[nm] = w_f

        # (1) current published command u_k for lag propagation
        u_k = self.last_cmd.copy() if self.last_cmd is not None else theta_cmd_from_theta_ref(self.robot, self.q_ref, np.exp(self.wekf.x))

        # (2) lag estimator -> time-aligned y_hat and tau_hat
        kp_for_lag = self.kp_hat_smooth if self.kp_hat_smooth is not None else np.exp(self.wekf.x)
        y_hat, tau_vec = self.cmdlag.update(
            u_k=u_k,
            g_obs=g_obs,
            omega_obs=w_obs if w_obs else None,
            kp_vec=kp_for_lag,
            solver=self.solver,
            theta_init=(self.wekf.last_theta_eq if self.wekf.last_theta_eq is not None else None)
        )
        self.pub_tau.publish(Float64MultiArray(data=tau_vec.tolist()))
        self.pub_tpub.publish(Float64MultiArray(data=self.cmdlag.get_tau_pub().tolist()))

        # (3) Build A_map from IMU gravity observations
        A_map: Dict[int, np.ndarray] = {}
        for nm, g_f in g_obs.items():
            if nm in self.frame_ids:
                A_map[self.frame_ids[nm]] = simple_bingham_unit(g_f, self.g_unit, self.A_param)

        # (4) EKF update for Kp using time-aligned y_hat
        theta_init = self.wekf.last_theta_eq if self.wekf.last_theta_eq is not None else y_hat
        if A_map:
            theta_eq = self.wekf.update_with_multi(
                self.solver, y_hat, A_map, self.robot,
                theta_init_eq_pred=theta_init, kp_lim=self.kp_lim
            )
        else:
            # no gravity obs; keep previous theta_eq or compute equilibrium at y_hat
            try:
                theta_eq = self.solver.solve(self.robot, theta_cmd=y_hat, kp_vec=np.exp(self.wekf.x), theta_init=theta_init)
                self.wekf.last_theta_eq = theta_eq.copy()
            except Exception:
                theta_eq = theta_init.copy()

        # (5) smooth Kp (EMA)
        kp_raw = np.clip(np.exp(self.wekf.x), self.kp_lim[0], self.kp_lim[1])
        if self.kp_hat_smooth is None:
            self.kp_hat_smooth = kp_raw.copy()
        else:
            a = float(np.clip(self.kp_smooth_alpha, 0.0, 1.0))
            self.kp_hat_smooth = (1.0 - a) * self.kp_hat_smooth + a * kp_raw
        self.pub_kp.publish(Float64MultiArray(data=self.kp_hat_smooth.tolist()))
        self.pub_kpc.publish(Float64MultiArray(data=np.diag(self.wekf.P).tolist()))

        # (6) compute S and L
        try:
            Htheta = self.robot.d_tau_gravity(theta_eq).astype(float)
        except Exception:
            Htheta = np.zeros((self.n, self.n), dtype=float)
        H = Htheta + np.diag(self.kp_hat_smooth)
        try:
            Hinv = np.linalg.pinv(H, rcond=1e-10)
        except Exception:
            Hinv = np.linalg.pinv(H + 1e-6 * np.eye(self.n))
        S = Hinv @ np.diag(self.kp_hat_smooth)  # (n x n)

        # a_vec from tau_hat
        a_vec = np.exp(-self.dt / np.maximum(tau_vec, 1e-6))
        # speed target
        beta_speed = np.exp(-self.dt / np.maximum(self.fb_tau_des, 1e-6))
        # noise bound: ((a - beta)*mu_max)/(1 - a + (a - beta)*mu_max) <= c
        # solve t = a - beta:
        #   t <= c*(1 - a)/(mu_max*(1 - c)), with 0 < c < 1
        Sinv = np.linalg.pinv(S, rcond=1e-10)
        # operator norm of S^{-1} (conservative, global)
        try:
            s_inv_norm = float(np.linalg.norm(Sinv, 2))
        except Exception:
            s_inv_norm = float(np.linalg.norm(Sinv))
        # effective c per joint from noise_gain_max
        # c = noise_gain_max / ||S^{-1}||, clamp to (0, 0.95)
        c_eff = np.clip(self.noise_gain_max / max(s_inv_norm, 1e-9), 1e-3, 0.95)
        mu_max = 1.0 + max(self.mu_margin, 0.0)  # uncertainty margin on S
        one = np.ones_like(a_vec)
        t_max = (c_eff * (one - a_vec)) / (mu_max * (1.0 - c_eff))
        beta_noise = a_vec - t_max
        beta_noise = np.clip(beta_noise, 0.0, 0.999)
        # final beta
        beta_vec = np.maximum(beta_speed, beta_noise)
        # feedback gain factor
        denom = np.maximum(1.0 - a_vec, self.inv_denom_min)
        lfac = (a_vec - beta_vec) / denom
        # L = diag(lfac) * S^{-1}
        L = (np.diag(lfac) @ Sinv)

        # (7) compute u* and feedback
        u_star = theta_cmd_from_theta_ref(self.robot, self.q_ref, self.kp_hat_smooth)
        theta_err = (theta_eq - self.q_ref)
        u_cmd = u_star - (L @ theta_err)

        # (8) safety: rate limit and bounds
        if self.last_cmd is not None:
            du = u_cmd - self.last_cmd
            max_step = self.rate_limit * self.dt
            du = np.clip(du, -max_step, max_step)
            u_cmd = self.last_cmd + du
        if self.u_min is not None:
            if self.u_min.size == 1:
                u_cmd = np.maximum(u_cmd, self.u_min[0])
            else:
                u_cmd = np.maximum(u_cmd, self.u_min)
        if self.u_max is not None:
            if self.u_max.size == 1:
                u_cmd = np.minimum(u_cmd, self.u_max[0])
            else:
                u_cmd = np.minimum(u_cmd, self.u_max)

        # (9) publish
        out = JointState()
        out.header.stamp = rospy.Time.now()
        out.name = self.model_joint_names
        out.position = u_cmd.tolist()
        self.pub_cmd.publish(out)

        # (10) book-keeping
        self.last_cmd = u_cmd.copy()
        self.last_cmd_t = now

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
    ap.add_argument("--kp-smooth-alpha", type=float, default=0.1)
    ap.add_argument("--fb-tau-des", type=str, default="0.08")
    ap.add_argument("--noise-gain-max", type=float, default=0.3)
    ap.add_argument("--mu-margin", type=float, default=0.2)
    ap.add_argument("--rate-limit", type=float, default=2.0)
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
        noise_gain_max=args.noise_gain_max,
        mu_margin=args.mu_margin,
        rate_limit=args.rate_limit
    )
    rospy.spin()

if __name__ == "__main__":
    main()
