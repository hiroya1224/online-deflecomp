#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64MultiArray

from online_deflecomp.utils.robot import RobotArm
from online_deflecomp.controller.command import theta_cmd_from_theta_ref
from online_deflecomp.estimator.ekf import MultiFrameWeirdEKF
from online_deflecomp.estimator.cmd_lag_ekf import CmdLagEKF, CmdLagEKFConfig
from online_deflecomp.controller.equilibrium import EquilibriumSolver, EquilibriumConfig

import threading
from bisect import bisect_left


# ---------- small helpers ----------
@dataclass
class ImuSample:
    t: float          # sec
    g: np.ndarray     # unit 3-vector (gravity dir in that frame)
    w: np.ndarray     # angular velocity in LOCAL frame (rad/s)

class ImuBuffer:
    """Time-sorted buffer with linear interpolation for g (unit) and w (3,)."""
    def __init__(self, maxlen: int = 2000) -> None:
        self.t_list: List[float] = []
        self.g_list: List[np.ndarray] = []
        self.w_list: List[np.ndarray] = []
        self.maxlen = int(maxlen)
        self.lock = threading.RLock()

    def push(self, t: float, g_dir: np.ndarray, w_loc: Optional[np.ndarray]) -> None:
        g = np.asarray(g_dir, dtype=float)
        n = float(np.linalg.norm(g)) + 1e-12
        g = g / n
        w = np.zeros(3, dtype=float) if w_loc is None else np.asarray(w_loc, dtype=float).reshape(3)
        with self.lock:
            idx = bisect_left(self.t_list, t)
            if idx < len(self.t_list) and abs(self.t_list[idx] - t) < 1e-12:
                self.t_list[idx] = t
                self.g_list[idx] = g
                self.w_list[idx] = w
            else:
                self.t_list.insert(idx, t)
                self.g_list.insert(idx, g)
                self.w_list.insert(idx, w)
            while len(self.t_list) > self.maxlen:
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
                return g1.copy()
            a = (t - t0) / (t1 - t0)
            g = (1.0 - a) * g0 + a * g1
            n = float(np.linalg.norm(g)) + 1e-12
            return g / n

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
            w = (1.0 - a) * w0 + a * w1
            return w

def simple_bingham_unit(before_vec3: np.ndarray, after_vec3: np.ndarray, parameter: float = 100.0) -> np.ndarray:
    # World vs frame rotation (frame->world): use (g_frame, g_world_unit)
    b = np.asarray(before_vec3, dtype=float); a = np.asarray(after_vec3, dtype=float)
    bn = b / (np.linalg.norm(b) + 1e-12); an = a / (np.linalg.norm(a) + 1e-12)
    vq = np.array([0.0, bn[0], bn[1], bn[2]], dtype=float)
    xq = np.array([0.0, an[0], an[1], an[2]], dtype=float)
    def Lmat(q: np.ndarray) -> np.ndarray:
        a,b,c,d = q
        return np.array([[a,-b,-c,-d],[b,a,-d,c],[c,d,a,-b],[d,-c,b,a]], dtype=float)
    def Rmat(q: np.ndarray) -> np.ndarray:
        w,x,y,z = q
        return np.array([[w,-x,-y,-z],[x,w,z,-y],[y,-z,w,x],[z,y,-x,w]], dtype=float)
    P = Lmat(xq) - Rmat(vq)
    return float(parameter) * (-0.25 * (P.T @ P))

def map_jointstate_to_model(msg: JointState, model_names: List[str]) -> np.ndarray:
    name_to_idx = {n: i for i, n in enumerate(msg.name)}
    q = np.zeros(len(model_names), dtype=float)
    pos = list(msg.position) if msg.position else []
    for i, n in enumerate(model_names):
        j = name_to_idx.get(n, None)
        if j is not None and j < len(pos):
            q[i] = float(pos[j])
    return q

def lpf_first_order_vec(x_raw: np.ndarray, x_prev: np.ndarray, dt: float, tau_vec: np.ndarray) -> np.ndarray:
    """Elementwise 1st-order low-pass: alpha = dt / (tau + dt)."""
    tau = np.maximum(np.asarray(tau_vec, dtype=float).reshape(-1), 1e-6)
    a = dt / (tau + dt)
    return (1.0 - a) * x_prev + a * x_raw

# ---------- main node ----------
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
                 kp_lim: Tuple[float],
                 q_proc: float) -> None:
        self.robot = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
        self.solver = EquilibriumSolver(EquilibriumConfig(maxiter=80))
        self.n = self.robot.nv
        self.model_joint_names = [self.robot.model.names[j] for j in range(1, self.robot.model.njoints)]

        self.frames = frames
        self.frame_ids: Dict[str, int] = {nm: self.robot.get_frame_id(nm) for nm in self.frames}
        self.g_world = np.array([0.0, 0.0, -9.81], dtype=float)
        self.g_unit = self.g_world / np.linalg.norm(self.g_world)
        self.A_param = float(A_param)

        # WEKF
        x0 = np.log(np.resize(np.array(kp0, dtype=float), self.n))
        P0 = np.eye(self.n) * 1.0
        Q  = np.eye(self.n) * float(q_proc)
        self.wekf = MultiFrameWeirdEKF(x0, P0, Q, eps_def=1e-6)
        self.kp_lim = kp_lim

        # Lag estimator (gyro-only RLS) + per-joint public LPF tau
        lag_tau_init = float(rospy.get_param("~lag_tau_init", 0.2))
        lag_tau_min  = float(rospy.get_param("~lag_tau_min", 0.0))
        lag_tau_max  = float(rospy.get_param("~lag_tau_max", 0.8))
        lag_cfg = CmdLagEKFConfig(
            dt=float(dt),
            # tau bounds/init
            tau_init=lag_tau_init,
            tau_min=lag_tau_min,
            tau_max=lag_tau_max,
            eps_tau=float(rospy.get_param("~lag_eps_tau", 1e-2)),
            # RLS hyper-params
            rls_lambda=float(rospy.get_param("~lag_rls_lambda", 0.99)),
            rls_P0=float(rospy.get_param("~lag_rls_P0", 1e2)),
            rls_ridge=float(rospy.get_param("~lag_rls_ridge", 1e-9)),
            # numeric guards / placeholders
            qy_diag=float(rospy.get_param("~lag_qy_diag", 1e-6)),
            qs_diag=float(rospy.get_param("~lag_qs_diag", 1e-6)),
            rk_diag=float(rospy.get_param("~lag_rk_diag", 1e-6)),
            ridge=float(rospy.get_param("~lag_ridge", 1e-6)),
            phi_norm_min=float(rospy.get_param("~lag_phi_norm_min", 1e-8)),
            e_min=float(rospy.get_param("~lag_e_min", 1e-6)),
            tau_pub_min=float(rospy.get_param("~lag_tau_pub_min", 1e-5)),
            tau_pub_max=float(rospy.get_param("~lag_tau_pub_max", 10.0)),
        )
        self.cmdlag = CmdLagEKF(self.robot, self.frames, self.frame_ids, self.g_unit, lag_cfg)

        # states
        self.q_ref = np.zeros(self.n, dtype=float)
        self.have_ref = False
        self.last_cmd: Optional[np.ndarray] = None
        self.last_cmd_t: Optional[float] = None

        # IMU buffers (store g and omega in LOCAL)
        self.imu_bufs: Dict[str, ImuBuffer] = {nm: ImuBuffer(maxlen=4000) for nm in self.frames}

        # ROS I/O
        self.sub_ref = rospy.Subscriber(topic_ref, JointState, self.cb_ref, queue_size=50)
        self.sub_imu = rospy.Subscriber(topic_imu, Imu, self.cb_imu, queue_size=400)
        self.pub_cmd = rospy.Publisher(topic_cmd_out, JointState, queue_size=10)
        self.pub_kp  = rospy.Publisher("/online_deflecomp/kp_hat", Float64MultiArray, queue_size=10)
        self.pub_kpc = rospy.Publisher("/online_deflecomp/kp_cov_diag", Float64MultiArray, queue_size=10)
        self.pub_tau = rospy.Publisher("/online_deflecomp/tau_hat", Float64MultiArray, queue_size=10)
        self.pub_tpub = rospy.Publisher("/online_deflecomp/tau_pub", Float64MultiArray, queue_size=10)

        self.dt = float(dt)
        self.timer = rospy.Timer(rospy.Duration.from_sec(self.dt), self.on_timer)

        rospy.loginfo("estimator_node: frames=%s", ", ".join(self.frames))

    # --- callbacks ---
    def cb_ref(self, msg: JointState) -> None:
        self.q_ref = map_jointstate_to_model(msg, self.model_joint_names)
        self.have_ref = True

    def cb_imu(self, msg: Imu) -> None:
        name = (msg.header.frame_id or "").strip()
        if name not in self.imu_bufs:
            return
        # gravity dir from linear acceleration: specific force ≈ a - g → at rest, -g
        la = np.array([msg.linear_acceleration.x,
                       msg.linear_acceleration.y,
                       msg.linear_acceleration.z], dtype=float)
        if np.linalg.norm(la) < 1e-9:
            return
        g_dir = -la / (np.linalg.norm(la) + 1e-12)
        w_loc = np.array([msg.angular_velocity.x,
                          msg.angular_velocity.y,
                          msg.angular_velocity.z], dtype=float)
        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        self.imu_bufs[name].push(t, g_dir, w_loc)

    # --- timer loop ---
    def on_timer(self, event) -> None:
        if not self.have_ref:
            return

        now = rospy.Time.now().to_sec()

        # Build imu observations (interpolated at 'now')
        g_obs: Dict[str, np.ndarray] = {}
        w_obs: Dict[str, np.ndarray] = {}
        for nm, buf in self.imu_bufs.items():
            g_f = buf.interpolate_g(now)
            if g_f is not None:
                g_obs[nm] = g_f
            w_f = buf.interpolate_w(now)
            if w_f is not None:
                w_obs[nm] = w_f

        # Input for lag estimator is the last published command (what plant is trying to follow)
        u_k_for_lag = self.last_cmd if self.last_cmd is not None else self.q_ref.copy()

        # (1) Lag EKF/RLS update: get y_hat (current commanded-as-seen-by-plant) and tau_hat
        # pass current Kp estimate (from previous WEKF step) for sensitivity S
        kp_for_lag = np.exp(self.wekf.x)
        y_for_ekf, tau_vec = self.cmdlag.update(
            u_k=u_k_for_lag, g_obs=g_obs, omega_obs=w_obs if w_obs else None,
            kp_vec=kp_for_lag, solver=self.solver
        )
        self.pub_tau.publish(Float64MultiArray(data=tau_vec.tolist()))

        # (2) Build A_map using y_hat's time alignment (IMU vs model)
        if g_obs:
            A_map = self._build_A_map_from_gobs(g_obs)
            theta_init = self.wekf.last_theta_eq if self.wekf.last_theta_eq is not None else self.q_ref
            _theta_eq = self.wekf.update_with_multi(
                self.solver, y_for_ekf, A_map, self.robot,
                theta_init_eq_pred=theta_init, kp_lim=self.kp_lim
            )
            kp_hat = np.exp(self.wekf.x)
            self.pub_kp.publish(Float64MultiArray(data=kp_hat.tolist()))
            cov_diag = np.clip(np.diag(self.wekf.P), 0.0, np.inf)
            self.pub_kpc.publish(Float64MultiArray(data=cov_diag.tolist()))
        else:
            kp_hat = np.exp(self.wekf.x)

        # (3) Compute raw theta_cmd from theta_ref and current Kp
        theta_cmd_raw = theta_cmd_from_theta_ref(self.robot, self.q_ref, kp_hat)

        # (4) Apply per-joint public LPF (safe poles): tau_pub = dt * a / (1 - a), a = exp(-dt / tau_hat)
        tau_pub = self.cmdlag.get_tau_pub()
        self.pub_tpub.publish(Float64MultiArray(data=tau_pub.tolist()))
        if self.last_cmd is None:
            theta_cmd = theta_cmd_raw.copy()
        else:
            theta_cmd = lpf_first_order_vec(theta_cmd_raw, self.last_cmd, dt=self.dt, tau_vec=tau_pub)

        # (5) publish command and update internal last_cmd
        self.last_cmd = theta_cmd.copy()
        self.last_cmd_t = now

        js = JointState()
        js.header.stamp = rospy.Time.from_sec(now)
        js.name = self.model_joint_names
        js.position = [float(x) for x in theta_cmd.tolist()]
        self.pub_cmd.publish(js)

    # --- helpers ---
    def _build_A_map_from_gobs(self, gobs: Dict[str, np.ndarray]) -> Optional[Dict[int, np.ndarray]]:
        A_map: Dict[int, np.ndarray] = {}
        ok = False
        for nm, g_f in gobs.items():
            if nm not in self.frame_ids:
                continue
            A_map[self.frame_ids[nm]] = simple_bingham_unit(g_f, self.g_unit, parameter=self.A_param)  # world-relative rotation (frame->world)
            ok = True
        return A_map if ok else None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, type=str)
    ap.add_argument("--frames", type=str, default="link6,link3,link2")
    ap.add_argument("--topic-ref", type=str, default="/ref/joint_states")
    ap.add_argument("--topic-imu", type=str, default="/imu")
    ap.add_argument("--topic-cmd-out", type=str, default="/cmd/joint_states")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--A", type=float, default=100.0)
    ap.add_argument("--kp0", type=str, default="50,50,50,50,50,50")
    ap.add_argument("--kp-min", type=float, default=5)
    ap.add_argument("--kp-max", type=float, default=500)
    ap.add_argument("--q-proc", type=float, default=1e-3)
    args = ap.parse_args()

    frames = [s.strip() for s in args.frames.split(",") if s.strip()]
    kp0 = [float(x) for x in args.kp0.split(",") if x.strip()]

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
        q_proc=args.q_proc
    )
    rospy.spin()

if __name__ == "__main__":
    main()
