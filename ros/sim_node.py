#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import rospy
from sensor_msgs.msg import JointState, Imu

# local lib (no catkin dependency)
from online_deflecomp.simulation.dynamic_simulator import DynamicSimulator, DynamicParams  # noqa: E402
from online_deflecomp.utils.robot import RobotArm

# ---- helpers (rpy/quat/imu parse) は前回提示のものと同じ ----
@dataclass
class ImuOffset:
    R_li: np.ndarray
    r_li: np.ndarray
    name: str

def rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0.0],[sy, cy, 0.0],[0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp],[0.0, 1.0, 0.0],[-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0],[0.0, cr, -sr],[0.0, sr, cr]])
    return Rz @ Ry @ Rx

def parse_imu_frames(spec: str) -> List[Tuple[str, ImuOffset]]:
    items = [s.strip() for s in spec.split(";") if s.strip()]
    out: List[Tuple[str, ImuOffset]] = []
    for it in items:
        name = it; r = np.zeros(3); R = np.eye(3)
        if "@" in it:
            parts = it.split("@"); name = parts[0].strip()
            if len(parts) >= 2 and parts[1]:
                r = np.array([float(x) for x in parts[1].split(",")], dtype=float)
            if len(parts) >= 3 and parts[2]:
                roll, pitch, yaw = [float(x) for x in parts[2].split(",")]
                R = rpy_to_R(roll, pitch, yaw)
        out.append((name, ImuOffset(R_li=R, r_li=r, name=name)))
    return out

def quat_xyzw_from_R(Rw: np.ndarray) -> Tuple[float, float, float, float]:
    t = float(np.trace(Rw))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (Rw[2,1] - Rw[1,2]) / s
        qy = (Rw[0,2] - Rw[2,0]) / s
        qz = (Rw[1,0] - Rw[0,1]) / s
    else:
        i = int(np.argmax([Rw[0,0], Rw[1,1], Rw[2,2]]))
        if i == 0:
            s = np.sqrt(1.0 + Rw[0,0] - Rw[1,1] - Rw[2,2]) * 2.0
            qx = 0.25 * s; qy = (Rw[0,1] + Rw[1,0]) / s; qz = (Rw[0,2] + Rw[2,0]) / s
            qw = (Rw[2,1] - Rw[1,2]) / s
        elif i == 1:
            s = np.sqrt(1.0 + Rw[1,1] - Rw[0,0] - Rw[2,2]) * 2.0
            qx = (Rw[0,1] + Rw[1,0]) / s; qy = 0.25 * s; qz = (Rw[1,2] + Rw[2,1]) / s
            qw = (Rw[0,2] - Rw[2,0]) / s
        else:
            s = np.sqrt(1.0 + Rw[2,2] - Rw[0,0] - Rw[1,1]) * 2.0
            qx = (Rw[0,2] + Rw[2,0]) / s; qy = (Rw[1,2] + Rw[2,1]) / s; qz = 0.25 * s
            qw = (Rw[1,0] - Rw[0,1]) / s
    return (float(qx), float(qy), float(qz), float(qw))

class SimNode:
    def __init__(self, urdf_path: str, dt: float, kp_true: List[float], zeta: float, vel_lim: float,
                 topic_cmd: str, topic_equil: str,
                 imu_frames: Optional[str] = None, imu_topic_base: str = "/imu",
                 integrator: str = "rk4", ref_tau: float = 0.04, ref_max_vel: float = 4.0,
                 eq_mode: str = "dynamic", tau_eq: float = 0.05) -> None:

        self.dt = float(dt)
        self.topic_cmd = topic_cmd
        self.topic_equil = topic_equil
        self.imu_topic_base = imu_topic_base.rstrip("/")

        self.robot = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
        self.n = self.robot.nv
        self.joint_names = [self.robot.model.names[j] for j in range(1, self.robot.model.njoints)]

        Ktrue = np.array(kp_true, dtype=float)
        if Ktrue.shape != (self.n,):
            rospy.logwarn("kp_true length mismatches dof; resizing"); Ktrue = np.resize(Ktrue, self.n)
        params = DynamicParams(
            K=Ktrue, D=None, zeta=float(zeta), q0_for_damp=np.zeros(self.n, dtype=float), use_pinv=True,
            limit_velocity=np.ones(self.n, dtype=float) * float(vel_lim),
            limit_position_low=self.robot.model.lowerPositionLimit,
            limit_position_high=self.robot.model.upperPositionLimit,
            integrator=integrator, ref_tau=ref_tau, ref_max_vel=ref_max_vel,
            eq_mode=eq_mode, tau_eq=tau_eq,   # ← 追加
        )
        self.sim = DynamicSimulator(self.robot, params)
        self.sim.reset(q=np.zeros(self.n), qd=np.zeros(self.n))

        # θcmd buffer（線形補間）
        self.cmd_t_prev: Optional[float] = None
        self.cmd_q_prev: Optional[np.ndarray] = None
        self.cmd_t_last: Optional[float] = None
        self.cmd_q_last: Optional[np.ndarray] = None
        self.have_cmd = False

        # pubs/subs
        self.pub_equil = rospy.Publisher(self.topic_equil, JointState, queue_size=10)
        self.sub_cmd = rospy.Subscriber(self.topic_cmd, JointState, self.cb_cmd, queue_size=50)

        # IMU publishers
        self.imu_offsets: Dict[str, ImuOffset] = {}
        self.imu_fids: Dict[str, int] = {}
        self.imu_pubs: Dict[str, rospy.Publisher] = {}
        if imu_frames is not None and imu_frames.strip():
            for lname, off in parse_imu_frames(imu_frames):
                fid = self.robot.get_frame_id(lname)
                self.imu_offsets[lname] = off
                self.imu_fids[lname] = fid
                topic = f"{self.imu_topic_base}"
                self.imu_pubs[lname] = rospy.Publisher(topic, Imu, queue_size=20)
                rospy.loginfo("IMU publisher: frame=%s -> %s", lname, topic)

        self.ready = True
        self.timer = rospy.Timer(rospy.Duration.from_sec(self.dt), self.on_timer)

    # --- θcmd callback ---
    def cb_cmd(self, msg: JointState) -> None:
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        q = np.zeros(self.n, dtype=float)
        pos = list(msg.position) if msg.position else []
        for i, n in enumerate(self.joint_names):
            j = name_to_idx.get(n, None)
            if j is not None and j < len(pos):
                q[i] = float(pos[j])
        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        if self.cmd_t_last is not None:
            self.cmd_t_prev, self.cmd_q_prev = self.cmd_t_last, self.cmd_q_last
        self.cmd_t_last, self.cmd_q_last = t, q
        self.have_cmd = True

    def _interp_cmd(self, t_now: float) -> np.ndarray:
        if self.cmd_t_last is None:
            return np.zeros(self.n, dtype=float)
        if self.cmd_t_prev is None or self.cmd_t_last <= self.cmd_t_prev + 1e-9:
            return self.cmd_q_last.copy()
        a = (t_now - self.cmd_t_prev) / (self.cmd_t_last - self.cmd_t_prev)
        a = np.clip(a, 0.0, 1.0)
        return (1.0 - a) * self.cmd_q_prev + a * self.cmd_q_last

    # --- IMU generation（前回提示と同じ） ---
    def publish_imus(self, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray, now: rospy.Time) -> None:
        import pinocchio as pin
        pin.forwardKinematics(self.robot.model, self.robot.data, q, qd, qdd)
        pin.updateFramePlacements(self.robot.model, self.robot.data)
        g_world = self.robot.model.gravity.linear
        for name, fid in self.imu_fids.items():
            off = self.imu_offsets[name]
            v6_l = pin.getFrameVelocity(self.robot.model, self.robot.data, fid, pin.ReferenceFrame.LOCAL)
            a6_l = pin.getFrameAcceleration(self.robot.model, self.robot.data, fid, pin.ReferenceFrame.LOCAL)
            w_l = np.array(v6_l.angular).reshape(3)
            alpha_l = np.array(a6_l.angular).reshape(3)
            a_o_l = np.array(a6_l.linear).reshape(3)
            r_li = off.r_li.reshape(3)
            a_p_l = a_o_l + np.cross(alpha_l, r_li) + np.cross(w_l, np.cross(w_l, r_li))
            R_wl = self.robot.data.oMf[fid].rotation
            R_li = off.R_li
            R_wi = R_wl @ R_li
            a_p_i = R_li.T @ a_p_l
            w_i = R_li.T @ w_l
            g_i = R_wi.T @ g_world
            a_meas = a_p_i - g_i
            qx, qy, qz, qw = quat_xyzw_from_R(R_wi)
            msg = Imu()
            msg.header.stamp = now
            msg.header.frame_id = name
            msg.orientation.x = qx; msg.orientation.y = qy; msg.orientation.z = qz; msg.orientation.w = qw
            msg.angular_velocity.x = float(w_i[0]); msg.angular_velocity.y = float(w_i[1]); msg.angular_velocity.z = float(w_i[2])
            msg.linear_acceleration.x = float(a_meas[0]); msg.linear_acceleration.y = float(a_meas[1]); msg.linear_acceleration.z = float(a_meas[2])
            msg.orientation_covariance[0] = -1.0
            msg.angular_velocity_covariance[0] = -1.0
            msg.linear_acceleration_covariance[0] = -1.0
            pub = self.imu_pubs.get(name)
            if pub is not None:
                pub.publish(msg)

    def on_timer(self, event) -> None:
        if not self.ready or not self.have_cmd:
            return
        theta_cmd = self._interp_cmd(rospy.get_time())
        # plant: q_ref = theta_cmd（バネの無負荷角）
        q_prev, qd_prev = self.sim.state()
        q_next, qd_next = self.sim.step(dt=self.dt, q_ref=theta_cmd, tau_ext=None)
        qdd_est = (qd_next - qd_prev) / max(self.dt, 1e-9)
        now = rospy.Time.now()
        # publish equilibrium (plant state)
        js = JointState(); js.header.stamp = now; js.name = self.joint_names
        js.position = [float(x) for x in q_next.tolist()]
        self.pub_equil.publish(js)
        # imu
        self.publish_imus(q_next, qd_next, qdd_est, now)

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", dest="urdf_path", type=str, required=True)
    p.add_argument("--dt", type=float, default=0.004)
    p.add_argument("--kp-true", type=str, default="60,60,40,40,20,20")
    p.add_argument("--zeta", type=float, default=0.9)
    p.add_argument("--vel", type=float, default=10.0)
    p.add_argument("--topic-cmd", type=str, default="/cmd/joint_states")      # subscribe
    p.add_argument("--topic-equil", type=str, default="/equil/joint_states")  # publish
    p.add_argument("--imu-frames", type=str, default="link2;link3;link6")
    p.add_argument("--imu-topic-base", type=str, default="/imu")
    p.add_argument("--integrator", type=str, default="rk4", choices=["rk4","semi_implicit_euler"])
    p.add_argument("--ref-tau", type=float, default=1e-9)
    p.add_argument("--ref-max-vel", type=float, default=1000.0)
    # 既存の argparse 定義の続きに追記
    p.add_argument("--eq-mode", type=str, default="dynamic",
                choices=["dynamic", "relax_to_eq", "quasistatic"],
                help="equilibrium tracking mode")
    p.add_argument("--tau-eq", type=float, default=0.05,
                help="time constant [s] for relax_to_eq")

    args = p.parse_args()

    kp_true = [float(x) for x in args.kp_true.split(",") if x.strip()]

    rospy.init_node("online_deflecomp_sim", anonymous=False)
    SimNode(
        urdf_path=args.urdf_path, dt=args.dt, kp_true=kp_true, zeta=args.zeta, vel_lim=args.vel,
        topic_cmd=args.topic_cmd, topic_equil=args.topic_equil,
        imu_frames=args.imu_frames, imu_topic_base=args.imu_topic_base,
        integrator=args.integrator, ref_tau=args.ref_tau, ref_max_vel=args.ref_max_vel,
        eq_mode=args.eq_mode, tau_eq=args.tau_eq,  # ← 追加
    )
    rospy.spin()

if __name__ == "__main__":
    main()
