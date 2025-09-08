#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys
from typing import Dict, List, Optional
import numpy as np
import rospy
from sensor_msgs.msg import JointState

# local lib (no catkin dependency)
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from online_deflecomp.simulation.dynamic_simulator import DynamicSimulator, DynamicParams  # noqa: E402

from online_deflecomp.utils.robot import RobotArm
from online_deflecomp.controller.command import theta_cmd_from_theta_ref

def map_joint_order(source_names: List[str], target_names: List[str]) -> List[Optional[int]]:
    idx: Dict[str, int] = {n: i for i, n in enumerate(source_names)}
    return [idx.get(n, None) for n in target_names]

def apply_prefix_strip(names: List[str], prefix: str) -> List[str]:
    if not prefix: return names
    plen = len(prefix); return [n[plen:] if n.startswith(prefix) else n for n in names]

class SimNode:
    def __init__(self, urdf_path: str, dt: float, kp_list: List[float], zeta: float, vel_lim: float,
                 strip_prefix_ref: str, topic_ref: str, topic_cmd: str, topic_equil: str) -> None:
        self.dt = float(dt); self.strip_prefix_ref = strip_prefix_ref
        self.topic_ref = topic_ref; self.topic_cmd = topic_cmd; self.topic_equil = topic_equil

        self.robot = RobotArm(urdf_path, tip_link="link6", base_link="base_link")
        self.n = self.robot.nv
        self.joint_names = [self.robot.model.names[j] for j in range(1, self.robot.model.njoints)]

        K = np.array(kp_list, dtype=float)
        if K.shape != (self.n,): rospy.logwarn("kp length mismatches dof; resizing"); K = np.resize(K, self.n)
        params = DynamicParams(
            K=K, D=None, zeta=float(zeta), q0_for_damp=np.zeros(self.n, dtype=float), use_pinv=True,
            limit_velocity=np.ones(self.n, dtype=float) * float(vel_lim),
            limit_position_low=self.robot.model.lowerPositionLimit,
            limit_position_high=self.robot.model.upperPositionLimit,
        )
        self.sim = DynamicSimulator(self.robot, params); self.sim.reset(q=np.zeros(self.n), qd=np.zeros(self.n))

        self.q_ref = np.zeros(self.n); self.have_ref = False; self.map_ref_to_model: Optional[List[Optional[int]]] = None
        self.pub_cmd = rospy.Publisher(self.topic_cmd, JointState, queue_size=10)
        self.pub_equil = rospy.Publisher(self.topic_equil, JointState, queue_size=10)
        self.sub_ref = rospy.Subscriber(self.topic_ref, JointState, self.cb_ref, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration.from_sec(self.dt), self.on_timer)

        # 追加: 参照バッファ
        self.ref_t_prev: Optional[float] = None
        self.ref_q_prev: Optional[np.ndarray] = None
        self.ref_t_last: Optional[float] = None
        self.ref_q_last: Optional[np.ndarray] = None

    def cb_ref(self, msg: JointState) -> None:
        in_names = apply_prefix_strip(list(msg.name), self.strip_prefix_ref)
        in_pos = list(msg.position) if msg.position else [0.0] * len(in_names)
        if self.map_ref_to_model is None:
            self.map_ref_to_model = map_joint_order(in_names, self.joint_names)
            missing = [self.joint_names[i] for i, m in enumerate(self.map_ref_to_model) if m is None]
            if missing: rospy.logwarn("missing joints in /joint_states_ref: %s", ", ".join(missing))
        q_ref = np.zeros(self.n)
        for i, src in enumerate(self.map_ref_to_model):
            if src is not None and src < len(in_pos): q_ref[i] = float(in_pos[src])
        self.q_ref = q_ref
        self.have_ref = True
        # 参照バッファ更新
        t = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()
        if self.ref_t_last is not None:
            self.ref_t_prev, self.ref_q_prev = self.ref_t_last, self.ref_q_last
        self.ref_t_last, self.ref_q_last = t, q_ref.copy()
            
    def _interp_ref(self, t_now: float) -> np.ndarray:
        # 直近 2 点の線形補間（なければ最後の値／ゼロ）
        if self.ref_t_last is None:
            return np.zeros(self.n, dtype=float)
        if self.ref_t_prev is None or self.ref_t_last <= self.ref_t_prev + 1e-9:
            return self.ref_q_last.copy()
        a = (t_now - self.ref_t_prev) / (self.ref_t_last - self.ref_t_prev)
        a = np.clip(a, 0.0, 1.0)
        return (1.0 - a) * self.ref_q_prev + a * self.ref_q_last

    def on_timer(self, event) -> None:
        if not self.have_ref:
            q_ref_now = np.zeros(self.n, dtype=float)
        else:
            q_ref_now = self._interp_ref(rospy.get_time())

        kp_vec = self.sim.params.K
        theta_cmd = theta_cmd_from_theta_ref(self.robot, q_ref_now, kp_vec)
        q_next, _ = self.sim.step(dt=self.dt, q_ref=theta_cmd, tau_ext=None)
        now = rospy.Time.now()
        self.pub_cmd.publish(self.build_js(self.joint_names, theta_cmd, now))
        self.pub_equil.publish(self.build_js(self.joint_names, q_next, now))


    @staticmethod
    def build_js(names: List[str], pos: np.ndarray, stamp: rospy.Time) -> JointState:
        msg = JointState(); msg.header.stamp = stamp; msg.name = list(names); msg.position = [float(x) for x in pos.tolist()]; return msg

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--urdf", dest="urdf_path", type=str, required=True)
    p.add_argument("--dt", type=float, default=0.004)
    p.add_argument("--kp", type=str, default="50,50,50,50,50,50")
    p.add_argument("--zeta", type=float, default=0.05)
    p.add_argument("--vel", type=float, default=5.0)
    p.add_argument("--strip-prefix-ref", type=str, default="")
    p.add_argument("--topic-ref", type=str, default="/ref/joint_states")
    p.add_argument("--topic-cmd", type=str, default="/cmd/joint_states")
    p.add_argument("--topic-equil", type=str, default="/equil/joint_states")
    args = p.parse_args()
    kp_list = [float(x) for x in args.kp.split(",") if x.strip()]

    rospy.init_node("online_deflecomp_sim", anonymous=False)
    SimNode(
        urdf_path=args.urdf_path, dt=args.dt, kp_list=kp_list, zeta=args.zeta, vel_lim=args.vel,
        strip_prefix_ref=args.strip_prefix_ref, topic_ref=args.topic_ref, topic_cmd=args.topic_cmd, topic_equil=args.topic_equil,
    )
    rospy.spin()

if __name__ == "__main__":
    main()
