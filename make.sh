# ① ディレクトリ作成
mkdir -p ros/lib ros/launch ros/rviz

# ② シミュレータ本体（rospy 直起動）
cat > ros/sim_node.py << 'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys
from typing import Dict, List, Optional
import numpy as np
import rospy
from sensor_msgs.msg import JointState

# local lib (no catkin dependency)
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from dynamic_simulator import DynamicSimulator, DynamicParams  # noqa: E402

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
        self.q_ref = q_ref; self.have_ref = True

    def on_timer(self, event) -> None:
        if not self.have_ref: self.q_ref = np.zeros(self.n)
        kp_vec = self.sim.params.K
        theta_cmd = theta_cmd_from_theta_ref(self.robot, self.q_ref, kp_vec)
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
    p.add_argument("--kp", type=str, default="18,12,14,9,7,5")
    p.add_argument("--zeta", type=float, default=0.03)
    p.add_argument("--vel", type=float, default=5.0)
    p.add_argument("--strip-prefix-ref", type=str, default="")
    p.add_argument("--topic-ref", type=str, default="/joint_states_ref")
    p.add_argument("--topic-cmd", type=str, default="/joint_states_cmd")
    p.add_argument("--topic-equil", type=str, default="/joint_states_equil")
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
PY
chmod +x ros/sim_node.py

# ③ ローカル動力学ライブラリ
cat > ros/lib/dynamic_simulator.py << 'PY'
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pinocchio as pin

@dataclass
class DynamicParams:
    K: np.ndarray
    D: Optional[np.ndarray] = None
    zeta: float = 0.05
    q0_for_damp: Optional[np.ndarray] = None
    use_pinv: bool = True
    limit_velocity: Optional[np.ndarray] = None
    limit_position_low: Optional[np.ndarray] = None
    limit_position_high: Optional[np.ndarray] = None

class DynamicSimulator:
    def __init__(self, robot, params: DynamicParams) -> None:
        self.robot = robot
        self.params = params
        n = self.robot.nv
        if params.K.shape != (n,):
            raise ValueError(f"K must have shape ({n},), got {params.K.shape}")
        if params.D is not None and params.D.shape != (n,):
            raise ValueError(f"D must have shape ({n},), got {params.D.shape}")
        if params.D is None:
            q0 = params.q0_for_damp
            if q0 is None:
                lo = self.robot.model.lowerPositionLimit
                hi = self.robot.model.upperPositionLimit
                if lo.shape[0] == n and hi.shape[0] == n:
                    q0 = 0.5 * (lo + hi)
                else:
                    q0 = np.zeros(n, dtype=float)
            M0 = pin.crba(self.robot.model, self.robot.data, q0)
            M0 = 0.5 * (M0 + M0.T)
            Mdiag = np.clip(np.diag(M0), 1e-6, np.inf)
            self.D = 2.0 * float(params.zeta) * np.sqrt(params.K * Mdiag)
        else:
            self.D = params.D.copy()
        self.q = np.zeros(n, dtype=float)
        self.qd = np.zeros(n, dtype=float)
        self.vel_lim = params.limit_velocity
        self.pos_lo = params.limit_position_low
        self.pos_hi = params.limit_position_high
        if self.pos_lo is None or self.pos_hi is None:
            lo = self.robot.model.lowerPositionLimit
            hi = self.robot.model.upperPositionLimit
            if lo.shape[0] == n and hi.shape[0] == n:
                self.pos_lo = lo.copy() if self.pos_lo is None else self.pos_lo
                self.pos_hi = hi.copy() if self.pos_hi is None else self.pos_hi

    def reset(self, q: Optional[np.ndarray] = None, qd: Optional[np.ndarray] = None) -> None:
        n = self.robot.nv
        self.q = np.zeros(n, dtype=float) if q is None else q.astype(float, copy=True)
        self.qd = np.zeros(n, dtype=float) if qd is None else qd.astype(float, copy=True)

    def state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.qd.copy()

    def set_state(self, q: np.ndarray, qd: np.ndarray) -> None:
        n = self.robot.nv
        if q.shape != (n,) or qd.shape != (n,):
            raise ValueError(f"q, qd must both have shape ({n},)")
        self.q = q.astype(float, copy=True)
        self.qd = qd.astype(float, copy=True)

    def step(self, dt: float, q_ref: np.ndarray, tau_ext: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        n = self.robot.nv
        if q_ref.shape != (n,):
            raise ValueError(f"q_ref must have shape ({n},), got {q_ref.shape}")
        if tau_ext is None:
            tau_ext = np.zeros(n, dtype=float)
        else:
            if tau_ext.shape != (n,):
                raise ValueError(f"tau_ext must have shape ({n},), got {tau_ext.shape}")
        M = pin.crba(self.robot.model, self.robot.data, self.q); M = 0.5 * (M + M.T)
        b = pin.nonLinearEffects(self.robot.model, self.robot.data, self.q, self.qd)
        tau_act = -self.params.K * (self.q - q_ref) - self.D * self.qd
        rhs = (tau_ext + tau_act) - b
        qdd = (np.linalg.pinv(M, rcond=1e-12) @ rhs) if self.params.use_pinv else np.linalg.solve(M, rhs)
        qd_next = self.qd + dt * qdd
        if self.vel_lim is not None:
            vlim = np.asarray(self.vel_lim, dtype=float); qd_next = np.clip(qd_next, -np.abs(vlim), np.abs(vlim))
        q_next = self.q + dt * qd_next
        if self.pos_lo is not None and self.pos_hi is not None:
            q_next = np.minimum(np.maximum(q_next, self.pos_lo), self.pos_hi)
        self.q, self.qd = q_next, qd_next
        return q_next.copy(), qd_next.copy()
PY

# ④ launch（URDFをそのまま各nsにloadし、frame_prefixで衝突回避）
cat > ros/launch/deflecomp_frames.launch << 'XML'
<launch>
  <arg name="urdf_path" default="$(env PWD)/simple6r.urdf"/>
  <arg name="dt" default="0.004"/>

  <group ns="ref">
    <rosparam param="robot_description" command="load" file="$(arg urdf_path)"/>
    <node pkg="joint_state_publisher_gui" type="joint_state_publisher_gui" name="jsp_gui" output="screen">
      <remap from="/joint_states" to="/joint_states_ref"/>
    </node>
    <node pkg="robot_state_publisher" type="robot_state_publisher" name="rsp_ref" output="screen">
      <param name="frame_prefix" value="ref_"/>
      <remap from="/joint_states" to="/joint_states_ref"/>
    </node>
  </group>

  <group ns="cmd">
    <rosparam param="robot_description" command="load" file="$(arg urdf_path)"/>
    <node pkg="robot_state_publisher" type="robot_state_publisher" name="rsp_cmd" output="screen">
      <param name="frame_prefix" value="cmd_"/>
      <remap from="/joint_states" to="/joint_states_cmd"/>
    </node>
  </group>

  <group ns="equil">
    <rosparam param="robot_description" command="load" file="$(arg urdf_path)"/>
    <node pkg="robot_state_publisher" type="robot_state_publisher" name="rsp_equil" output="screen">
      <param name="frame_prefix" value="eq_"/>
      <remap from="/joint_states" to="/joint_states_equil"/>
    </node>
  </group>

  <node pkg="rviz" type="rviz" name="rviz" args="-d $(env PWD)/ros/rviz/deflecomp.rviz" required="false" />
</launch>
XML

# ⑤ RViz の最小設定
cat > ros/rviz/deflecomp.rviz << 'RVIZ'
Panels:
  - Class: rviz/Displays
    Name: Displays
  - Class: rviz/Views
    Name: Views
Visualization Manager:
  Global Options:
    Fixed Frame: eq_base_link
  Displays:
    - Class: rviz/RobotModel
      Name: Equil
      Robot Description: /equil/robot_description
      Enabled: true
    - Class: rviz/RobotModel
      Name: Cmd
      Robot Description: /cmd/robot_description
      Enabled: true
    - Class: rviz/RobotModel
      Name: Ref
      Robot Description: /ref/robot_description
      Enabled: true
  Views:
    Current:
      Class: rviz/Orbit
      Distance: 1.5
RVIZ
