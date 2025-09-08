from typing import List
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as Rsc

class RobotArm:
    def __init__(self, urdf_path: str, tip_link: str = 'link6', base_link: str = 'base_link') -> None:
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.nv = self.model.nv
        self.tip_fid = self.model.getFrameId(tip_link)
        self.base_fid = self.model.getFrameId(base_link)
        if hasattr(self.model, 'gravity'):
            self.model.gravity.linear = np.array([0.0, 0.0, -9.81], dtype=float)
        self.total_mass = float(sum(inert.mass for inert in self.model.inertias))

    def get_frame_id(self, frame_name: str) -> int:
        return self.model.getFrameId(frame_name)

    def _fk_update(self, theta: np.ndarray) -> None:
        pin.forwardKinematics(self.model, self.data, theta)
        pin.updateFramePlacements(self.model, self.data)

    def frame_rotation_in_base(self, theta: np.ndarray, fid: int) -> np.ndarray:
        self._fk_update(theta)
        R_wb = self.data.oMf[self.base_fid].rotation
        R_wf = self.data.oMf[fid].rotation
        return R_wb.T @ R_wf

    def frame_quaternion_wxyz_base(self, theta: np.ndarray, fid: int) -> np.ndarray:
        R_bf = self.frame_rotation_in_base(theta, fid)
        q_xyzw = Rsc.from_matrix(R_bf).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
        n = np.linalg.norm(q_wxyz) + 1e-18
        return q_wxyz / n

    def frame_angular_jacobian_world(self, theta: np.ndarray, fid: int) -> np.ndarray:
        pin.computeJointJacobians(self.model, self.data, theta)
        pin.updateFramePlacements(self.model, self.data)
        J6 = pin.computeFrameJacobian(self.model, self.data, theta, fid, pin.ReferenceFrame.WORLD)
        return J6[3:6, :]

    def gravity_dir_in_frame(self, theta: np.ndarray, g_base: np.ndarray, fid: int) -> np.ndarray:
        # NOTE: Despite the argument name `g_base`, this function now interprets the input
        # as the gravity vector expressed in the WORLD frame and returns the unit gravity
        # direction expressed in the *frame* coordinates. This change is to ensure consistency
        # with the Bingham construction which compares WORLD gravity to the link-frame gravity.
        # (Function name is kept for compatibility; do NOT rename.)
        self._fk_update(theta)
        R_wf = self.data.oMf[fid].rotation
        gw = g_base / (np.linalg.norm(g_base) + 1e-12)
        gf = R_wf.T @ gw
        return gf / (np.linalg.norm(gf) + 1e-12)
        Ts.append(pin.SE3(self.data.oMf[self.tip_fid]))
        return Ts

    def tau_gravity(self, theta: np.ndarray) -> np.ndarray:
        return pin.computeGeneralizedGravity(self.model, self.data, theta)

    def d_tau_gravity(self, theta: np.ndarray) -> np.ndarray:
        return pin.computeGeneralizedGravityDerivatives(self.model, self.data, theta)

    def potential_gravity(self, theta: np.ndarray) -> float:
        com = pin.centerOfMass(self.model, self.data, theta)
        g = self.model.gravity.linear
        return -self.total_mass * float(np.dot(g, com))
