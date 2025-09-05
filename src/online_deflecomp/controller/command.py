import numpy as np
from ..utils.robot import RobotArm

def theta_cmd_from_theta_ref(robot: RobotArm, theta_ref: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
    tau_g = robot.tau_gravity(theta_ref)
    kp_safe = np.maximum(kp_vec, 1e-12)
    return theta_ref + (tau_g / kp_safe)
