from typing import Dict, List, Optional, Tuple
import numpy as np
from ..utils.robot import RobotArm
from ..utils.bingham import BinghamUtils
from ..controller.equilibrium import EquilibriumSolver, EquilibriumConfig

class ObservationBuilder:
    def __init__(self, robot_sim: RobotArm, g_base: np.ndarray, parameter_A: float = 100.0) -> None:
        self.robot_sim = robot_sim
        self.g_base = np.asarray(g_base, dtype=float)
        self.parameter_A = float(parameter_A)

    def build_A_multi(
        self,
        theta_cmd: np.ndarray,
        kp_true: np.ndarray,
        frame_names: List[str],
        newton_iter_true: int = 60,
        theta_ws_true: Optional[np.ndarray] = None,
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        cfg = EquilibriumConfig(maxiter=newton_iter_true)
        tmp_solver = EquilibriumSolver(cfg=cfg)
        theta_equil_true = tmp_solver.solve(
            robot=self.robot_sim, theta_cmd=theta_cmd, kp_vec=kp_true, theta_init=theta_ws_true
        )
        A_map: Dict[int, np.ndarray] = {}
        for name in frame_names:
            fid = self.robot_sim.get_frame_id(name)
            g_f = self.robot_sim.gravity_dir_in_frame(theta_equil_true, self.g_base, fid)

        # NOTE on Bingham/A matrix coordinate frames:
        # We intentionally pass (before_vec3=g_in_frame, after_vec3=g_in_world) so that the
        # Bingham construction matches the quaternion z used in EKF: z represents the
        # frame->WORLD rotation (world-relative). This aligns with Qz = G(z) for world angular
        # velocity and with J_w computed in the WORLD frame. We do NOT rename any functions
        # to avoid API changes.
            A_f = BinghamUtils.simple_bingham_unit(g_f, self.g_base, parameter=self.parameter_A)  # NOTE: flipped args to use world-relative rotation (frame->world). See comment below.
            A_map[fid] = A_f
        return A_map, theta_equil_true
