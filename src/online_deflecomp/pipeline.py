from typing import List, Tuple, Optional
import numpy as np
from .utils.urdf_utils import require_urdf
from .utils.robot import RobotArm
from .controller.equilibrium import EquilibriumSolver, EquilibriumConfig
from .controller.command import theta_cmd_from_theta_ref
from .estimator.observations import ObservationBuilder
from .estimator.ekf import MultiFrameWeirdEKF

def run_estimation_pipeline(
    urdf_path: str,
    frames: List[str],
    theta_ref_path: np.ndarray,
    kp_init: np.ndarray,
    g_base: Optional[np.ndarray] = None,
    parameter_A: float = 100.0,
    newton_iter_true: int = 60,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if g_base is None:
        g_base = np.array([0.0, 0.0, -9.81], dtype=float)

    abs_urdf = require_urdf(urdf_path)

    robot_sim = RobotArm(abs_urdf, tip_link='link6', base_link='base_link')
    robot_est = RobotArm(abs_urdf, tip_link='link6', base_link='base_link')

    T, n = theta_ref_path.shape
    assert kp_init.shape[0] == n

    x0 = np.log(kp_init)
    P0 = np.eye(n) * 1.0
    Q = np.eye(n) * 1e-3
    wekf = MultiFrameWeirdEKF(x0, P0, Q, eps_def=1e-6)

    obs_builder = ObservationBuilder(robot_sim, g_base, parameter_A=parameter_A)
    solver_default = EquilibriumSolver(EquilibriumConfig(maxiter=80))

    theta_cmd_seq = np.zeros_like(theta_ref_path)
    theta_ws_true: Optional[np.ndarray] = None
    theta_eq_pred_prev: Optional[np.ndarray] = None

    for k in range(T):
        theta_ref_k = theta_ref_path[k]
        kp_hat = np.exp(wekf.x)

        theta_cmd_k = theta_cmd_from_theta_ref(robot_est, theta_ref_k, kp_hat)
        theta_cmd_seq[k] = theta_cmd_k

        A_map, theta_equil_true_k = obs_builder.build_A_multi(
            theta_cmd=theta_cmd_k,
            kp_true=kp_hat,
            frame_names=frames,
            newton_iter_true=newton_iter_true,
            theta_ws_true=theta_ws_true,
        )
        theta_ws_true = theta_equil_true_k

        theta_eq_used = wekf.update_with_multi(
            solver=solver_default,
            theta_cmd=theta_cmd_k,
            A_map=A_map,
            robot_est=robot_est,
            theta_init_eq_pred=(theta_eq_pred_prev if theta_eq_pred_prev is not None else theta_ref_k),
        )
        theta_eq_pred_prev = theta_eq_used

    kp_est = np.exp(wekf.x)
    P_est = wekf.P
    return theta_cmd_seq, kp_est, P_est
