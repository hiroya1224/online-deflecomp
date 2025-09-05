import numpy as np
from ikpy.chain import Chain as IkChain
from ..utils.geometry import se3_to_homog

class RikGaikPlanner:
    def __init__(self, chain: IkChain) -> None:
        self.chain = chain

    def rik_solve(self, T_target, theta_init: np.ndarray, max_iter: int = 1200) -> np.ndarray:
        T_h = se3_to_homog(T_target)
        q0_full = np.zeros(len(self.chain.links), dtype=float)
        q0_full[1 : 1 + theta_init.size] = theta_init
        sol_full = self.chain.inverse_kinematics_frame(T_h, initial_position=q0_full, max_iter=max_iter, orientation_mode="all")
        return np.array(sol_full[1 : 1 + theta_init.size], dtype=float)

    @staticmethod
    def make_theta_ref_path(theta_init: np.ndarray, theta_ref_goal: np.ndarray, n_steps: int) -> np.ndarray:
        a = np.linspace(0.0, 1.0, n_steps)
        return (1 - a)[:, None] * theta_init[None, :] + a[:, None] * theta_ref_goal[None, :]
