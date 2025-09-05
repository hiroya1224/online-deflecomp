import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
from .robot import RobotArm

class Visualizer:
    @staticmethod
    def draw_frame(ax, T: pin.SE3, axis_len: float = 0.06) -> None:
        o = T.translation
        Rm = T.rotation
        cols = ['r', 'g', 'b']
        for k in range(3):
            a = Rm[:, k] * axis_len
            ax.plot([o[0], o[0]+a[0]], [o[1], o[1]+a[1]], [o[2], o[2]+a[2]], cols[k], linewidth=1.8)
        ax.scatter([o[0]], [o[1]], [o[2]], s=35, c='k')

    @staticmethod
    def joint_positions(robot: RobotArm, theta: np.ndarray) -> np.ndarray:
        robot._fk_update(theta)
        pts = [robot.data.oMi[jid].translation for jid in range(1, robot.model.njoints)]
        pts.append(robot.data.oMf[robot.tip_fid].translation)
        return np.vstack(pts)

    def show_triplet(self, robot_sim: RobotArm, robot_est: RobotArm, theta_cmd_final: np.ndarray, kp_true: np.ndarray, kp_est: np.ndarray, T_target_se3: pin.SE3, title: str = 'Final comparison') -> None:
        P_rigid = self.joint_positions(robot_sim, theta_cmd_final)
        from ..controller.equilibrium import EquilibriumSolver, EquilibriumConfig
        theta_equil_true = EquilibriumSolver(EquilibriumConfig(maxiter=80)).solve(robot_sim, theta_cmd_final, kp_true, theta_init=theta_cmd_final)
        P_true = self.joint_positions(robot_sim, theta_equil_true)
        theta_equil_est = EquilibriumSolver(EquilibriumConfig(maxiter=80)).solve(robot_est, theta_cmd_final, np.maximum(kp_est, 1e-8), theta_init=theta_cmd_final)
        P_est = self.joint_positions(robot_est, theta_equil_est)

        fig = plt.figure(figsize=(7,7)); ax = fig.add_subplot(111, projection='3d')
        ax.plot(P_rigid[:,0], P_rigid[:,1], P_rigid[:,2], 'o--', label='sim rigid (theta_cmd)')
        ax.plot(P_true[:,0],  P_true[:,1],  P_true[:,2],  'o-',  label='sim gravity (Kp_true)')
        ax.plot(P_est[:,0],   P_est[:,1],   P_est[:,2],   'o-.', label='est gravity (Kp_est)')
        self.draw_frame(ax, T_target_se3, axis_len=0.08)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.set_title(title); ax.legend()
        all_pts = np.vstack([P_rigid, P_true, P_est, T_target_se3.translation.reshape(1,3)])
        c = all_pts.mean(axis=0); r = max(np.max(np.linalg.norm(all_pts-c, axis=1)), 0.25)
        ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
        ax.view_init(elev=25, azim=45); plt.gca().set_box_aspect([1,1,1]); plt.tight_layout(); plt.show()
