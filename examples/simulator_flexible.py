import numpy as np
import matplotlib.pyplot as plt

# If you placed the simulator inside online_deflecomp:
from online_deflecomp.utils.robot import RobotArm
from online_deflecomp.simulation.dynamic_simulator import DynamicSimulator, DynamicParams

# Otherwise, adapt imports to your local package name / path.
# from your_pkg.utils.robot import RobotArm
# from your_pkg.simulation.dynamic_simulator import DynamicSimulator, DynamicParams


def make_reference_trajectory(robot: RobotArm, T: int, seed: int = 3) -> np.ndarray:
    """
    Build a smooth joint reference trajectory (fake IK output).
    """
    rng = np.random.default_rng(seed)
    lo = robot.model.lowerPositionLimit
    hi = robot.model.upperPositionLimit
    n = robot.nv

    q_start = np.zeros(n, dtype=float)
    q_goal = np.array([rng.uniform(l, h) for l, h in zip(lo, hi)], dtype=float)

    t = np.linspace(0.0, 1.0, T)
    alpha = t * t * (3.0 - 2.0 * t)  # smoothstep 0->1
    q_ref = (1.0 - alpha)[:, None] * q_start[None, :] + alpha[:, None] * q_goal[None, :]

    # Add mild sinusoid to excite vibrations
    amp = 0.05 * (hi - lo)
    q_ref += (amp[None, :] * np.sin(2.0 * np.pi * (3.0 * t)[:, None]))
    q_ref = np.clip(q_ref, lo, hi)
    return q_ref


def main() -> None:
    urdf_path = "simple6r.urdf"  # ensure the file exists
    robot = RobotArm(urdf_path, tip_link="link6", base_link="base_link")

    n = robot.nv
    T = 4000
    dt = 0.001  # 1 ms simulation step
    q_ref_seq = make_reference_trajectory(robot, T=T)

    # Stiffness (try lowering to see more visible deflection)
    K = np.array([18.0, 12.0, 14.0, 9.0, 7.0, 5.0], dtype=float) * 2.0
    # Let the simulator auto-build D from zeta and M(q0)
    params = DynamicParams(
        K=K,
        D=None,
        zeta=0.03,  # light damping to allow oscillations
        q0_for_damp=np.zeros(n, dtype=float),
        use_pinv=True,
        limit_velocity=np.ones(n, dtype=float) * 5.0,  # rad/s
        limit_position_low=robot.model.lowerPositionLimit,
        limit_position_high=robot.model.upperPositionLimit,
    )

    sim = DynamicSimulator(robot, params)
    sim.reset(q=np.zeros(n, dtype=float), qd=np.zeros(n, dtype=float))

    Q, Qd = sim.simulate(dt=dt, q_ref_seq=q_ref_seq)

    # Plot a few joints
    t = np.arange(T) * dt
    idx_to_plot = [0, 2, 5]
    fig, axes = plt.subplots(len(idx_to_plot), 1, figsize=(8, 7), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, j in zip(axes, idx_to_plot):
        ax.plot(t, q_ref_seq[:, j], label=f"q_ref[{j}]")
        ax.plot(t, Q[:, j], label=f"q[{j}]")
        ax.set_ylabel("rad")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Flexible joint simulation (spring-mass-damper, semi-implicit Euler)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
