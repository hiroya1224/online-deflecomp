import numpy as np
from ..utils.robot import RobotArm

def theta_cmd_from_theta_ref(robot: RobotArm, theta_ref: np.ndarray, kp_vec: np.ndarray) -> np.ndarray:
   """Closed-form S^1-consistent feedforward command.

   Solves at equilibrium (quasi-static):
       tau_spring = -tau_gravity(theta_ref)
       tau_spring = 2*Kp*sin((theta_ref - theta_cmd)/2)
    => theta_cmd = theta_ref - 2*asin( -tau_g(theta_ref)/(2*Kp) )

   This avoids iterative inverse solving and remains well-behaved near +-pi.
   """
   tau_g = robot.tau_gravity(theta_ref)
   kp_safe = np.maximum(kp_vec, 1e-12)
   asin_arg = -tau_g / (2.0 * kp_safe)
   # physical saturation; keep continuity by tight margins
   asin_arg = np.clip(asin_arg, -1.0 + 1e-9, 1.0 - 1e-9)
   return theta_ref - 2.0 * np.arcsin(asin_arg)

def lowpass_theta_cmd(theta_raw: np.ndarray,
                      theta_prev: np.ndarray,
                      dt: float,
                      tau: float = 0.2,
                      alpha: float = None) -> np.ndarray:
    """First-order low-pass (lag) filter for theta_cmd.
    y_{t} = y_{t-1} + a * (x_t - y_{t-1}), with
      a = 1 - exp(-dt / tau)  if alpha is None,
      else a = clip(alpha, 0, 1).

    Args:
        theta_raw: newly computed command (unfiltered)
        theta_prev: previous filtered command
        dt: time interval [s]
        tau: time constant [s] (ignored if alpha is given)
        alpha: optional direct smoothing factor in [0, 1]

    Returns:
        Filtered theta command.
    """
    # safety
    if theta_prev is None or dt is None:
        return np.asarray(theta_raw, dtype=float)

    if alpha is None:
        tau_safe = float(max(1e-6, tau))
        a = 1.0 - float(np.exp(-float(max(0.0, dt)) / tau_safe))
    else:
        a = float(max(0.0, min(1.0, alpha)))

    theta_raw = np.asarray(theta_raw, dtype=float)
    theta_prev = np.asarray(theta_prev, dtype=float)
    return theta_prev + a * (theta_raw - theta_prev)