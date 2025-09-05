import numpy as np
import pinocchio as pin

def se3_to_homog(M: pin.SE3) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = M.rotation
    T[:3, 3] = M.translation
    return T

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v.copy()
    return v / n
