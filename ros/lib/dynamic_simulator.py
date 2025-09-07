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
