# ros/lib/dynamic_simulator.py
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
    # new:
    integrator: str = "rk4"              # "rk4" or "semi_implicit_euler"
    ref_tau: Optional[float] = 0.03      # [s] 1st-order low-pass for q_ref (None/<=0 to disable)
    ref_max_vel: Optional[float] = 4.0   # [rad/s] slew limit for q_ref (None to disable)

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
                q0 = 0.5 * (lo + hi) if (lo.shape[0] == n and hi.shape[0] == n) else np.zeros(n)
            M0 = pin.crba(self.robot.model, self.robot.data, q0)
            M0 = 0.5 * (M0 + M0.T)
            Mdiag = np.clip(np.diag(M0), 1e-6, np.inf)
            self.D = 2.0 * float(params.zeta) * np.sqrt(params.K * Mdiag)
        else:
            self.D = params.D.copy()

        self.q = np.zeros(n, dtype=float)
        self.qd = np.zeros(n, dtype=float)

        self.vel_lim = params.limit_velocity
        lo = self.robot.model.lowerPositionLimit
        hi = self.robot.model.upperPositionLimit
        self.pos_lo = params.limit_position_low if params.limit_position_low is not None else (lo if lo.shape[0] == n else None)
        self.pos_hi = params.limit_position_high if params.limit_position_high is not None else (hi if hi.shape[0] == n else None)

        # reference shaping states
        self.q_ref_filt = np.zeros(n, dtype=float)
        self.q_ref_prev = np.zeros(n, dtype=float)

    def reset(self, q: Optional[np.ndarray] = None, qd: Optional[np.ndarray] = None) -> None:
        n = self.robot.nv
        self.q = np.zeros(n, dtype=float) if q is None else q.astype(float, copy=True)
        self.qd = np.zeros(n, dtype=float) if qd is None else qd.astype(float, copy=True)
        self.q_ref_filt = self.q.copy()
        self.q_ref_prev = self.q.copy()

    def state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.qd.copy()

    def _shape_reference(self, dt: float, q_ref_in: np.ndarray) -> np.ndarray:
        # 1) slew limit
        q_ref = q_ref_in.astype(float, copy=False)
        if self.params.ref_max_vel is not None and self.params.ref_max_vel > 0.0:
            dv = np.clip(q_ref - self.q_ref_prev,
                         -abs(self.params.ref_max_vel) * dt,
                         +abs(self.params.ref_max_vel) * dt)
            q_ref_slew = self.q_ref_prev + dv
        else:
            q_ref_slew = q_ref
        self.q_ref_prev = q_ref_slew

        # 2) first-order low-pass
        if self.params.ref_tau is not None and self.params.ref_tau > 0.0:
            alpha = 1.0 - np.exp(-dt / float(self.params.ref_tau))
            self.q_ref_filt = self.q_ref_filt + alpha * (q_ref_slew - self.q_ref_filt)
        else:
            self.q_ref_filt = q_ref_slew
        return self.q_ref_filt

    def _dyn_rhs(self, q: np.ndarray, qd: np.ndarray, q_ref_eff: np.ndarray,
                 tau_ext: Optional[np.ndarray]) -> np.ndarray:
        """Return qdd using M(q) qdd = tau_ext + tau_spring - b(q,qd)"""
        n = self.robot.nv
        if tau_ext is None:
            tau_ext = np.zeros(n, dtype=float)

        M = pin.crba(self.robot.model, self.robot.data, q)
        M = 0.5 * (M + M.T)
        b = pin.rnea(self.robot.model, self.robot.data, q, qd, np.zeros(n, dtype=float))
        tau_spring = self.params.K * (q_ref_eff - q) - self.D * qd
        rhs = (tau_ext + tau_spring) - b
        if self.params.use_pinv:
            qdd = np.linalg.pinv(M, rcond=1e-12) @ rhs
        else:
            qdd = np.linalg.solve(M, rhs)
        return qdd

    def step(self, dt: float, q_ref: np.ndarray, tau_ext: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        n = self.robot.nv
        if q_ref.shape != (n,):
            raise ValueError(f"q_ref must have shape ({n},), got {q_ref.shape}")
        
        # add noise (simulating motor vibration)
        q_ref = q_ref + np.random.randn(q_ref.shape[0]) * 5 * np.pi/180

        # reference shaping
        q_ref_eff = self._shape_reference(dt, q_ref)

        if self.params.integrator.lower() == "rk4":
            # state y = [q, qd]
            q0 = self.q.copy()
            v0 = self.qd.copy()

            def f(q: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                a = self._dyn_rhs(q, v, q_ref_eff, tau_ext)
                return v, a

            k1_v, k1_a = f(q0, v0)
            k2_v, k2_a = f(q0 + 0.5*dt*k1_v, v0 + 0.5*dt*k1_a)
            k3_v, k3_a = f(q0 + 0.5*dt*k2_v, v0 + 0.5*dt*k2_a)
            k4_v, k4_a = f(q0 + dt*k3_v,    v0 + dt*k3_a)

            q_next  = q0 + (dt/6.0) * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v)
            qd_next = v0 + (dt/6.0) * (k1_a + 2.0*k2_a + 2.0*k3_a + k4_a)

        else:
            # semi-implicit (symplectic) Euler
            a = self._dyn_rhs(self.q, self.qd, q_ref_eff, tau_ext)
            qd_next = self.qd + dt * a
            q_next  = self.q + dt * qd_next

        if self.vel_lim is not None:
            vlim = np.asarray(self.vel_lim, dtype=float)
            qd_next = np.clip(qd_next, -np.abs(vlim), np.abs(vlim))
        if self.pos_lo is not None and self.pos_hi is not None:
            q_next = np.minimum(np.maximum(q_next, self.pos_lo), self.pos_hi)

        self.q, self.qd = q_next, qd_next
        return q_next.copy(), qd_next.copy()
