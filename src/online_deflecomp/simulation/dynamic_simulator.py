from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pinocchio as pin

# If you are placing this inside the online-deflecomp package, keep this import:
# from online_deflecomp.utils.robot import RobotArm
# Otherwise, adapt the import path to your project structure.
from online_deflecomp.utils.robot import RobotArm


@dataclass
class DynamicParams:
    """
    Spring-mass-damper parameters for joint-space dynamics.

    qdd = M(q)^{-1} * ( tau_ext + tau_act - b(q, qd) ),
    where:
      tau_act = -K (q - q_ref) - D qd
      b = C(q, qd) qd + g(q)

    Notes:
      - K and D are per-joint diagonal gains.
      - You can set D = None and provide (zeta) to auto-compute a diagonal D
        from M(q0) and K: D_i = 2*zeta*sqrt(K_i * Mii(q0)).
    """
    K: np.ndarray                           # shape (n,)
    D: Optional[np.ndarray] = None          # shape (n,) or None (auto)
    zeta: float = 0.05                      # damping ratio for auto-D
    q0_for_damp: Optional[np.ndarray] = None  # used to estimate M(q0) for auto-D
    use_pinv: bool = True                   # use pinv for M^{-1} (robust), else solve
    limit_velocity: Optional[np.ndarray] = None  # shape (n,) max |qd|
    limit_position_low: Optional[np.ndarray] = None
    limit_position_high: Optional[np.ndarray] = None


class DynamicSimulator:
    """
    Simple joint-space spring-mass-damper simulator on top of Pinocchio.

    Integrator: semi-implicit Euler (a.k.a. symplectic Euler).
    - qd_{k+1} = qd_k + dt * qdd(q_k, qd_k, ...)
    - q_{k+1}  = q_k  + dt * qd_{k+1}

    Provides light oscillatory behavior when K is large and D is small.
    """

    def __init__(self, robot: RobotArm, params: DynamicParams) -> None:
        self.robot = robot
        self.params = params

        n = self.robot.nv
        if params.K.shape != (n,):
            raise ValueError(f"K must have shape ({n},), got {params.K.shape}")

        if params.D is not None and params.D.shape != (n,):
            raise ValueError(f"D must have shape ({n},), got {params.D.shape}")

        # Auto-compute D from zeta if not provided
        if params.D is None:
            q0 = params.q0_for_damp
            if q0 is None:
                # default: middle of limits (if available), else zeros
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

        # State
        self.q = np.zeros(n, dtype=float)
        self.qd = np.zeros(n, dtype=float)

        # Limits
        self.vel_lim = params.limit_velocity
        self.pos_lo = params.limit_position_low
        self.pos_hi = params.limit_position_high

        # If limits not provided, use model's limits when available
        if self.pos_lo is None or self.pos_hi is None:
            lo = self.robot.model.lowerPositionLimit
            hi = self.robot.model.upperPositionLimit
            if lo.shape[0] == n and hi.shape[0] == n:
                self.pos_lo = lo.copy() if self.pos_lo is None else self.pos_lo
                self.pos_hi = hi.copy() if self.pos_hi is None else self.pos_hi

    # ---------------------- public API ----------------------

    def reset(self, q: Optional[np.ndarray] = None, qd: Optional[np.ndarray] = None) -> None:
        n = self.robot.nv
        if q is None:
            self.q = np.zeros(n, dtype=float)
        else:
            if q.shape != (n,):
                raise ValueError(f"q must have shape ({n},)")
            self.q = q.astype(float, copy=True)
        if qd is None:
            self.qd = np.zeros(n, dtype=float)
        else:
            if qd.shape != (n,):
                raise ValueError(f"qd must have shape ({n},)")
            self.qd = qd.astype(float, copy=True)

    def state(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.q.copy(), self.qd.copy()

    def set_state(self, q: np.ndarray, qd: np.ndarray) -> None:
        n = self.robot.nv
        if q.shape != (n,) or qd.shape != (n,):
            raise ValueError(f"q, qd must both have shape ({n},)")
        self.q = q.astype(float, copy=True)
        self.qd = qd.astype(float, copy=True)

    def step(
        self,
        dt: float,
        q_ref: np.ndarray,
        tau_ext: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advance one time step with semi-implicit Euler.

        Args:
            dt: integration time step [s]
            q_ref: desired joint positions (virtual spring anchor), shape (n,)
            tau_ext: external joint torques, shape (n,), default zeros

        Returns:
            (q_next, qd_next)
        """
        n = self.robot.nv
        if q_ref.shape != (n,):
            raise ValueError(f"q_ref must have shape ({n},), got {q_ref.shape}")
        if tau_ext is None:
            tau_ext = np.zeros(n, dtype=float)
        else:
            if tau_ext.shape != (n,):
                raise ValueError(f"tau_ext must have shape ({n},), got {tau_ext.shape}")

        # M(q)
        M = pin.crba(self.robot.model, self.robot.data, self.q)
        M = 0.5 * (M + M.T)

        # b(q,qd) = C(q,qd) + g(q)  ← RNEAで取得（qdd=0）
        b = pin.rnea(self.robot.model, self.robot.data, self.q, self.qd, np.zeros(n, dtype=float))

        # spring-damper torque: K(θ_cmd - q) - D qd   ※“K(θ_cmd - q)”に統一
        tau_spring = self.params.K * (q_ref - self.q) - self.D * self.qd

        # M qdd = tau_ext + tau_spring - b
        rhs = (tau_ext + tau_spring) - b
        qdd = np.linalg.pinv(M, rcond=1e-12) @ rhs  # or np.linalg.solve

        qd_next = self.qd + dt * qdd
        if self.vel_lim is not None:
            vlim = np.asarray(self.vel_lim, dtype=float)
            qd_next = np.clip(qd_next, -np.abs(vlim), np.abs(vlim))
        q_next = self.q + dt * qd_next

        if self.pos_lo is not None and self.pos_hi is not None:
            q_next = np.minimum(np.maximum(q_next, self.pos_lo), self.pos_hi)

        self.q, self.qd = q_next, qd_next
        return q_next.copy(), qd_next.copy()

    # ---------------------- helpers ----------------------

    def simulate(
        self,
        dt: float,
        q_ref_seq: np.ndarray,
        tau_ext_seq: Optional[np.ndarray] = None,
        q0: Optional[np.ndarray] = None,
        qd0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run a batch simulation over a sequence.

        Args:
            dt: time step
            q_ref_seq: shape (T, n)
            tau_ext_seq: optional, shape (T, n); if None, zeros
            q0, qd0: optional initial state

        Returns:
            (Q, Qd) with shapes (T, n)
        """
        T, n = q_ref_seq.shape
        if q0 is not None or qd0 is not None:
            self.reset(q=q0, qd=qd0)

        if tau_ext_seq is None:
            tau_ext_seq = np.zeros((T, n), dtype=float)

        Q = np.zeros((T, n), dtype=float)
        Qd = np.zeros((T, n), dtype=float)

        for k in range(T):
            self.step(dt=dt, q_ref=q_ref_seq[k], tau_ext=tau_ext_seq[k])
            Q[k] = self.q
            Qd[k] = self.qd

        return Q, Qd
