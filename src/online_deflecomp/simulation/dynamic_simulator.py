# ros/lib/dynamic_simulator.py
from dataclasses import dataclass
from typing import Optional, Tuple, Literal  # ← Literal を追加
import numpy as np
import pinocchio as pin
from online_deflecomp.controller.equilibrium import EquilibriumSolver, EquilibriumConfig

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
    integrator: str = "rk4"
    ref_tau: float = 1e-9
    ref_max_vel: float = 1000.0
    eq_mode: str = "dynamic"
    tau_eq: Optional[float] = None
    # quasi-static perturbation parameters
    qs_noise_std_deg: float = 0.0
    qs_vib_amp_deg: float = 0.0
    qs_vib_freq_hz: float = 50.0
    qs_vib_axes: Optional[np.ndarray] = None
    qs_seed: Optional[int] = None
    limit_position_high: Optional[np.ndarray] = None
    # new:
    integrator: str = "rk4"              # "rk4" or "semi_implicit_euler"
    ref_tau: Optional[float] = 1e-4      # [s] 1st-order low-pass for q_ref (None/<=0 to disable)
    ref_max_vel: Optional[float] = 10.0  # [rad/s] slew limit for q_ref (None to disable)

    # ---- ここから追記 ----
    # 平衡点直行きモード：
    #   "dynamic"      … 従来どおりの慣性ありダイナミクス
    #   "relax_to_eq"  … 毎ステップ平衡点 q_eq を解き、一次遅れで q に反映（時定数 tau_eq）
    #   "quasistatic"  … 平衡点 q_eq に即ジャンプ（瞬時収束）
    eq_mode: Literal["dynamic", "relax_to_eq", "quasistatic"] = "dynamic"
    tau_eq: Optional[float] = 0.05       # [s] relax_to_eq の時定数（None/<=0 は 0.05 と同等）
    # ---- 追記ここまで ----


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
        self.q_ref_filt = self.q.copy()
        self.q_ref_prev = self.q.copy()

        # ---- ここから追記：任意の平衡点ソルバを後付け可能に ----
        self.eq_solver = EquilibriumSolver(EquilibriumConfig(maxiter=80))  # 任意: 外側から set_eq_solver() で注入
        # ---- 追記ここまで ----

    # ---- ここから追記：平衡点ソルバの setter とフォールバック実装 ----
    def set_eq_solver(self, solver) -> None:
        """Inject external equilibrium solver object with .solve(theta_cmd, kp_vec, theta_init)."""
        self.eq_solver = solver

    def _solve_equilibrium(self, theta_cmd: np.ndarray, kp_vec: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """Return q_eq for given theta_cmd, using external solver if provided."""
        # if self.eq_solver is not None and hasattr(self.eq_solver, "solve"):
        return self.eq_solver.solve(self.robot, theta_cmd=theta_cmd, kp_vec=kp_vec, theta_init=q_init)
        # fallback: RobotArm の既存 S^1-Newton
        # return self.robot.equilibrium_s1(theta_cmd, kp_vec, maxiter=80, theta_init=q_init)
    # ---- 追記ここまで ----


    def reset(self, q: Optional[np.ndarray] = None, qd: Optional[np.ndarray] = None) -> None:
        # time state for quasi-static vibration/noise
        if getattr(self, "_rng", None) is None:
            seed = None if self.params.qs_seed is None else int(self.params.qs_seed)
            try:
                self._rng = np.random.default_rng(seed)
            except Exception:
                self._rng = np.random.default_rng()
        self._t = 0.0
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
        # ---- ここから追記：平衡点直行きモード ----
        mode = getattr(self.params, "eq_mode", "dynamic")
        if mode in ("quasistatic", "relax_to_eq"):
            kp_vec = self.params.K
            q_init = self.q  # warm start
            q_eq = self._solve_equilibrium(theta_cmd=q_ref, kp_vec=kp_vec, q_init=q_init)

            if mode == "quasistatic":
                # quasi-static: hold at equilibrium with small noise + vibration
                self._t = float(self._t + dt)
                n = q_eq.shape[0]
                # noise
                sig = float(max(0.0, self.params.qs_noise_std_deg))
                noise = np.zeros(n, dtype=float)
                if sig > 0.0:
                    noise = self._rng.standard_normal(n) * (np.deg2rad(sig))
                # vibration
                amp = float(max(0.0, self.params.qs_vib_amp_deg))
                vib = np.zeros(n, dtype=float)
                if amp > 0.0:
                    freq = float(max(0.0, self.params.qs_vib_freq_hz))
                    phase = 0.0
                    s = np.sin(2.0*np.pi*freq*self._t + phase)
                    vib_vec = np.ones(n, dtype=float) * np.deg2rad(amp) * s
                    axes = self.params.qs_vib_axes
                    if axes is not None:
                        try:
                            mask = np.zeros(n, dtype=float)
                            idxs = [int(i) for i in np.array(axes).ravel().tolist()]
                            idxs = [i for i in idxs if 0 <= i < n]
                            for i in idxs:
                                mask[i] = 1.0
                            vib_vec = vib_vec * mask
                        except Exception:
                            pass
                    vib = vib_vec
                q_next = q_eq + noise + vib
                qd_next = np.zeros_like(q_eq)
            else:
                tau = float(self.params.tau_eq if (self.params.tau_eq is not None and self.params.tau_eq > 0.0) else 0.05)
                alpha = 1.0 - float(np.exp(-dt / max(tau, 1e-6)))
                q_next = self.q + alpha * (q_eq - self.q)
                qd_next = (q_next - self.q) / max(dt, 1e-9)
                if self.vel_lim is not None:
                    vlim = np.asarray(self.vel_lim, dtype=float)
                    qd_next = np.clip(qd_next, -np.abs(vlim), np.abs(vlim))
                    q_next = self.q + qd_next * dt

            if self.pos_lo is not None and self.pos_hi is not None:
                q_next = np.minimum(np.maximum(q_next, self.pos_lo), self.pos_hi)

            self.q, self.qd = q_next, qd_next
            return q_next.copy(), qd_next.copy()
        # ---- 追記ここまで ----

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