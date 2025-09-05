from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from scipy.optimize import minimize

@dataclass
class EquilibriumConfig:
    maxiter: int = 200
    k_stiffness: float = 100.0
    n_lambda: int = 10
    ftol: float = 1e-9
    verbose: bool = False

class EquilibriumSolver:
    def __init__(self, cfg: Optional[EquilibriumConfig] = None) -> None:
        self.cfg = cfg or EquilibriumConfig()
        self.eq_path_last: List[np.ndarray] = []

    @staticmethod
    def _pack_cs(c: np.ndarray, s: np.ndarray) -> np.ndarray:
        out = np.empty(c.size * 2, dtype=float)
        out[0::2] = c; out[1::2] = s
        return out

    @staticmethod
    def _unpack_cs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return x[0::2], x[1::2]

    @staticmethod
    def _theta_from_cs(c: np.ndarray, s: np.ndarray) -> np.ndarray:
        return np.arctan2(s, c)

    @staticmethod
    def _V_total(robot, theta: np.ndarray, theta_cmd: np.ndarray, k_eff_diag: np.ndarray) -> float:
        U = robot.potential_gravity(theta)
        d = theta - theta_cmd
        return float(U + 0.5 * np.dot(d * k_eff_diag, d))

    @staticmethod
    def _grad_theta(robot, theta: np.ndarray, theta_cmd: np.ndarray, k_eff_diag: np.ndarray) -> np.ndarray:
        tau_g = robot.tau_gravity(theta)
        return tau_g + k_eff_diag * (theta - theta_cmd)

    @staticmethod
    def _grad_x_from_grad_theta(g_theta: np.ndarray, c: np.ndarray, s: np.ndarray) -> np.ndarray:
        denom = np.maximum(c * c + s * s, 1e-12)
        dtheta_dc = -s / denom
        dtheta_ds =  c / denom
        gx = np.empty(g_theta.size * 2, dtype=float)
        gx[0::2] = g_theta * dtheta_dc; gx[1::2] = g_theta * dtheta_ds
        return gx

    @staticmethod
    def _cons_fun(x: np.ndarray) -> np.ndarray:
        c, s = EquilibriumSolver._unpack_cs(x)
        return c * c + s * s - 1.0

    @staticmethod
    def _cons_jac(x: np.ndarray) -> np.ndarray:
        c, s = EquilibriumSolver._unpack_cs(x)
        n = c.size
        J = np.zeros((n, 2 * n), dtype=float)
        idx = np.arange(n)
        J[idx, 2 * idx] = 2.0 * c
        J[idx, 2 * idx + 1] = 2.0 * s
        return J

    def _stage_minimize(self, robot, theta_cmd: np.ndarray, k_eff_diag: np.ndarray, x0: np.ndarray):
        def f_obj(x: np.ndarray) -> float:
            c, s = self._unpack_cs(x)
            theta = self._theta_from_cs(c, s)
            return self._V_total(robot, theta, theta_cmd, k_eff_diag)

        def f_jac(x: np.ndarray) -> np.ndarray:
            c, s = self._unpack_cs(x)
            theta = self._theta_from_cs(c, s)
            g_theta = self._grad_theta(robot, theta, theta_cmd, k_eff_diag)
            return self._grad_x_from_grad_theta(g_theta, c, s)

        cons = { "type": "eq", "fun": self._cons_fun, "jac": self._cons_jac }

        res = minimize(
            fun=f_obj, x0=x0, jac=f_jac, constraints=[cons], method="SLSQP",
            options={"maxiter": int(self.cfg.maxiter), "ftol": float(self.cfg.ftol), "disp": bool(self.cfg.verbose)},
        )
        x_opt = res.x
        c_opt, s_opt = self._unpack_cs(x_opt)
        theta_opt = self._theta_from_cs(c_opt, s_opt)
        return x_opt, theta_opt

    def solve(self, robot, theta_cmd: np.ndarray, kp_vec: np.ndarray, theta_init: Optional[np.ndarray] = None, lambdas: Optional[np.ndarray] = None) -> np.ndarray:
        if lambdas is None:
            lambdas = np.linspace(1.0, 0.0, self.cfg.n_lambda)

        theta0 = theta_init.copy() if theta_init is not None else theta_cmd.copy()
        c0 = np.cos(theta0); s0 = np.sin(theta0)
        x0 = self._pack_cs(c0, s0)

        self.eq_path_last = []
        for lam in lambdas:
            k_eff_diag = kp_vec + float(lam) * float(self.cfg.k_stiffness)
            x0, theta_opt = self._stage_minimize(robot, theta_cmd, k_eff_diag, x0)
            self.eq_path_last.append(theta_opt.copy())
        return theta_opt
