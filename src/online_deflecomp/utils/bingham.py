import numpy as np
from .geometry import normalize

class BinghamUtils:
    @staticmethod
    def qmat_from_quat_wxyz(z: np.ndarray) -> np.ndarray:
        w, x, y, zc = z
        return np.array([
            [-x, -y, -zc],
            [w, -zc, y],
            [zc, w, -x],
            [-y, x, w],
        ], dtype=float)

    @staticmethod
    def _lmat(q: np.ndarray) -> np.ndarray:
        a, b, c, d = q
        return np.array([
            [a, -b, -c, -d],
            [b, a, -d, c],
            [c, d, a, -b],
            [d, -c, b, a],
        ], dtype=float)

    @staticmethod
    def _rmat(q: np.ndarray) -> np.ndarray:
        w, x, y, zc = q
        return np.array([
            [w, -x, -y, -zc],
            [x, w, zc, -y],
            [y, -zc, w, x],
            [zc, y, -x, w],
        ], dtype=float)

    @staticmethod
    def simple_bingham_unit(before_vec3: np.ndarray, after_vec3: np.ndarray, parameter: float = 100.0) -> np.ndarray:
        b = normalize(np.asarray(before_vec3, dtype=float))
        a = normalize(np.asarray(after_vec3, dtype=float))
        vq = np.array([0.0, b[0], b[1], b[2]], dtype=float)
        xq = np.array([0.0, a[0], a[1], a[2]], dtype=float)
        P = BinghamUtils._lmat(xq) - BinghamUtils._rmat(vq)
        A0 = -0.25 * (P.T @ P)
        return float(parameter) * A0


class NoiseCovarianceHelper:
    @classmethod
    def calc_covQupdate(cls, covA, covB, covQ=None):
        if covQ is None:
            covQ = 0.25 * np.eye(4)
        covAB = np.zeros((6,6))
        covAB[:3,:3] = covA
        covAB[3:,3:] = covB
        N = cls.get_Nmat()
        return N @ np.kron(covAB, covQ) @ N.T

    @staticmethod
    def get_Nmat():
        ## initialize
        N = np.zeros((4, 24))
        ## set non-zero elements
        N[0, 1] = -1.
        N[0, 6] = -1.
        N[0, 11] = -1.
        N[0, 13] = 1.
        N[0, 18] = 1.
        N[0, 23] = 1.
        N[1, 0] = 1.
        N[1, 7] = -1.
        N[1, 10] = 1.
        N[1, 12] = -1.
        N[1, 19] = -1.
        N[1, 22] = 1.
        N[2, 3] = 1.
        N[2, 4] = 1.
        N[2, 9] = -1.
        N[2, 15] = 1.
        N[2, 16] = -1.
        N[2, 21] = -1.
        N[3, 2] = -1.
        N[3, 5] = 1.
        N[3, 8] = 1.
        N[3, 14] = -1.
        N[3, 17] = 1.
        N[3, 20] = -1.
        return N

    @staticmethod
    def get_Hmat(vec3D_bef_rot, vec3D_aft_rot):
        ## alias
        v0 = vec3D_aft_rot
        v1 = vec3D_bef_rot

        ## initialize
        H = np.zeros((4,4))

        ## pseudo-measurement matrix
        # H[0,0] = 0
        H[0,1] = -v0[0] + v1[0]
        H[0,2] = -v0[1] + v1[1]
        H[0,3] = -v0[2] + v1[2]
        H[1,0] = v0[0] - v1[0]
        # H[1,1] = 0
        H[1,2] = -v0[2] - v1[2]
        H[1,3] = v0[1] + v1[1]
        H[2,0] = v0[1] - v1[1]
        H[2,1] = v0[2] + v1[2]
        # H[2,2] = 0
        H[2,3] = -v0[0] - v1[0]
        H[3,0] = v0[2] - v1[2]
        H[3,1] = -v0[1] - v1[1]
        H[3,2] = v0[0] + v1[0]
        # H[3,3] = 0

        return H

    @classmethod
    def calc_bingham_A(cls, avec, bvec, cov_A, cov_B):
        H = cls.get_Hmat(bvec, avec)
        S = cls.calc_covQupdate(cov_A, cov_B)
        Sinv = np.linalg.pinv(S)
        return -0.5 * H.T @ Sinv @ H