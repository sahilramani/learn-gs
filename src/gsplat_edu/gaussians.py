"""Gaussian parameterization and the 3D -> 2D covariance projection."""
import numpy as np


def quat_to_R(q):
    """Unit quaternion (w, x, y, z) -> rotation matrix. Normalizes first,
    so any 4-vector maps to a valid rotation (this is why training stores quats)."""
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def build_cov3d(scale, R):
    """Sigma = (R S)(R S)^T. Positive semi-definite by construction."""
    M = R @ np.diag(np.asarray(scale, float))
    return M @ M.T


def project_cov3d(cov3d, mean_cam, cam):
    """World-frame 3x3 covariance -> screen-space 2x2, linearized at the mean.

    Sigma2d = J W Sigma W^T J^T.  W = camera rotation (affine, exact).
    J = Jacobian of (u, v) at the mean (perspective, first-order).
    J is 2x3: dropping the depth row is the marginalization along the view ray.
    """
    x, y, z = mean_cam
    J = np.array([
        [cam.fx / z, 0.0,        -cam.fx * x / z ** 2],
        [0.0,        cam.fy / z, -cam.fy * y / z ** 2],
    ])
    T = J @ cam.R
    return T @ cov3d @ T.T
