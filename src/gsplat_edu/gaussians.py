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


def quat_to_R_batch(q):
    """quat_to_R over rows. q (N, 4) wxyz -> (N, 3, 3). Promoted from
    notebook 07; loading a real scene converts every splat at once."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - w * z)
    R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y)
    R[..., 2, 1] = 2 * (y * z + w * x)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_cov3d_batch(scales, Rs):
    """build_cov3d over rows. scales (N, 3), Rs (N, 3, 3) -> (N, 3, 3)."""
    M = np.asarray(Rs, float) * np.asarray(scales, float)[..., None, :]
    return np.einsum("...ij,...kj->...ik", M, M)


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
