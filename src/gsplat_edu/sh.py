"""Spherical harmonics color. Promoted from notebook 06.

Degrees 0-3, constants and sign convention of the official 3DGS CUDA
(computeColorFromSH). Trained plys decode correctly only under exactly
these signs. Coefficient layout is (N, 16, 3); evaluation adds the 0.5
offset and clips at zero, so an all-zero coefficient vector is middle gray.
"""
import numpy as np

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435]


def sh_basis(dirs):
    """Basis values at unit directions. dirs (..., 3) -> (..., 16)."""
    dirs = np.asarray(dirs, float)
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z
    b = np.empty(dirs.shape[:-1] + (16,))
    b[..., 0] = C0
    b[..., 1] = -C1 * y
    b[..., 2] = C1 * z
    b[..., 3] = -C1 * x
    b[..., 4] = C2[0] * xy
    b[..., 5] = C2[1] * yz
    b[..., 6] = C2[2] * (2 * zz - xx - yy)
    b[..., 7] = C2[3] * xz
    b[..., 8] = C2[4] * (xx - yy)
    b[..., 9] = C3[0] * y * (3 * xx - yy)
    b[..., 10] = C3[1] * xy * z
    b[..., 11] = C3[2] * y * (4 * zz - xx - yy)
    b[..., 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    b[..., 13] = C3[4] * x * (4 * zz - xx - yy)
    b[..., 14] = C3[5] * z * (xx - yy)
    b[..., 15] = C3[6] * x * (xx - 3 * yy)
    return b


def eval_sh(deg, sh, dirs):
    """Decode SH coefficients to RGB at unit view directions.

    sh (N, K, 3) with K >= (deg+1)^2, dirs (N, 3). Returns (N, 3),
    clip(sum_i b_i(d) sh_i + 0.5, 0, None).
    """
    k = (deg + 1) ** 2
    sh = np.asarray(sh, float)
    assert sh.shape[-2] >= k, f"need {k} coefficients for degree {deg}"
    b = sh_basis(dirs)[..., :k]
    c = np.einsum("...k,...kc->...c", b, sh[..., :k, :])
    return np.clip(c + 0.5, 0.0, None)
