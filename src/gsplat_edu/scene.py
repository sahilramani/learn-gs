"""Toy scenes for testing the renderer."""
import numpy as np


def sphere_shell(n=2500, s_t=0.06, s_n=0.01, opacity=0.8, seed=0):
    """Unit sphere covered with n thin disks tangent to the surface.

    Each disk: two tangential scales s_t, one radial scale s_n.
    Covariance built directly as B diag(s^2) B^T with B = (t1, t2, normal).
    """
    rng = np.random.default_rng(seed)
    th = rng.uniform(0, 2 * np.pi, n)
    ph = np.arccos(rng.uniform(-1, 1, n))
    P = np.stack([np.sin(ph) * np.cos(th),
                  np.sin(ph) * np.sin(th),
                  np.cos(ph)], axis=-1)
    covs = np.empty((n, 3, 3))
    for i, nrm in enumerate(P):
        a = np.eye(3)[np.argmin(np.abs(nrm))]
        t1 = np.cross(nrm, a)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(nrm, t1)
        B = np.stack([t1, t2, nrm], axis=-1)
        covs[i] = B @ np.diag([s_t ** 2, s_t ** 2, s_n ** 2]) @ B.T
    colors = (P + 1) / 2
    return P, covs, colors, np.full(n, opacity)
