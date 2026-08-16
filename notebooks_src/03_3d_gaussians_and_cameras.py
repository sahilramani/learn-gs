# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 03 - 3D Gaussians, and the projection problem
#
# Notebook 02 gave us a screen-space ellipse. But nobody authors screen-space
# ellipses; the scene lives in 3D. Same construction, one dimension up:
#
# $$G(x) = \exp\!\left(-\tfrac{1}{2}
# (x - \mu)^{T} \Sigma^{-1} (x - \mu)\right),
# \qquad x, \mu \in \mathbb{R}^3$$
#
# $$\Sigma = R S S^{T} R^{T}, \qquad S = \mathrm{diag}(s_x, s_y, s_z)$$
#
# A fuzzy ellipsoid: position $\mu$, three semi-axis scales, one rotation.
# The scene is a bag of these. The renderer's job is to turn each into a
# notebook-02 ellipse and blend.
#
# ## Storing the rotation
#
# Notebook 02 established that $\Sigma$ must be built from a rotation and
# scales,
# never optimized entrywise. That makes the rotation a stored, optimized
# quantity, and its representation matters:
#
# A 3x3 matrix is 9 numbers carrying 6 constraints (orthonormality); a gradient
# step breaks the constraints and the "rotation" starts shearing. Euler angles
# are 3 numbers but have coordinate singularities (gimbal lock) where gradients
# degenerate. A quaternion is 4 numbers whose only constraint is unit length,
# and normalization projects any 4-vector back onto a valid rotation. So the
# update rule is: take a gradient step in $\mathbb{R}^4$, normalize. That is
# why 3DGS
# stores `(w, x, y, z)` per splat, and we adopt the same, w first.

# %%
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

def quat_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])

def build_cov3d(scale, R):
    M = R @ np.diag(np.asarray(scale, float))
    return M @ M.T

# identity
assert np.allclose(quat_to_R([1, 0, 0, 0]), np.eye(3))
# 90 degrees about z: x axis -> y axis
R90 = quat_to_R([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
assert np.allclose(R90 @ [1, 0, 0], [0, 1, 0], atol=1e-12)
# always a rotation, even from an unnormalized quat
Rq = quat_to_R([0.3, -1.2, 0.4, 2.0])
assert np.allclose(Rq @ Rq.T, np.eye(3)) and np.isclose(np.linalg.det(Rq), 1.0)
print("quaternion checks pass")

# %% [markdown]
# ## The camera, consolidated
#
# Notebook 01's pieces, packaged. Nothing new.

# %%
def look_at(eye, target, up=(0, -1, 0)):
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    f = target - eye; f = f / np.linalg.norm(f)
    r = np.cross(f, up); r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])

@dataclass
class Camera:
    R: np.ndarray
    eye: np.ndarray
    fx: float; fy: float; cx: float; cy: float
    H: int; W: int

    @classmethod
    def looking_at(cls, eye, target, fx, fy, H, W, up=(0, -1, 0)):
        return cls(look_at(eye, target, up), np.asarray(eye, float),
                   fx, fy, W / 2, H / 2, H, W)

    def world_to_cam(self, P):
        return (np.asarray(P, float) - self.eye) @ self.R.T

    def project(self, Pc):
        z = Pc[..., 2]
        u = self.fx * Pc[..., 0] / z + self.cx
        v = self.fy * Pc[..., 1] / z + self.cy
        return u, v, z

cam = Camera.looking_at(eye=[0, 0, -4], target=[0, 0, 0],
                        fx=600, fy=600, H=512, W=512)

# %% [markdown]
# ## Where does a 3D Gaussian go under projection?
#
# The mean is easy: it is a point, project it. The question is the covariance.
# Is the projected splat still a Gaussian, so that notebook 02 applies?
#
# No formula yet. Measure it instead. A Gaussian is exactly the distribution of
# `mu + R S n` with `n ~ N(0, I)`, so: draw thousands of samples from the 3D
# Gaussian, push every sample through the full nonlinear projection, and look
# at the 2D cloud. Whatever shape appears is the ground truth the renderer must
# reproduce.

# %%
def sample_gaussian3d(mu, R, scale, n, rng):
    z = rng.standard_normal((n, 3))
    return np.asarray(mu, float) + (z * np.asarray(scale, float)) @ R.T

rng = np.random.default_rng(1)

# Case A: small splat, far away
muA, sA, RA = [0.4, 0.2, 11.0], [0.5, 0.3, 0.4], quat_to_R([0.9, 0.2, 0.3, 0.1])
# Case B: large splat, close, elongated along the view direction
muB, sB, RB = [0.0, 0.0, -2.0], [0.08, 0.08, 0.55], np.eye(3)

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (mu, s, R, name) in zip(axes, [(muA, sA, RA, "A: small, far"),
                                       (muB, sB, RB, "B: large, close, along view")]):
    X = sample_gaussian3d(mu, R, s, 20_000, rng)
    Xc = cam.world_to_cam(X)
    keep = Xc[:, 2] > 0.05                      # notebook 01's near cull
    u, v, z = cam.project(Xc[keep])
    ax.scatter(u, v, s=1, alpha=0.08, c="k")
    ax.set_xlim(0, 512); ax.set_ylim(512, 0)    # y down, like the image
    ax.set_title(f"{name}  ({keep.sum()} of 20000 in front)")
    ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% [markdown]
# Case A projects to something indistinguishable from an ellipse-shaped cloud.
# Case B does not: the near end of the ellipsoid, where `z` is small, blows up
# under the `1/z`, and the cloud comes out as a funnel with a heavy tail. Not
# an ellipse, so not a 2D Gaussian.
#
# ## Isolating the culprit
#
# The projection has two stages: a rigid transform (world to camera), then the
# perspective divide. Test the rigid stage alone. If `y = A x + b`, then
# `Cov[y] = A Cov[x] A^T`; for the camera transform `A = R_cam`. Check it
# empirically on case B's samples:

# %%
SigmaB = build_cov3d(sB, RB)
X = sample_gaussian3d(muB, RB, sB, 20_000, rng)
Xc = cam.world_to_cam(X)

S_emp = np.cov(Xc.T)
S_pred = cam.R @ SigmaB @ cam.R.T
err = np.abs(S_emp - S_pred).max() / np.abs(S_pred).max()
print(f"max relative error, empirical vs R Sigma R^T: {err:.3%}")
assert err < 0.05
print("rigid motion preserves Gaussians (to sampling noise)")

# %% [markdown]
# So the rigid stage is exact and Gaussian-preserving; the perspective divide
# `u = fx x / z` is the nonlinearity that bends case B. The general fact:
# Gaussians map to Gaussians under affine maps and only under affine maps.
#
# Two roads from here. Render by sampling, which is what we just did: correct,
# noisy, and thousands of evaluations per splat. Or replace the projection with
# an affine map that agrees with it near the mean, where the splat's mass is,
# and accept error in the tails. The second road is one Taylor expansion.
# Notebook 04.
