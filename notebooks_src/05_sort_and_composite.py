# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05 - Sort, composite, and the first real renderer
#
# We can turn one 3D Gaussian into a screen ellipse. A scene is thousands, and
# several of them land on the same pixel. The missing piece is a combination
# rule.
#
# ## The over operator
#
# Treat each splat's value at a pixel as an opacity:
#
# $$\alpha_i = o_i \, G_i(\text{pixel})$$
#
# Walk the splats covering the pixel from near to far. Track transmittance
# $T$, the fraction of the pixel's light budget still unclaimed. Start at
# $T = 1$. Splat $i$ claims $\alpha_i$ of what remains:
#
# $$C \mathrel{+}= c_i \alpha_i T, \qquad T \mathrel{*}= (1 - \alpha_i)$$
#
# Unrolled, that is
#
# $$C = \sum_i c_i \alpha_i \prod_{j<i} (1 - \alpha_j)
# \; + \; T_\text{final} \cdot \text{background}$$
#
# This is alpha compositing's "over" operator, the same rule Porter and Duff
# wrote down in 1984, and the entire volume-rendering step of 3DGS.
#
# Two clamps ride along, both from notebook 02's world of 8-bit budgets plus
# one new concern:
#
# $$\alpha < \tfrac{1}{255} \rightarrow \text{skip}
# \quad\text{(cannot change the pixel)}$$
#
# $$\alpha = \min(\alpha, 0.99)
# \quad\text{(keep } 1 - \alpha \text{ away from zero)}$$
#
# The 0.99 cap matters later: training divides by $(1 - \alpha)$ in the
# backward pass, and a splat that reaches exact opacity would zero `T` and
# erase every gradient behind it. Cheap insurance, installed now.
#
# ## The order
#
# The product over $j < i$ only means something if "before" is defined per
# pixel. Exact per-pixel depth ordering is expensive, so 3DGS approximates:
# sort splats once per frame by the camera-space depth of their means, and use
# that single global order for every pixel. The approximation is wrong whenever
# splats interpenetrate: two splats can swap global order as the camera moves,
# and the blend visibly pops. Known artifact, accepted cost. We inherit it.

# %%
import time
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

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

def project_cov3d(cov3d, mean_cam, cam):
    x, y, z = mean_cam
    J = np.array([
        [cam.fx / z, 0.0,        -cam.fx * x / z ** 2],
        [0.0,        cam.fy / z, -cam.fy * y / z ** 2],
    ])
    T = J @ cam.R
    return T @ cov3d @ T.T

DILATION = 0.3          # screen-space low-pass, notebook 02
ALPHA_MIN = 1.0 / 255   # below this a splat cannot change an 8-bit pixel
ALPHA_MAX = 0.99        # keep (1 - alpha) away from zero
T_STOP = 1e-4           # region is opaque, further splats are dead work

# %% [markdown]
# ## A scene worth rendering
#
# Callback to notebook 01: the sphere that fell apart as confetti. Rebuild it
# from Gaussians, and rebuild it the honest way: not fuzzy balls, but thin
# disks lying tangent on the surface, the shape 3DGS actually converges to on
# real surfaces. Each disk's covariance comes straight from the notebook-02
# construction, with the basis `(t1, t2, normal)` playing the role of R:
# two tangential scales `s_t`, one small radial scale `s_n`.

# %%
def sphere_shell(n=2500, s_t=0.06, s_n=0.01, opacity=0.8, seed=0):
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

means, cov3ds, colors, opacities = sphere_shell()
print(means.shape, cov3ds.shape)

# %% [markdown]
# ## Stage 1: cull and project
#
# Everything below is assembled from earlier notebooks. Near cull (01),
# projection of mean and covariance (04), dilation, conic, radius, bbox (02),
# plus a frustum cull that is just "the bbox missed the screen".

# %%
cam = Camera.looking_at(eye=[0, 0, -3], target=[0, 0, 0],
                        fx=600, fy=600, H=512, W=512)

def project_stage(cam, means, cov3ds, colors, opacities, z_near=0.05):
    H, W = cam.H, cam.W
    Pc = cam.world_to_cam(means)
    z_all = Pc[:, 2]
    splats = []
    for i in np.where(z_all > z_near)[0]:
        x, y, z = Pc[i]
        u = cam.fx * x / z + cam.cx
        v = cam.fy * y / z + cam.cy
        c2 = project_cov3d(cov3ds[i], Pc[i], cam)
        c2[0, 0] += DILATION
        c2[1, 1] += DILATION
        a, b, c = c2[0, 0], c2[0, 1], c2[1, 1]
        det = a * c - b * b
        if det <= 0:
            continue
        conic = (c / det, -b / det, a / det)
        mid = 0.5 * (a + c)
        lam_max = mid + np.sqrt(max(0.1, mid * mid - det))
        r = int(np.ceil(3.0 * np.sqrt(lam_max)))
        x0, x1 = max(int(np.floor(u)) - r, 0), min(int(np.ceil(u)) + r + 1, W)
        y0, y1 = max(int(np.floor(v)) - r, 0), min(int(np.ceil(v)) + r + 1, H)
        if x0 >= x1 or y0 >= y1:
            continue
        splats.append((z, u, v, conic, (x0, x1, y0, y1), colors[i], opacities[i]))
    return splats

splats = project_stage(cam, means, cov3ds, colors, opacities)
pairs = sum((x1 - x0) * (y1 - y0) for _, _, _, _, (x0, x1, y0, y1), _, _ in splats)
print(f"splats total {len(means)}, on screen {len(splats)}")
print(f"pixel-splat pairs: {pairs/1e6:.1f}M  ({pairs / (512*512):.0f} per pixel avg)")

# %% [markdown]
# ## Stage 2: sort and composite
#
# One global sort by mean depth, then front-to-back over. Compositing per
# splat over its bbox is exact: each pixel still sees its covering splats in
# the one global order, so the per-pixel unrolled sum comes out right.
# The `T_STOP` check skips splats whose entire bbox is already opaque; that is
# per-region early termination, and its per-tile version is the heart of the
# fast CUDA path later.

# %%
def composite_stage(cam, splats, bg=(0, 0, 0)):
    H, W = cam.H, cam.W
    img = np.zeros((H, W, 3))
    Tmap = np.ones((H, W))
    splats = sorted(splats, key=lambda s: s[0])         # near to far
    for _, u, v, conic, (x0, x1, y0, y1), col, o in splats:
        Treg = Tmap[y0:y1, x0:x1]
        if Treg.max() < T_STOP:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx, dy = xs - u, ys - v
        power = -0.5 * (conic[0] * dx * dx + conic[2] * dy * dy) - conic[1] * dx * dy
        alpha = np.minimum(ALPHA_MAX, o * np.exp(power))
        alpha[alpha < ALPHA_MIN] = 0.0
        img[y0:y1, x0:x1] += (Treg * alpha)[..., None] * np.asarray(col, float)
        Tmap[y0:y1, x0:x1] = Treg * (1.0 - alpha)
    img += Tmap[..., None] * np.asarray(bg, float)
    return img, Tmap

t0 = time.perf_counter()
img, Tmap = composite_stage(cam, splats)
dt = time.perf_counter() - t0

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(np.clip(img, 0, 1)); axes[0].set_title("color")
axes[1].imshow(Tmap, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("final transmittance T")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()
print(f"composite: {dt:.2f} s  ({pairs / dt / 1e6:.1f}M pairs/s in numpy)")

# %% [markdown]
# A solid sphere. The transmittance map tells the same story from the other
# side: T ~ 0 across the body (opaque, splats behind were skipped), T = 1
# outside (background shows), a soft ring at the silhouette where coverage
# tapers instead of stepping. That soft ring is notebook 02's whole argument.
#
# ## The moment of truth: walk toward it again
#
# Same primitives, same count, both renderers. Points on top, splats below.

# %%
def render_points(P, C, eye, H=512, W=512, fx=600.0, fy=600.0):
    cx, cy = W / 2, H / 2
    R = look_at(eye, [0, 0, 0])
    Xc = (P - np.asarray(eye, float)) @ R.T
    front = Xc[:, 2] > 1e-4
    Xc, col = Xc[front], C[front]
    z = Xc[:, 2]
    u = np.round(fx * Xc[:, 0] / z + cx).astype(int)
    v = np.round(fy * Xc[:, 1] / z + cy).astype(int)
    vis = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, col, z = u[vis], v[vis], col[vis], z[vis]
    order = np.argsort(-z)
    img = np.zeros((H, W, 3))
    img[v[order], u[order]] = col[order]
    return img

dists = [4.0, 2.2, 1.4]
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for j, d in enumerate(dists):
    axes[0, j].imshow(render_points(means, colors, eye=[0, 0, -d]))
    axes[0, j].set_title(f"points, d={d}")
    c = Camera.looking_at(eye=[0, 0, -d], target=[0, 0, 0],
                          fx=600, fy=600, H=512, W=512)
    im, _ = composite_stage(c, project_stage(c, means, cov3ds, colors, opacities))
    axes[1, j].imshow(np.clip(im, 0, 1))
    axes[1, j].set_title(f"splats, d={d}")
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# 2500 points cannot hold a surface together at any distance. The same 2500
# Gaussians are a solid object at all three, because their footprints scale
# with `fx s / z` instead of staying one pixel. Notebook 01's failure is
# closed.
#
# ## Promotion
#
# This pipeline is stable enough to live in the package. The two stages,
# consolidated into one function, are `gsplat_edu.render_gaussians`; the scene
# builder is `gsplat_edu.sphere_shell`. Same code, importable. Prove it:

# %%
import gsplat_edu

pkg_img, pkg_T = gsplat_edu.render_gaussians(cam, means, cov3ds, colors, opacities)
assert np.allclose(pkg_img, img, atol=1e-8)
assert np.allclose(pkg_T, Tmap, atol=1e-8)
print("package renderer matches the notebook, pixel for pixel")

# %% [markdown]
# From here on, notebooks import the renderer instead of rebuilding it, and
# `scripts/render_toy.py` renders this scene from the command line.
#
# ## Where the time went
#
# The stats above are the map for everything that follows. A few million
# pixel-splat pairs took seconds in numpy; a real scene is millions of splats
# at 60 frames per second, roughly five orders of magnitude away. Nothing
# about the math changes from here. The rest of the project is about who
# evaluates it (CUDA threads), in what grouping (16x16 tiles with per-tile
# splat lists), and how the gradients flow back through this exact pipeline
# (training).
#
# Two problems are now live, and either one forces the next notebook:
# our splats have one RGB from every direction, which real captured materials
# do not (notebook 06, spherical harmonics), and nothing yet chooses the
# Gaussians; we placed them by hand (phase C, training).
