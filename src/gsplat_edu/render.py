"""Reference CPU renderer. Promoted from notebook 05.

Pipeline: cull -> project -> dilate -> sort by depth -> composite front-to-back.
Slow on purpose: every stage is visible. The CUDA phases reimplement exactly this.
"""
import numpy as np
from .gaussians import project_cov3d

DILATION = 0.3          # screen-space low-pass, notebook 02
ALPHA_MIN = 1.0 / 255   # below this a splat cannot change an 8-bit pixel
ALPHA_MAX = 0.99        # keep (1 - alpha) away from zero
T_STOP = 1e-4           # region is opaque, further splats are dead work


def render_gaussians(cam, means, cov3ds, colors, opacities, bg=(0, 0, 0),
                     z_near=0.05):
    H, W = cam.H, cam.W
    img = np.zeros((H, W, 3))
    Tmap = np.ones((H, W))

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
            continue  # frustum cull: bbox misses the screen
        splats.append((z, u, v, conic, (x0, x1, y0, y1), colors[i], opacities[i]))

    splats.sort(key=lambda s: s[0])  # near to far

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
