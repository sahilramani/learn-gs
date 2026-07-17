"""Train 3D Gaussians against renders of the toy sphere. Notebook 09 is
the derivation and the checks; this is the same pipeline as a script.

Usage: python train/train_3d.py [--iters 800] [--k0 400] [--cap 1600]
       [--size 128] [--out out/fit3d.png]
"""
import argparse
import pathlib
import time
import numpy as np

try:
    import torch
except ImportError:
    raise SystemExit("train_3d.py needs torch: pip install -r requirements.txt")

from gsplat_edu import Camera, render_gaussians, sphere_shell

DILATION = 0.3
ALPHA_MIN = 1.0 / 255
ALPHA_MAX = 0.99
Z_NEAR = 0.05
C0 = 0.28209479177387814
DTYPE = torch.float32


def quat_to_R_t(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                     2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                     2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x),
                     1 - 2 * (x * x + y * y)], -1)], -2)


def cov3d_t(log_s, quats):
    M = quat_to_R_t(quats) * torch.exp(log_s)[..., None, :]
    return M @ M.mT


def render_t(means, cov3ds, colors, opacities, cam, alpha_min=ALPHA_MIN,
             chunk=512, track_means2d=False):
    dt = means.dtype
    R = torch.tensor(cam.R, dtype=dt)
    eye = torch.tensor(cam.eye, dtype=dt)
    Pc = (means - eye) @ R.T
    vis = Pc[:, 2] > Z_NEAR
    vis_idx = torch.where(vis)[0]
    Pc, cov3ds = Pc[vis], cov3ds[vis]
    colors, opacities = colors[vis], opacities[vis]
    order = torch.argsort(Pc[:, 2])
    Pc, cov3ds = Pc[order], cov3ds[order]
    colors, opacities = colors[order], opacities[order]
    vis_idx = vis_idx[order]

    x, y, z = Pc.unbind(-1)
    u = cam.fx * x / z + cam.cx
    v = cam.fy * y / z + cam.cy
    means2d = torch.stack([u, v], -1)
    if track_means2d:
        means2d.retain_grad()
    u, v = means2d.unbind(-1)

    zero = torch.zeros_like(z)
    J = torch.stack([
        torch.stack([cam.fx / z, zero, -cam.fx * x / z ** 2], -1),
        torch.stack([zero, cam.fy / z, -cam.fy * y / z ** 2], -1)], -2)
    T_JW = J @ R
    S2 = T_JW @ cov3ds @ T_JW.mT
    a = S2[:, 0, 0] + DILATION
    c = S2[:, 1, 1] + DILATION
    b = S2[:, 0, 1]
    det = a * c - b * b
    A00, A01, A11 = c / det, -b / det, a / det

    ys = torch.arange(cam.H, dtype=dt)
    xs = torch.arange(cam.W, dtype=dt)
    img = torch.zeros(cam.H, cam.W, 3, dtype=dt)
    T_run = torch.ones(cam.H, cam.W, dtype=dt)
    for k0 in range(0, len(u), chunk):
        k1 = min(k0 + chunk, len(u))
        dx = xs[None, None, :] - u[k0:k1, None, None]
        dy = ys[None, :, None] - v[k0:k1, None, None]
        power = (-0.5 * (A00[k0:k1, None, None] * dx * dx
                         + A11[k0:k1, None, None] * dy * dy)
                 - A01[k0:k1, None, None] * dx * dy)
        alpha = opacities[k0:k1, None, None] * torch.exp(power)
        alpha = torch.clamp(alpha, max=ALPHA_MAX)
        alpha = torch.where(alpha < alpha_min, torch.zeros_like(alpha), alpha)
        Tc = torch.cumprod(1.0 - alpha, dim=0)
        T_before = torch.cat([T_run[None], T_run * Tc[:-1]], dim=0)
        img = img + torch.einsum("khw,kc->hwc", T_before * alpha,
                                 colors[k0:k1])
        T_run = T_run * Tc[-1]
    return img, T_run, means2d, vis_idx


class AdamDict:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, b1, b2, eps, 0
        self.m = {k: torch.zeros_like(v) for k, v in params.items()}
        self.v = {k: torch.zeros_like(v) for k, v in params.items()}

    @torch.no_grad()
    def step(self, params):
        self.t += 1
        for k, p in params.items():
            if p.grad is None:
                continue
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * p.grad
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * p.grad ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            p -= self.lr[k] * mhat / (torch.sqrt(vhat) + self.eps)
            p.grad = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--k0", type=int, default=400)
    ap.add_argument("--cap", type=int, default=1600)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--out", default="out/fit3d.png")
    ap.add_argument("--seed", type=int, default=9)
    args = ap.parse_args()

    size, fx = args.size, 600.0 * args.size / 512
    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    def orbit_camera(az_deg, el, r=3.0):
        a = np.radians(az_deg)
        eye = r * np.array([np.cos(el) * np.sin(a), np.sin(el),
                            -np.cos(el) * np.cos(a)])
        return Camera.looking_at(eye=eye, target=[0, 0, 0],
                                 fx=fx, fy=fx, H=size, W=size)

    gt_means, gt_covs, gt_colors, gt_opac = sphere_shell()
    cams = [orbit_camera(az, el)
            for el in (-0.35, 0.0, 0.35) for az in range(0, 360, 45)]
    held_out = orbit_camera(202.5, 0.18)

    def render_np(cam):
        img, _ = render_gaussians(cam, gt_means, gt_covs, gt_colors, gt_opac)
        return np.clip(img, 0, 1)

    gts = [torch.tensor(render_np(c), dtype=DTYPE) for c in cams]
    gt_held = torch.tensor(render_np(held_out), dtype=DTYPE)

    d = torch.randn(args.k0, 3, generator=gen, dtype=DTYPE)
    d = d / d.norm(dim=-1, keepdim=True)
    r = torch.rand(args.k0, 1, generator=gen, dtype=DTYPE) ** (1 / 3)
    params = {
        "means": (d * r).requires_grad_(),
        "log_s": torch.full((args.k0, 3), np.log(0.09),
                            dtype=DTYPE).requires_grad_(),
        "quats": (torch.tensor([[1.0, 0, 0, 0]] * args.k0, dtype=DTYPE)
                  + 0.01 * torch.randn(args.k0, 4, generator=gen, dtype=DTYPE)
                  ).requires_grad_(),
        "sh0": (0.2 * torch.randn(args.k0, 3, generator=gen, dtype=DTYPE)
                ).requires_grad_(),
        "o_logit": torch.full((args.k0,), -2.2, dtype=DTYPE).requires_grad_(),
    }
    lr = {"means": 1e-3, "log_s": 5e-3, "quats": 1e-3,
          "sh0": 2.5e-3, "o_logit": 5e-2}
    opt = AdamDict(params, lr)

    def render_current(cam, track=False):
        colors = torch.clamp(C0 * params["sh0"] + 0.5, min=0.0)
        return render_t(params["means"],
                        cov3d_t(params["log_s"], params["quats"]),
                        colors, torch.sigmoid(params["o_logit"]), cam,
                        track_means2d=track)

    def densify_and_prune(accum, count):
        with torch.no_grad():
            opac = torch.sigmoid(params["o_logit"])
            avg = (accum / count.clamp(min=1)) * (size / 2)
            keep = opac >= 0.005
            big = torch.exp(params["log_s"]).max(dim=-1).values > 0.01 * 3.0
            hot = avg > 2e-4
            clone = keep & hot & ~big
            split = keep & hot & big
            keep_after = keep & ~split
            pieces = {k: [v[keep_after]] for k, v in params.items()}
            ms = {k: [opt.m[k][keep_after]] for k in params}
            vs = {k: [opt.v[k][keep_after]] for k in params}

            def add(idx, jitter, shrink):
                if int(idx.sum()) == 0:
                    return
                for k, v in params.items():
                    new = v[idx].clone()
                    if k == "means" and jitter:
                        R = quat_to_R_t(params["quats"][idx])
                        s = torch.exp(params["log_s"][idx])
                        eps = torch.randn(len(new), 3, generator=gen,
                                          dtype=DTYPE)
                        new = new + (R @ (s * eps)[..., None])[..., 0]
                    if k == "log_s" and shrink:
                        new = new - np.log(1.6)
                    pieces[k].append(new)
                    ms[k].append(torch.zeros_like(new))
                    vs[k].append(torch.zeros_like(new))

            add(clone, False, False)
            add(split, True, True)
            add(split, True, True)
            for k in params:
                params[k] = torch.cat(pieces[k])[:args.cap].clone(
                    ).requires_grad_()
                opt.m[k] = torch.cat(ms[k])[:args.cap]
                opt.v[k] = torch.cat(vs[k])[:args.cap]

    accum = torch.zeros(args.k0, dtype=DTYPE)
    count = torch.zeros(args.k0, dtype=DTYPE)
    order = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    for it in range(args.iters):
        cam_i = int(order.integers(len(cams)))
        img, _, means2d, vis_idx = render_current(cams[cam_i], track=True)
        loss = (img - gts[cam_i]).abs().mean()
        loss.backward()
        with torch.no_grad():
            if means2d.grad is not None:
                accum[vis_idx] += means2d.grad.norm(dim=-1)
                count[vis_idx] += 1
        opt.step(params)
        if 100 <= it < int(0.75 * args.iters) and it % 100 == 0 and it > 0:
            densify_and_prune(accum, count)
            accum = torch.zeros(len(params["means"]), dtype=DTYPE)
            count = torch.zeros_like(accum)
        if it % 100 == 0 or it == args.iters - 1:
            print(f"iter {it:4d}  L1 {float(loss):.4f}  "
                  f"K {len(params['means'])}", flush=True)
    wall = time.perf_counter() - t0
    print(f"{args.iters} iters in {wall:.0f} s "
          f"({wall / args.iters * 1e3:.0f} ms/iter)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with torch.no_grad():
        final = render_current(held_out)[0]
    l1 = float((final - gt_held).abs().mean())
    print(f"held-out L1 {l1:.4f}")
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(np.clip(final.numpy(), 0, 1))
    axes[0].set_title("trained, held-out view")
    axes[1].imshow(gt_held.numpy()); axes[1].set_title("ground truth")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
