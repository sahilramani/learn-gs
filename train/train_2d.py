"""Fit 2D Gaussians to the notebook-08 procedural target. Notebook 08 is
the derivation; this is the same pipeline as a script.

Usage: python train/train_2d.py [--k 300] [--steps 600] [--size 96]
       [--out out/fit2d.png]
"""
import argparse
import pathlib
import numpy as np

ALPHA_MIN = 1.0 / 255
ALPHA_MAX = 0.99


def make_target(size):
    yy, xx = np.mgrid[0:size, 0:size]
    u = (xx - size / 2) / (size / 2)
    v = (yy - size / 2) / (size / 2)
    r = np.sqrt((u + 0.25) ** 2 + (v + 0.15) ** 2)
    t = np.stack([0.5 + 0.5 * np.cos(9 * r - p) for p in (0.0, 2.1, 4.2)],
                 axis=-1)
    return np.clip(0.65 * t + 0.35 * ((u + 1)[..., None] / 2), 0, 1)


def activate(params):
    return (params["mu"], np.exp(params["log_s"]), params["theta"],
            params["color"], 1.0 / (1.0 + np.exp(-params["o_logit"])))


def conic_of(s, theta):
    c, n = np.cos(theta), np.sin(theta)
    R = np.stack([np.stack([c, -n], -1), np.stack([n, c], -1)], -2)
    Sig = R @ (s[..., None] ** 2 * np.eye(2)) @ np.swapaxes(R, -1, -2)
    det = Sig[..., 0, 0] * Sig[..., 1, 1] - Sig[..., 0, 1] * Sig[..., 1, 0]
    inv = np.empty_like(Sig)
    inv[..., 0, 0] = Sig[..., 1, 1]
    inv[..., 1, 1] = Sig[..., 0, 0]
    inv[..., 0, 1] = inv[..., 1, 0] = -Sig[..., 0, 1]
    return inv / det[..., None, None]


def forward(params, H, W):
    mu, s, theta, color, opa = activate(params)
    A = conic_of(s, theta)
    K = len(mu)
    ys, xs = np.mgrid[0:H, 0:W]
    img = np.zeros((H, W, 3))
    T = np.ones((H, W))
    alphas = np.zeros((K, H, W))
    live = np.zeros((K, H, W), bool)
    for k in range(K):
        dx, dy = xs - mu[k, 0], ys - mu[k, 1]
        power = -0.5 * (A[k, 0, 0] * dx * dx + A[k, 1, 1] * dy * dy) \
                - A[k, 0, 1] * dx * dy
        a = opa[k] * np.exp(power)
        capped = a > ALPHA_MAX
        a = np.where(capped, ALPHA_MAX, a)
        a = np.where(a < ALPHA_MIN, 0.0, a)
        live[k] = (a > 0) & ~capped
        alphas[k] = a
        img += (T * a)[..., None] * color[k]
        T = T * (1.0 - a)
    return img, (alphas, live, T, A)


def backward(params, target, H, W):
    mu, s, theta, color, opa = activate(params)
    img, (alphas, live, T_final, A) = forward(params, H, W)
    K = len(mu)
    dLdC = (2.0 / img.size) * (img - target)
    ys, xs = np.mgrid[0:H, 0:W]
    g = {k: np.zeros_like(v) for k, v in params.items()}
    S = np.zeros((H, W, 3))
    T_run = T_final.copy()
    c_th, s_th = np.cos(theta), np.sin(theta)
    for k in range(K - 1, -1, -1):
        a = alphas[k]
        one_m = 1.0 - a
        T_k = T_run / one_m
        w = a * T_k
        g["color"][k] = np.einsum("hwc,hw->c", dLdC, w)
        dCda = T_k[..., None] * color[k] - S / one_m[..., None]
        dLda = np.einsum("hwc,hwc->hw", dLdC, dCda) * live[k]
        g["o_logit"][k] = np.sum(dLda * a / opa[k]) * opa[k] * (1.0 - opa[k])
        dLdp = dLda * a
        dx, dy = xs - mu[k, 0], ys - mu[k, 1]
        g["mu"][k] = [np.sum(dLdp * (A[k, 0, 0] * dx + A[k, 0, 1] * dy)),
                      np.sum(dLdp * (A[k, 1, 0] * dx + A[k, 1, 1] * dy))]
        dLdA = -0.5 * np.array([
            [np.sum(dLdp * dx * dx), np.sum(dLdp * dx * dy)],
            [np.sum(dLdp * dy * dx), np.sum(dLdp * dy * dy)]])
        dLdSig = -A[k] @ dLdA @ A[k]
        R = np.array([[c_th[k], -s_th[k]], [s_th[k], c_th[k]]])
        Rp = np.array([[-s_th[k], -c_th[k]], [c_th[k], -s_th[k]]])
        D = np.diag(s[k] ** 2)
        for m in range(2):
            g["log_s"][k, m] = 2 * s[k, m] ** 2 * (R[:, m] @ dLdSig @ R[:, m])
        g["theta"][k] = np.sum(dLdSig * (Rp @ D @ R.T + R @ D @ Rp.T))
        S = S + w[..., None] * color[k]
        T_run = T_k
    return float(np.mean((img - target) ** 2)), img, g


class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * grads[k]
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * grads[k] ** 2
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr[k] * mhat / (np.sqrt(vhat) + self.eps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--out", default="out/fit2d.png")
    ap.add_argument("--seed", type=int, default=8)
    args = ap.parse_args()

    H = W = args.size
    target = make_target(args.size)
    rng = np.random.default_rng(args.seed)
    mu = np.stack([rng.uniform(2, W - 2, args.k),
                   rng.uniform(2, H - 2, args.k)], -1)
    params = {"mu": mu,
              "log_s": np.log(rng.uniform(2.5, 5.0, (args.k, 2))),
              "theta": rng.uniform(0, np.pi, args.k),
              "color": target[mu[:, 1].astype(int), mu[:, 0].astype(int)].copy(),
              "o_logit": np.full(args.k, np.log(0.25 / 0.75))}
    lr = {"mu": 2e-3 * W, "color": 1e-2, "log_s": 5e-3,
          "theta": 5e-3, "o_logit": 5e-3}

    opt = Adam(params, lr)
    for it in range(args.steps):
        loss, img, grads = backward(params, target, H, W)
        opt.step(params, grads)
        if it % 50 == 0 or it == args.steps - 1:
            p = -10 * np.log10(np.mean((np.clip(img, 0, 1) - target) ** 2))
            print(f"step {it:4d}  loss {loss:.5f}  psnr {p:.2f} dB", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    final, _ = forward(params, H, W)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(target); axes[0].set_title("target")
    axes[1].imshow(np.clip(final, 0, 1)); axes[1].set_title("fitted")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
