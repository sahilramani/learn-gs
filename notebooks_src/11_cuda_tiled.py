# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 11 - Tiles: the actual 3DGS design
#
# Notebook 10's kernel pays `N * H * W`: every pixel evaluates every
# splat, and the exercise priced the memory traffic of that term at
# terabytes per frame before caches. Almost all of it buys nothing. A
# splat's footprint is a few dozen pixels; for every pixel outside it, the
# read, the exponential, and the compare are dead work on a splat that
# cannot pass the alpha floor.
#
# The fix that ships in every real 3DGS renderer: carve the screen into
# 16x16 tiles and hand each tile the short list of splats whose bounding
# radius touches it. One block renders one tile; the block stages its
# list through shared memory in batches of 256, so each splat is read
# from global memory once per tile it covers instead of once per pixel.
# The dead term `N * H * W` becomes `sum(tiles covered)`, and this
# notebook builds the machinery that computes it: bounding radii, tile
# spans, duplicated keys, one sort, range extraction. All of that runs on
# the host in torch and is checked on any machine; only the final
# rasterize call needs the GPU.

# %%
import time
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import torch
from gsplat_edu import Camera, render_gaussians, sphere_shell

torch.manual_seed(11)
ROOT = pathlib.Path(".").resolve()

DILATION = 0.3
ALPHA_MIN = 1.0 / 255
ALPHA_MAX = 0.99
T_STOP = 1e-4
Z_NEAR = 0.05
TILE = 16

HAS_GPU = torch.cuda.is_available()
print("CUDA device:" if HAS_GPU else "No CUDA device -",
      torch.cuda.get_device_name(0) if HAS_GPU
      else "GPU cells skip; host-side machinery still runs and asserts.")

# %% [markdown]
# ## Host prep, plus the one number the tiles need
#
# Projection is notebook 10's prep verbatim. New is the bounding radius,
# the exact formula `docs/DECISIONS.md` fixed in notebook 02: three sigma
# of the dilated covariance's larger eigenvalue, ceiled.

# %%
def prep_splats(cam, means_t, cov3ds_t, colors_t, opacities_t):
    """Cull, project, dilate, invert, sort. Adds bounding radius."""
    with torch.no_grad():
        dev = means_t.device
        R = torch.tensor(cam.R, dtype=torch.float32, device=dev)
        eye = torch.tensor(cam.eye, dtype=torch.float32, device=dev)
        Pc = (means_t - eye) @ R.T
        vis = Pc[:, 2] > Z_NEAR
        Pc, cov, col, opa = Pc[vis], cov3ds_t[vis], colors_t[vis], opacities_t[vis]
        x, y, z = Pc.unbind(-1)
        u = cam.fx * x / z + cam.cx
        v = cam.fy * y / z + cam.cy
        zero = torch.zeros_like(z)
        J = torch.stack([
            torch.stack([cam.fx / z, zero, -cam.fx * x / z ** 2], -1),
            torch.stack([zero, cam.fy / z, -cam.fy * y / z ** 2], -1)], -2)
        T_JW = J @ R
        S2 = T_JW @ cov @ T_JW.mT
        a = S2[:, 0, 0] + DILATION
        c = S2[:, 1, 1] + DILATION
        b = S2[:, 0, 1]
        det = a * c - b * b
        ok = det > 0
        conic = torch.stack([c / det, -b / det, a / det], -1)
        mid = 0.5 * (a + c)
        lam_max = mid + torch.sqrt(torch.clamp(mid * mid - det, min=0.1))
        radius = torch.ceil(3.0 * torch.sqrt(lam_max))
        order = torch.argsort(z[ok])
        pick = lambda t: t[ok][order].contiguous()
        return (pick(torch.stack([u, v], -1)), pick(conic), pick(col),
                pick(opa), pick(z), pick(radius))

# %% [markdown]
# ## Tile lists as one sort
#
# Assigning splats to tiles wants no pointer-chasing data structure on a
# GPU; the trick that replaces it is the heart of the 3DGS paper's
# implementation. Emit one entry per (splat, covered tile) pair; pack each
# entry's sort key as
#
# ```
# key = (tile_id << 32) | float_bits(depth)
# ```
#
# and sort the entries once. Everything falls out: entries group by tile,
# and within a tile they sit in depth order, because the IEEE-754 bit
# pattern of a positive float is monotone in the float. Each tile's list
# is then one contiguous range of the sorted array, found by binary
# search. The official implementation sorts with cub's radix sort; here
# `torch.sort` stands in. The keys, the order, and the ranges are
# identical, torch just sorts slower, and at notebook scale the sort is
# nowhere near the frame's critical path.

# %%
def tile_spans(means2d, radius, W, H):
    """Inclusive-exclusive tile index box covered by each splat's square."""
    TX, TY = (W + TILE - 1) // TILE, (H + TILE - 1) // TILE
    u, v = means2d.unbind(-1)
    x_lo = torch.clamp(torch.floor((u - radius) / TILE).long(), 0, TX)
    x_hi = torch.clamp(torch.floor((u + radius) / TILE).long() + 1, 0, TX)
    y_lo = torch.clamp(torch.floor((v - radius) / TILE).long(), 0, TY)
    y_hi = torch.clamp(torch.floor((v + radius) / TILE).long() + 1, 0, TY)
    return x_lo, x_hi, y_lo, y_hi, TX, TY

def build_tile_lists(means2d, radius, depths, W, H):
    """Duplicate, pack keys, sort, extract ranges.

    Returns ranges (n_tiles, 2) int32, point_list (n_dup,) int32.
    """
    x_lo, x_hi, y_lo, y_hi, TX, TY = tile_spans(means2d, radius, W, H)
    spanx = torch.clamp(x_hi - x_lo, min=0)
    spany = torch.clamp(y_hi - y_lo, min=0)
    counts = spanx * spany
    total = int(counts.sum())
    off = torch.cumsum(counts, 0) - counts
    sid = torch.repeat_interleave(torch.arange(len(counts)), counts)
    w = torch.arange(total) - off[sid]
    tx = x_lo[sid] + w % spanx[sid]
    ty = y_lo[sid] + w // spanx[sid]
    tile_id = ty * TX + tx

    depth_bits = depths.to(torch.float32)[sid].view(torch.int32).long()
    keys = (tile_id << 32) | depth_bits
    keys_sorted, order = torch.sort(keys)
    point_list = sid[order].to(torch.int32)

    tiles_sorted = keys_sorted >> 32
    grid = torch.arange(TX * TY)
    starts = torch.searchsorted(tiles_sorted, grid)
    ends = torch.searchsorted(tiles_sorted, grid, right=True)
    ranges = torch.stack([starts, ends], -1).to(torch.int32)
    return ranges, point_list, tile_id, sid

# %% [markdown]
# Three claims just went by, each checkable right now on the CPU: the bit
# trick orders depths, the spans cover exactly the tiles a splat's square
# touches, and the ranges partition the sorted list tile by tile with
# depth nondecreasing inside every tile.

# %%
gen = torch.Generator().manual_seed(11)

# claim 1: positive-float bits sort like the floats
d = (torch.rand(20_000, generator=gen) * 50 + Z_NEAR)
bit_order = torch.argsort(d.view(torch.int32).long())
assert bool((d[bit_order].diff() >= 0).all())

# claim 2: spans match a brute-force intersection test
u = torch.rand(300, 2, generator=gen) * 80 - 8      # includes off-screen
r = torch.randint(1, 30, (300,), generator=gen).float()
x_lo, x_hi, y_lo, y_hi, TX, TY = tile_spans(u, r, 64, 64)
for i in range(0, 300, 23):
    for tx in range(TX):
        for ty in range(TY):
            overlap = (tx * TILE <= u[i, 0] + r[i]
                       and (tx + 1) * TILE > u[i, 0] - r[i]
                       and ty * TILE <= u[i, 1] + r[i]
                       and (ty + 1) * TILE > u[i, 1] - r[i])
            listed = (x_lo[i] <= tx < x_hi[i]) and (y_lo[i] <= ty < y_hi[i])
            assert overlap == listed

# claim 3: ranges partition the list; depth sorted within each tile
means, cov3ds, colors, opacities = sphere_shell()
cam = Camera.looking_at(eye=[0, 0, -3], target=[0, 0, 0],
                        fx=300, fy=300, H=256, W=256)
tt = [torch.tensor(t, dtype=torch.float32)
      for t in (means, cov3ds, colors, opacities)]
means2d, conics, cols, opas, depths, radius = prep_splats(cam, *tt)
ranges, point_list, tile_of_dup, sid = build_tile_lists(
    means2d, radius, depths, cam.W, cam.H)

lens = (ranges[:, 1] - ranges[:, 0]).long()
assert int(lens.sum()) == len(point_list)
assert bool((ranges[1:, 0] == ranges[:-1, 1]).all())
tile_sorted = tile_of_dup[torch.argsort((tile_of_dup << 32)
                          | depths[sid].view(torch.int32).long())]
for t in range(0, len(ranges), 37):
    s, e = int(ranges[t, 0]), int(ranges[t, 1])
    if e > s:
        assert bool((tile_sorted[s:e] == t).all())
        dseg = depths[point_list[s:e].long()]
        assert bool((dseg.diff() >= -1e-6).all())
print(f"tile machinery holds: {len(point_list):,} duplicated entries, "
      f"{len(ranges)} tiles")

# %% [markdown]
# The duplication count is the new traffic bill. Compare it with what the
# naive kernel reads for the same frame:

# %%
naive_reads = len(means2d) * cam.H * cam.W
tiled_reads = len(point_list) * TILE * TILE     # shared-mem reads, cheap
tiled_global = len(point_list)                  # global reads, the real bill
print(f"naive:  {naive_reads / 1e6:8.1f}M global splat reads per frame")
print(f"tiled:  {tiled_global / 1e6:8.3f}M global splat reads per frame "
      f"({naive_reads / tiled_global:.0f}x fewer)")

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
img_ctx, _ = render_gaussians(cam, means, cov3ds, colors, opacities)
axes[0].imshow(np.clip(img_ctx, 0, 1))
big = int(torch.argmax(radius))
uu, vv, rr = float(means2d[big, 0]), float(means2d[big, 1]), float(radius[big])
xl, xh, yl, yh, TX, TY = tile_spans(means2d[big:big + 1],
                                    radius[big:big + 1], cam.W, cam.H)
for t in range(0, cam.W + 1, TILE):
    axes[0].axvline(t - 0.5, lw=0.3, c="w", alpha=0.5)
    axes[0].axhline(t - 0.5, lw=0.3, c="w", alpha=0.5)
for tx in range(int(xl[0]), int(xh[0])):
    for ty in range(int(yl[0]), int(yh[0])):
        axes[0].add_patch(plt.Rectangle((tx * TILE - 0.5, ty * TILE - 0.5),
                                        TILE, TILE, fc="yellow", alpha=0.25))
axes[0].add_patch(plt.Circle((uu, vv), rr, fill=False, color="yellow"))
axes[0].set_title("one splat's radius, and the tiles that get it")
axes[0].axis("off")
axes[1].hist(lens.numpy(), bins=50, log=True)
axes[1].set_xlabel("splats per tile"); axes[1].set_ylabel("tiles (log)")
axes[1].set_title("tile list lengths: work now proportional to coverage")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## The kernel
#
# `cuda/tiled/tiled.cu`. One block per tile, one thread per pixel, and the
# loop alternates two roles for every thread:
#
# ```
# for (base = start; base < end; base += 256) {
#     if (__syncthreads_count(done) == 256) break;   // whole tile opaque
#     if (base + tid < end)  s_data[tid] = fetch(point_list[base + tid]);
#     __syncthreads();
#     if (!done)  composite the batch from shared memory;
# }
# ```
#
# The `__syncthreads_count` line is the per-tile version of the numpy
# renderer's `T_STOP` skip from notebook 05: it votes across the block,
# and only when all 256 pixels are saturated does the tile stop reading.
#
# ## Exercise
#
# A thread whose pixel saturates sets `done` and stops compositing. Why
# must it keep executing the loop anyway, fetching splats it will never
# use? Two reasons, one about correctness, one about the hardware's
# rules. Work it before scrolling.

# %% [markdown]
# ## Solution
#
# First, the fetches are cooperative: thread `tid` loads shared slot
# `tid` for the whole block, so a retired thread is still some other
# pixel's memory subsystem. Second, `__syncthreads` is a block-wide
# barrier, and CUDA leaves the behavior undefined if any thread of the
# block skips it; a thread that returned early would deadlock or corrupt
# the block on real hardware. Threads therefore retire by flag, never by
# leaving, and the block retires only by the unanimous vote above.

# %%
if HAS_GPU:
    from torch.utils.cpp_extension import load
    naive = load(name="naive_raster",
                 sources=[str(ROOT / "cuda" / "naive" / "naive.cu")])
    tiled = load(name="tiled_raster",
                 sources=[str(ROOT / "cuda" / "tiled" / "tiled.cu")])

    args_cpu = (means2d, conics, cols, opas)
    args_gpu = [t.cuda() for t in args_cpu]
    r_gpu = ranges.cuda(); pl_gpu = point_list.cuda()

    img_n, _ = naive.rasterize(*args_gpu, cam.H, cam.W,
                               ALPHA_MIN, ALPHA_MAX, T_STOP)
    img_t, T_t = tiled.rasterize(r_gpu, pl_gpu, *args_gpu, cam.H, cam.W,
                                 ALPHA_MIN, ALPHA_MAX, T_STOP)
    diff = (img_t - img_n).abs().max().item()
    print(f"tiled vs naive: max |diff| {diff:.5f}")
    assert diff < 2e-3, "tiled kernel disagrees with naive beyond tails"

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(np.clip(img_t.cpu().numpy(), 0, 1))
    axes[0].set_title("tiled kernel"); axes[0].axis("off")
    axes[1].imshow((img_t - img_n).abs().max(-1).values.cpu().numpy(),
                   cmap="inferno")
    axes[1].set_title("abs diff vs naive"); axes[1].axis("off")
    plt.tight_layout(); plt.show()
else:
    print("skipped: kernel parity (no CUDA device)")

# %% [markdown]
# The tolerance story is the radius, this time in the other direction:
# the tiled path never evaluates a splat outside its 3-sigma square, the
# naive path evaluates everything and keeps whatever clears the alpha
# floor. Same above-floor ring notebook 10 measured, seen from the other
# side and slightly wider, because the tile span clips at exactly
# u +- r while the reference bbox rounds outward; the measured diff
# lands near 1.5e-3 and the assert allows the same 2e-3.
#
# ## The speedup curve

# %%
if HAS_GPU:
    def bench(fn, reps=20):
        fn(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    rows = []
    for n in [10_000, 40_000, 160_000, 640_000]:
        m, c3, col, op = sphere_shell(n, s_t=0.06 * np.sqrt(2500 / n),
                                      s_n=0.01)
        p = prep_splats(cam, *[torch.tensor(t, dtype=torch.float32)
                               for t in (m, c3, col, op)])
        rg, pl, _, _ = build_tile_lists(p[0], p[5], p[4], cam.W, cam.H)
        a = [t.cuda() for t in p[:4]]
        rg, pl = rg.cuda(), pl.cuda()
        t_n = bench(lambda: naive.rasterize(*a, cam.H, cam.W,
                                            ALPHA_MIN, ALPHA_MAX, T_STOP))
        t_t = bench(lambda: tiled.rasterize(rg, pl, *a, cam.H, cam.W,
                                            ALPHA_MIN, ALPHA_MAX, T_STOP))
        rows.append((n, t_n, t_t))
        print(f"{n:>7d} splats: naive {t_n * 1e3:8.2f} ms   "
              f"tiled {t_t * 1e3:7.2f} ms   ({t_n / t_t:5.1f}x)")

    ns, tn, tt_ = zip(*rows)
    plt.figure(figsize=(6.5, 3.4))
    plt.loglog(ns, tn, "o-", label="naive")
    plt.loglog(ns, tt_, "s-", label="tiled")
    plt.axhline(0.016, ls="--", c="gray", label="16 ms budget")
    plt.xlabel("splats"); plt.ylabel("seconds per frame"); plt.legend()
    plt.title("the fix: cost tracks coverage, and the budget is reachable")
    plt.tight_layout(); plt.show()
else:
    print("skipped: speedup curve (no CUDA device)")

# %% [markdown]
# ## The next live problem
#
# Rendering is solved twice over now, and training is exactly where
# notebook 09 left it: autograd through the dense torch forward,
# hundreds of milliseconds per iteration, because the fast kernels above
# compute an image and keep no memory of how. Gradients need the
# backward pass through this tiled design: the same tile walk run in
# reverse, notebook 08's transmittance recurrence reconstructing each
# `T_i` from the end state, and per-splat gradients accumulated across
# every pixel a splat touched. That kernel, wrapped as a
# `torch.autograd.Function` so torch's autograd drives it, is notebook
# 12, and it is the last piece of the whole method.
