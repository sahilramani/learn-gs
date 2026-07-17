# Roadmap

Per-notebook specs. Build strictly in order; each spec assumes everything
before it exists. Statuses: [x] shipped, [ ] next up in sequence.

Global acceptance for every new notebook:

1. `python tools/build_notebooks.py` (no pattern) exits 0.
2. Opens with the forcing problem; contains at least one figure showing the
   break and one showing the fix.
3. Every new formula has an assert, finite-difference check, or Monte Carlo
   check in-notebook.
4. Ends by naming the next live problem.
5. README curriculum table and this file's checkboxes updated in the same
   commit.

## Phase A - CPU forward renderer [x]

- [x] 01 points and why they fail
- [x] 02 the 2D gaussian primitive
- [x] 03 3D gaussians and cameras
- [x] 04 projection (EWA)
- [x] 05 sort and composite, promotion to `gsplat_edu`

## Phase B - real scenes

### [x] 06_spherical_harmonics

Forcing problem: the artifact we want to render next (a trained 3DGS `.ply`)
stores 48 numbers per splat for color, not 3. Real captures need
view-dependent color (specular sheen, baked lighting); flat RGB cannot
express it. Open with that fact, then build the machinery.

Content, in order:

1. View direction per splat: `d = normalize(mean - cam.eye)`.
2. The SH basis, degrees 0-3, with the exact constants the whole GS
   ecosystem uses:

```
C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
      -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
      0.3731763325901154, -0.4570457994644658, 1.445305721320277,
      -0.5900435899266435]
```

   Evaluation, matching the official CUDA (`x, y, z` = view dir components):

```
deg 0: c  = C0*sh[0]
deg 1: c += -C1*y*sh[1] + C1*z*sh[2] - C1*x*sh[3]
deg 2: c += C2[0]*x*y*sh[4] + C2[1]*y*z*sh[5]
          + C2[2]*(2z^2 - x^2 - y^2)*sh[6]
          + C2[3]*x*z*sh[7] + C2[4]*(x^2 - y^2)*sh[8]
deg 3: c += C3[0]*y*(3x^2 - y^2)*sh[9] + C3[1]*x*y*z*sh[10]
          + C3[2]*y*(4z^2 - x^2 - y^2)*sh[11]
          + C3[3]*z*(2z^2 - 3x^2 - 3y^2)*sh[12]
          + C3[4]*x*(4z^2 - x^2 - y^2)*sh[13]
          + C3[5]*z*(x^2 - y^2)*sh[14] + C3[6]*x*(x^2 - 3y^2)*sh[15]
final: color = clip(c + 0.5, 0, None)
```

3. Why a basis on the sphere at all, kept short: smooth low-frequency
   functions of direction, few coefficients, closed-form evaluation.
4. Figures: a lobe/polar visualization of a few basis functions; one splat
   whose color shifts across an orbit with hand-authored coefficients; the
   toy sphere rendered deg 0 vs deg 3 with a fake specular lobe added,
   orbit montage.

Asserts: Monte Carlo orthonormality of the implemented basis
(`integral(Y_i Y_j) ~ delta_ij` over uniform sphere samples, tolerance a few
percent at 200k samples; this catches any constant or sign slip); deg 0
reduces to a constant color; clip behavior at negative values.

Promotion: `src/gsplat_edu/sh.py` with the constants and
`eval_sh(deg, sh, dirs)` where `sh` is `(N, 16, 3)` and `dirs` is `(N, 3)`.
Renderer stays color-agnostic: callers compute `colors = eval_sh(...)` and
pass RGB in, `render_gaussians` unchanged.

### [x] 07_render_a_real_scene

Forcing problem: every scene so far was hand-placed; the method exists to
render photographed reality. A trained `.ply` is photographed reality in our
exact representation.

Content:

1. Add `plyfile` to requirements (uncomment). Data policy: nothing is
   downloaded by code. `data/README.md` tells the user where to get scenes
   (official Inria pretrained models, or any 3DGS training output) and where
   to put them: `data/<scene>/point_cloud.ply`.
2. The field layout to parse: `x y z`, `nx ny nz` (unused),
   `f_dc_0..2`, `f_rest_0..44`, `opacity`, `scale_0..2`, `rot_0..3`.
   Activations per docs/DECISIONS.md. `rot` is wxyz.
   SH assembly: `f_dc` is `(N, 3)`; `f_rest` was flattened from `(N, 3, 15)`,
   so load as `(N, 3, 15)`, transpose to `(N, 15, 3)`, concat dc in front
   for `(N, 16, 3)`. Verify this layout against a round-trip: write a tiny
   synthetic ply with known values, read it back, assert bit-exact.
3. Batched parameter conversion: `quat_to_R_batch`, `build_cov3d_batch`
   (einsum), promoted to `gsplat_edu.gaussians` with asserts against the
   scalar versions on random inputs.
4. CPU budget: random-subsample to 30k-80k splats, render around 400 px.
   State the honest cost and that phase D exists because of it.
5. Camera: compute scene center (mean of means) and radius (90th percentile
   distance), orbit at 1.5x radius. Render an orbit montage. Save a frame to
   `out/`.

Asserts: ply round-trip; activated opacities in (0, 1); activated scales
positive; loaded quats normalize cleanly.

Acceptance: a recognizable render of a real scene in the executed notebook.
If no scene file is present, the notebook must degrade to a clear message
plus the synthetic round-trip still passing, so the build stays green.

## Phase C - training

### [x] 08_fit_an_image_2d

Forcing problem: nothing chooses the gaussians. Strip the problem to 2D
where every gradient can be derived by hand and checked.

Content:

1. Target: procedural image, 96x96 (colorful rings plus a gradient), no
   external assets.
2. Parameters per gaussian, K around 300: `mu` (2), log-scales (2), `theta`
   (1), color (3), opacity logit (1). Forward pass is notebook 05's
   compositing in 2D, written to also record what backward needs.
3. Backward, derived in markdown then coded in numpy:
   - MSE to `dL/dC` per pixel.
   - Through the over operator: process splats back to front per pixel;
     reconstruct `T_i` by dividing the running transmittance by
     `(1 - alpha_i)`; maintain a running suffix of blended color behind `i`;
     from those, `dL/dalpha_i` and `dL/dc_i`. This recurrence is exactly the
     structure of the official CUDA backward and gets reused in notebook 12.
   - `dalpha` through opacity and the kernel; kernel through conic and `mu`;
     conic through `Sigma` via `d(S^-1) = -S^-1 dS S^-1`; `Sigma` through
     log-scales and `theta`.
4. Adam in numpy (betas 0.9/0.999, eps 1e-8), per-group learning rates
   (starting point: mu 2e-3 in normalized coords, color 1e-2, rest 5e-3;
   tune and record).
5. Figures: target / init / fitted; loss curve; montage across iterations.

Asserts: full finite-difference gradcheck on a tiny config (3 gaussians,
16x16, float64, rtol 1e-4) covering every parameter group; loss strictly
decreases over the first 50 steps.

Script: `train/train_2d.py`, same pipeline, argparse for K/steps/output.

Acceptance: fitted image visibly reproduces the target; report PSNR.

### [x] 09_train_3d_torch

Forcing problem: hand gradients through the full 3D pipeline are a phase D
job; autograd buys correctness now at the price of speed, and the speed bill
motivates CUDA.

Content:

1. Forward in torch, differentiable end to end: batched projection, per
   splat bbox compositing without python-per-pixel loops. Small budget:
   128 px, K up to about 2000.
2. Ground truth: 20-40 renders of the toy sphere shell from the numpy
   renderer (self-supervised setup, no COLMAP needed). Init: random cloud in
   the unit ball, random color, low opacity.
3. Parameterization and activations per docs/DECISIONS.md; SH degree 0 only
   here. Loss: L1. Per-group Adam (starting point: means 2e-4, scales 5e-3,
   quats 1e-3, opacity 5e-2, color 2.5e-3; tune and record).
4. Densification loop, simplified 3DGS: every 100 iters in a window, clone
   small high-gradient splats, split large ones (scale / 1.6, two copies),
   prune opacity < 0.005. Track screen-space mean-gradient norms as the
   signal, threshold around 2e-4. Document every deviation from the paper.
5. `torch.autograd.gradcheck` on a tiny double-precision forward.
6. Figures: init / mid / final vs target for a held-out camera; loss curve;
   splat count over time.

Script: `train/train_3d.py`.

Acceptance: held-out view improves dramatically over init and is
recognizably the sphere; gradcheck passes.

## Phase D - CUDA

All three notebooks guard on `torch.cuda.is_available()` and degrade to
instructions on CPU machines, keeping the build green. Kernels live in
`cuda/`, compiled at import via `torch.utils.cpp_extension.load`.

### [x] 10_cuda_naive

Forcing problem: the pairs/s numbers from 05 and the seconds-per-iteration
from 09, against real scenes at 60 fps. Quote the measured gap.

Content: host does cull, project, sort (torch). Kernel `cuda/naive/`: one
thread per pixel, loop all sorted splats, over-composite with early exit.
Parity assert vs `gsplat_edu.render_gaussians` on the toy scene, fp32
tolerance atol 2e-3 (see DECISIONS.md, "Tile rasterizer"). Timing table:
numpy vs naive CUDA at growing splat counts.

### [x] 11_cuda_tiled

Forcing problem: the naive kernel makes every pixel read every splat;
measured bandwidth is the wall. The fix is the actual 3DGS design.

Content: 16x16 tiles; per-splat tile span from the bounding radius; one
duplicated entry per covered tile with key
`(tile_id << 32) | float_bits(depth)`; sort int64 keys with `torch.sort`
(state plainly that production uses cub radix sort and why the stand-in is
fine here); tile ranges from key boundaries; rasterize kernel: one block per
tile, one thread per pixel, shared-memory batches of splat data (batch size
256), per-thread transmittance, block-wide early exit via
`__syncthreads_count`. Parity vs naive; speedup curve vs splat count.

### [x] 12_cuda_backward

Forcing problem: notebook 09's wall-clock, now that the forward is fast.

Content: forward additionally stores per-pixel final T and last-contributor
index. Backward kernel walks each tile back to front using notebook 08's
recurrence, produces grads for color, opacity, 2D mean, conic; chain to 3D
(cov3d through `T = J W`, scale and quat through the `R S` product, mean
through `J`'s dependence on the mean as in the official
`computeCov2DCUDA` backward). Wrap in `torch.autograd.Function`. Gradcheck
against notebook 09's autograd forward on a tiny scene. Then: full training
run, real scene if `data/` has one, toy otherwise. This is a minimal
`diff-gaussian-rasterization`.

Acceptance: gradcheck passes; training matches notebook 09 quality at an
order of magnitude or more speedup; report both numbers.

## Phase E - measurement [optional]

Profile naive vs tiled with the torch profiler and, where available, nsight
compute: stage breakdown (preprocess, sort, rasterize), occupancy, memory
traffic. Findings go in a final notebook or `docs/PROFILE.md`. No new
features; this phase only explains the numbers the others produced.
