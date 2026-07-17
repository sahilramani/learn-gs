# Gaussian Splatting from Scratch

An educational build of a 3D Gaussian Splatting renderer and trainer. Starts from
"project a point," ends at a CUDA tile rasterizer with a training loop.

One rule governs the whole project: nothing is introduced until a problem in front
of us forces it. Every primitive, formula, and optimization exists because the
previous version broke, and each notebook shows the break before the fix.

## Curriculum

### Phase A - CPU forward renderer (numpy) [done]

| Notebook | Problem it solves |
|---|---|
| `01_points_and_why_they_fail` | Pinhole camera, point cloud renderer. Breaks: points have no extent, surfaces turn to confetti. |
| `02_the_2d_gaussian_primitive` | A primitive with extent: the unnormalized 2D Gaussian. Covariance from R S, truncation, the 0.3 dilation and its energy bug. |
| `03_3d_gaussians_and_cameras` | 3D Gaussians (quaternion + scale). Monte Carlo shows perspective projection bends them: they stop being Gaussian. |
| `04_projection_ewa` | The fix: linearize projection at the mean. Jacobian, Sigma2d = J W Sigma Wt Jt, where the approximation holds and where it visibly fails. |
| `05_sort_and_composite` | Many splats per pixel: the over operator, global depth sort, full renderer. Points vs splats side by side. Code promoted to `src/gsplat_edu`. |

### Phase B - real scenes [done]

| Notebook | Problem it solves |
|---|---|
| `06_spherical_harmonics` | View-dependent color: flat RGB averages highlights away. SH basis degrees 0-3 with the ecosystem constants and signs, sheen demo, `eval_sh` promoted to the package. |
| `07_render_a_real_scene` | A trained `.ply` is photographed reality in our exact representation. Layout pinned by a bit-exact round-trip, activations, batched conversions, auto-framed orbit on an honest CPU budget. Degrades to a synthetic stand-in when `data/` is empty. |

### Phase C - training [done]

| Notebook | Problem it solves |
|---|---|
| `08_fit_an_image_2d` | Nothing chooses the Gaussians. Stripped to 2D: the over operator's backward recurrence derived by hand, finite-difference checked on every parameter, numpy Adam fits a procedural image. `train/train_2d.py`. |
| `09_train_3d_torch` | The 3D chain, differentiable in PyTorch: autograd owns the gradients (gradcheck-verified), a random cloud becomes the sphere from 24 self-rendered views, densify and prune with every paper deviation named. The ms/iter it prints is phase D's opening argument. `train/train_3d.py`. |

### Phase D - CUDA [done, GPU verification pending]

| Notebook | Problem it solves |
|---|---|
| `10_cuda_naive` | The measured CPU-vs-16ms gap. One thread per pixel, every thread reads every splat; parity with the numpy renderer, timing table, and the traffic arithmetic that convicts the design. |
| `11_cuda_tiled` | Naive traffic scales as splats x pixels. 16x16 tiles, `(tile << 32) \| float_bits(depth)` key sort, per-tile ranges, shared-memory batches, block-wide early exit. The actual 3DGS forward. |
| `12_cuda_backward` | Fast kernels learn nothing. Forward stores final T + last contributor; backward replays notebook 08's recurrence per tile with atomics, wrapped in `torch.autograd.Function`; training at kernel speed. A minimal `diff-gaussian-rasterization`. |

Phase D notebooks were authored on a CPU-only machine: kernel cells guard on
`torch.cuda.is_available()` and skip cleanly, and the host-side machinery
(tile spans, key packing, ranges) is asserted everywhere. The first notebook
build on an NVIDIA machine compiles the kernels and runs the parity,
gradcheck, and speedup asserts.

### Phase E - measurement

- Profiling with nsight, occupancy, why the sort and rasterize stages dominate, what the tile design buys.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # installs deps + this package (editable)
jupyter lab notebooks/
```

Phases A-C run on CPU. Phase D requires an NVIDIA GPU, the CUDA toolkit, and PyTorch.

Notebooks ship executed, so figures are visible without running anything. Each
contains inline `assert` checks; if a cell runs, the math it demonstrates held.
Exercises appear before their solutions. Attempt them: the next notebook usually
builds on the answer.

## Layout

```
notebooks/      the curriculum, in order, shipped executed
notebooks_src/  editable sources (jupytext percent); never hand-edit .ipynb
tools/          build_notebooks.py executes sources and writes notebooks/
src/            gsplat_edu: code promoted from notebooks once it stabilizes
scripts/        runnable demos (render_toy.py renders the Phase A test scene)
train/          training scripts (Phase C)
cuda/           kernels (Phase D)
data/           scenes and captures (gitignored, see data/README.md)
docs/           DECISIONS.md (binding conventions), ROADMAP.md (specs)
```

Contributing flow: edit a source in `notebooks_src/`, run
`python tools/build_notebooks.py`, commit both. `CLAUDE.md` holds the full
working agreement.

## References

- Kerbl, Kopanas, Leimkuehler, Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. SIGGRAPH 2023.
- Zwicker, Pfister, van Baar, Gross. EWA Volume Splatting. IEEE Visualization 2001.
- Yu et al. Mip-Splatting: Alias-free 3D Gaussian Splatting. CVPR 2024.
