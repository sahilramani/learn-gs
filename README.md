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

### Phase B - real scenes [next]

- `06_spherical_harmonics`: view-dependent color. Why per-splat RGB fails on real capture, SH basis, degree 0-3 evaluation.
- `07_render_a_real_scene`: load a trained `.ply` from the official 3DGS release, render it with our numpy renderer. Slow, small resolution, real.

### Phase C - training

- `08_fit_an_image_2d`: optimize 2D Gaussians to reproduce a photo. Gradients derived and coded by hand in numpy. This is the backward pass, learned concretely.
- `09_train_3d_torch`: the forward pass rewritten in PyTorch, autograd training on a small synthetic scene. Adam, opacity reset, densify and prune.
- `train/train_2d.py`, `train/train_3d.py`: the notebook pipelines as scripts.

### Phase D - CUDA

- `10_cuda_naive`: one thread per pixel, loop over all Gaussians. Correct and slow. Compiled via `torch.utils.cpp_extension`.
- `11_cuda_tiled`: 16x16 tiles, per-tile splat lists, (tile, depth) key sort, shared-memory batched rasterization. The actual 3DGS design.
- `12_cuda_backward`: analytic gradients in the kernel, full training loop on real data. A minimal `diff-gaussian-rasterization`.

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
