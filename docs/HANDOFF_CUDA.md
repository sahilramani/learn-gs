# Handoff: GPU validation of phase D

For whoever (or whatever) picks this repo up on a machine with an NVIDIA
GPU. Read `CLAUDE.md` first; it is the working agreement and it wins over
this file. `docs/DECISIONS.md` is binding. `docs/ROADMAP.md` holds the
specs.

## State of the repo

Notebooks 01-12 are shipped and the build is green on a CPU-only machine.
Phases A-C are fully verified. Phase D (notebooks 10-12, kernels in
`cuda/`) was authored on a Mac with no NVIDIA hardware: the CUDA sources
have never met nvcc, and every GPU cell in notebooks 10-12 currently
guards on `torch.cuda.is_available()` and skips itself. The host-side
machinery (projection prep, tile spans, key packing, range extraction) is
asserted on CPU and passes.

Your job: make the first GPU build pass honestly.

## Setup

```
git clone https://github.com/sahilramani/learn-gs.git && cd learn-gs
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"   # must be True
nvcc --version                                               # toolkit present
```

Notes: `requirements.txt` pins nothing; on Linux, plain `pip install
torch` ships the CUDA wheel. If `torch.cuda.is_available()` is False,
stop and fix that before touching anything else (driver, or a CPU-only
wheel; reinstall from the matching CUDA index-url). `ninja` is in
requirements; `torch.utils.cpp_extension.load` needs it and nvcc on PATH.
No scene data is required: notebook 07 falls back to its synthetic
stand-in and notebook 12 trains on the toy sphere.

## The task

```
python tools/build_notebooks.py
```

That is the whole entry point. Notebooks 10-12 compile
`cuda/naive/naive.cu`, `cuda/tiled/tiled.cu`, and
`cuda/backward/diff_raster.cu` at import and run their GPU cells for the
first time. A raised assert fails the build; that is the test suite
working, not an obstacle to route around.

What must come out green, per notebook:

- 10: kernel vs `gsplat_edu.render_gaussians` parity, max |diff| < 1e-3
  on the toy scene; timing table prints numpy vs CUDA at four splat
  counts.
- 11: tiled vs naive parity < 1e-3; speedup curve prints naive vs tiled
  at four splat counts.
- 12: forward parity vs the dense torch renderer < 1e-3; per-parameter
  gradient relative L2 vs autograd < 1e-2; training run asserts >= 10x
  over the measured dense CPU baseline and held-out L1 < 0.3x init.

## Likely failures, in order of probability

1. nvcc compile errors. The kernels were written blind. Fix the syntax,
   keep the semantics: the backward recurrence must stay line-for-line
   notebook 08's derivation (see the comment blocks in
   `cuda/backward/diff_raster.cu` and the markdown in notebook 12).
2. Stale extension cache between attempts:
   `rm -rf ~/.cache/torch_extensions` and rebuild.
3. Arch detection: if load() dies picking a gencode, set
   `TORCH_CUDA_ARCH_LIST` (e.g. "8.6") and retry.
4. Parity asserts tripping. Do not loosen a tolerance to pass. The 1e-3
   tolerances have a stated cause (3-sigma bbox tails, fp32 accumulation;
   see DECISIONS.md, "Tile rasterizer"). If the diff is above tolerance,
   something is wrong in a kernel; the diff-map figures in 10 and 11 show
   where. Structured error (tile seams, offsets) means an indexing bug,
   not a tolerance problem.
5. The 12 speedup assert. If a real GPU shows less than 10x over the
   dense CPU baseline, investigate before weakening it; the expected
   margin is far larger. Report the measured number either way.

## Rules that bite here

- Never hand-edit `notebooks/*.ipynb`; sources are `notebooks_src/*.py`.
- Rebuilding notebooks whose sources did not change still dirties their
  .ipynb (cell ids, timestamps, timing prints). Before committing,
  `git checkout -- ` the .ipynb of every notebook whose source you did
  not touch, then commit the rest.
- A .cu fix is a source change: commit the fixed kernel together with
  the rebuilt notebooks 10-12.

## When the build is green

One commit, containing:

1. Any `cuda/*.cu` fixes.
2. Rebuilt `notebooks/10..12.ipynb`, now with real GPU figures and
   timings embedded.
3. The GPU-verification caveats removed, all in this same commit:
   the "Status caveat" paragraph in `docs/ROADMAP.md` (phase D intro),
   the "GPU verification pending" wording in `README.md` (phase D
   heading and the paragraph under its table), the caveat sentence in
   `CLAUDE.md` Status, and this file (delete `docs/HANDOFF_CUDA.md`; it
   is done serving).
4. Real numbers quoted where placeholders reasoned abstractly: the
   measured speedups in the README phase D table row for 12 if you touch
   that text.

Short imperative commit subject, e.g. "Validate phase D on GPU: fix
kernels, embed measured results". Push to origin.

## After that

Phase E in `docs/ROADMAP.md` is optional profiling (torch profiler,
nsight compute where available). It only explains numbers phase D
produced; nothing new gets built. Take it only if asked.
