# CLAUDE.md

Educational Gaussian splatting: a renderer and trainer built notebook by
notebook, from "project a point" to a CUDA tile rasterizer with training.
The reader is someone learning the field. Clarity beats performance
everywhere except phase D, where performance is the subject.

## Status

Notebooks 01-06 shipped: phase A (renderer, package `gsplat_edu`,
`scripts/render_toy.py`) plus spherical harmonics (`gsplat_edu.sh`).
Next task: notebook 07. Specs and order live in `docs/ROADMAP.md`.

## The prime rule

Nothing is introduced until a problem in front of the reader forces it.
Every notebook shows the break before the fix. Never add machinery because a
later notebook will need it; if a concept has no live forcing problem,
restructure until it does. Each notebook ends by naming the next live problem.

## Workflow

- Notebook sources are `notebooks_src/*.py` (jupytext percent format).
  Never hand-edit `notebooks/*.ipynb`.
- Build: `python tools/build_notebooks.py [pattern]`. Executes and writes
  `.ipynb` with embedded figures. Inline asserts are the test suite; a raised
  assert fails the build.
- Before any commit that touches sources or `src/`: run the full build with
  no pattern. All notebooks must still pass, not just the new one.
- Notebooks are self-contained: they redefine helpers inline even when the
  package already has them. When code stabilizes it gets promoted to
  `src/gsplat_edu`, and the promoting notebook keeps its inline copy plus an
  exact-equality assert against the package import (see notebook 05).
- `python scripts/render_toy.py` is the package smoke test.

## Style, strict

- ASCII only in sources and code: straight quotes, `->` not arrows, hyphens
  not em dashes. The build tool rejects non-ASCII.
- Prose: short declarative sentences. Derive, state, move on. Markdown cells
  stay short. Banned: "Let's", "It's worth noting", rhetorical
  question-then-answer, "not X but Y" reveal constructions, bold-led bullet
  lists, stakes inflation, vague attributions ("experts say"). Prefer prose
  over bullets.
- Exercises come before their solutions, in separate cells, with an explicit
  instruction to attempt first.
- Every new formula gets an inline assert, a finite-difference check, or a
  Monte Carlo check in the same notebook.
- Code: numpy first, plain loops where clearer. CPU slowness is not a bug;
  it is measured on purpose and becomes the motivation for phase D.

## Conventions

`docs/DECISIONS.md` is binding: coordinate convention, quaternion order,
kernel normalization, every renderer constant. Do not change a decision
silently. If one must change, update the doc and every notebook that states
it, in the same commit.

## Environment

- Setup: `pip install -r requirements.txt` (installs `gsplat_edu` editable
  plus the notebook toolchain).
- Phases A-C are CPU-only. Phase D needs an NVIDIA GPU, CUDA toolkit, and
  torch; notebooks 10-12 must check `torch.cuda.is_available()` and degrade
  to clear instructions when it is false, so the build never hard-fails on a
  CPU machine.
- New dependencies only when the phase that needs them starts; keep
  `requirements.txt` phase-commented as it is now.

## Commits

Short imperative subject, body only when the change needs explanation.
Do not mention assistants or code-generation tools in commit or PR messages.
Do not renumber existing notebooks.
