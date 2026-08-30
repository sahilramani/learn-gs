# Third-party notices

This repository vendors no third-party source. Every file in `src/`, `cuda/`,
`train/`, `scripts/`, `tools/`, `notebooks_src/`, `notebooks/`, `colab/`, and
`docs/` is original work covered by [LICENSE](LICENSE). The notices below cover
code this project depends on but does not redistribute, code that appears in
the published site, and material a reader may bring to it.

## Dependencies

Installed by `requirements.txt`, never bundled here. Each remains under its own
license.

| Package | License |
|---|---|
| numpy | BSD-3-Clause (bundled components under 0BSD, MIT, Zlib, CC0-1.0) |
| matplotlib | Matplotlib License (PSF-derived, BSD-compatible) |
| jupyterlab | BSD-3-Clause |
| jupytext | MIT |
| nbclient | BSD-3-Clause |
| nbconvert | BSD-3-Clause |
| plyfile | GPL-3.0-or-later |
| torch | BSD-3-Clause |
| ninja | Apache-2.0 |

The GitHub Actions workflows in `.github/workflows/` use the `actions/checkout`,
`actions/setup-python`, `actions/configure-pages`, `actions/upload-pages-artifact`,
and `actions/deploy-pages` actions, all MIT. They run in CI and ship in nothing.

## plyfile is copyleft

`plyfile` is GPL-3.0-or-later, and it is the one dependency whose terms reach
past its own code. Three facts bound that reach here.

The installable package declares only numpy and matplotlib as dependencies, and
every module under `src/gsplat_edu` imports nothing outside numpy and the
standard library. Installing or redistributing `gsplat_edu` involves no GPL code.

Within this repository `plyfile` is imported in exactly one place: notebook 07,
`notebooks_src/07_render_a_real_scene.py`, which reads and writes the trained
`.ply` format. That notebook's own text and code are MIT like everything else.
Running it against a GPL library on your own machine is plain use, which the
GPL does not restrict. The same holds for the Colab editions, whose setup cells
`pip install plyfile` into a throwaway VM.

The obligation lands on a reader who lifts the ply-loading path out of notebook
07 into software they distribute. Combining that code with `plyfile` produces a
work whose distribution is governed by the GPL, and MIT permission from this
repository does not relax it. Write your own PLY reader, or accept the GPL for
the result.

## The published site

`tools/build_site.py` renders the notebooks with nbconvert's lab template and a
GitHub Actions workflow deploys the result to GitHub Pages. `site/` is
gitignored, so none of the following is in this repository. It is served from
the published pages.

Each page carries the JupyterLab stylesheet that nbconvert's lab template
inlines, which is BSD-3-Clause, Copyright (c) Jupyter Development Team. The CSS
this project adds on top is original: it sets the `--jp-*` variables that
template already reads, and recolours the pages without copying the template's
rules.

That same template loads MathJax to typeset the LaTeX in markdown cells.
MathJax is Apache-2.0, Copyright (c) The MathJax Consortium, and is fetched
from its CDN rather than served from this project. `tune_mathjax` in
`tools/build_site.py` rewrites two settings in the emitted configuration; it
modifies no MathJax source.

The pages link three faces from Google Fonts, Space Grotesk, JetBrains Mono,
and Instrument Serif, each under the SIL Open Font License 1.1 and served by
Google. The site's base theme is loaded from `sahilramani.com`, which is the
author's own. The Colab badge in the README is served by Google from
`colab.research.google.com`.

## Scene data

No data is distributed with this repository, and nothing in it downloads any.
`data/` is gitignored except for its README. Any scene you place there carries
the license of whoever produced it, entirely separate from this repository's
MIT grant.

This matters most for the pretrained models linked from the Inria 3D Gaussian
Splatting release, which are published for non-commercial research use. The
same applies to the datasets those models were trained on, several of which
carry their own research-only terms. Read the terms attached to a scene before
using it for anything beyond study.

## Papers and prior implementations

The algorithms taught here are published research, cited in the README: Kerbl
et al. on 3D Gaussian Splatting, Zwicker et al. on EWA volume splatting, and Yu
et al. on Mip-Splatting. The implementations are written from scratch, derived
from the published mathematics rather than copied from any existing codebase.
No code originates in `graphdeco-inria/gaussian-splatting`,
`diff-gaussian-rasterization`, or `gsplat`, and the non-commercial terms on the
first two do not apply to anything here.

The MIT grant in [LICENSE](LICENSE) is a copyright license to this source. It
grants no rights under any patent held by a third party, and no such patents
have been surveyed. Anyone deploying these techniques commercially should do
that analysis independently.
