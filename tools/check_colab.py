"""Run the Colab editions the way Colab runs them, and fail if they break.

Usage:
    python tools/check_colab.py                 # default set
    python tools/check_colab.py 05 07           # only matching editions
    python tools/check_colab.py --all           # every edition

Colab starts a kernel on a machine that has one .ipynb and nothing else,
then runs the setup cell, which clones this repo and puts the package on
the path. Nothing in tools/build_notebooks.py exercises that: it executes
from a checkout that already exists, with gsplat_edu already installed. So
the setup cell is the one piece of this project with no coverage, and it
is also the piece every Colab reader hits first.

This runs each edition in a scratch directory with a fresh kernel and no
repo present. The setup cell does the real clone over the network. Set
LEARN_GS_REF to test a branch other than the shipped default, which is what
CI does so a change is proven before it reaches anyone.

Notebooks 10-12 are excluded from the default set: their preflight cell
raises without a GPU, by design, and this runs on machines without one.
"""
import os
import shutil
import sys
import tempfile
import time
import pathlib
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = pathlib.Path(__file__).resolve().parents[1]
COLAB = ROOT / "colab"

# Needs a GPU; the preflight cell stops without one on purpose.
GPU_ONLY = ("10", "11", "12")

# Enough to cover what the setup cell has to get right, without paying for
# the slow training loops in 08 and 09: 01 needs nothing but the clone, 05
# imports the package, 07 additionally needs plyfile and the data and out
# directories to exist relative to the working directory.
DEFAULT = ("01", "05", "07")

TIMEOUT = int(os.environ.get("NB_TIMEOUT", "1800"))

args = [a for a in sys.argv[1:] if not a.startswith("-")]
every = "--all" in sys.argv[1:]

# The setup cell runs "pip install -e ." from a clone in a scratch directory
# that this deletes afterwards. In Colab and on a CI runner the environment
# is disposable and that is fine. In a developer's venv it repoints the
# editable install of gsplat_edu at a path that then stops existing, so the
# package stops importing until it is reinstalled.
if not os.environ.get("CI") and "--mutate-my-env" not in sys.argv[1:]:
    sys.exit(
        "refusing to run outside CI.\n"
        "The setup cell pip-installs this package from a temporary clone,\n"
        "which repoints an editable install in whatever environment runs\n"
        "it. Use a throwaway virtualenv and pass --mutate-my-env, or let\n"
        "the colab workflow run this.")

editions = sorted(COLAB.glob("*.ipynb"))
if not editions:
    sys.exit("no editions in %s; run tools/build_colab.py first" % COLAB)

if args:
    picked = [p for p in editions if any(a in p.name for a in args)]
elif every:
    picked = editions
else:
    picked = [p for p in editions if p.name[:2] in DEFAULT]

if not picked:
    sys.exit("no editions matched %s" % args)

skipped = [p.name for p in picked if p.name[:2] in GPU_ONLY]
if skipped and not every:
    picked = [p for p in picked if p.name[:2] not in GPU_ONLY]

ref = os.environ.get("LEARN_GS_REF")
repo = os.environ.get("LEARN_GS_REPO")
print("testing %d edition(s) against ref %s" % (len(picked), ref or "main"))
if repo:
    print("repo override:", repo)

failures = []
for path in picked:
    nb = nbformat.read(path, as_version=4)
    work = pathlib.Path(tempfile.mkdtemp(prefix="colabsim-"))
    started = time.time()
    try:
        # resources path is the kernel's working directory, so the notebook
        # starts where a Colab kernel starts: somewhere with no repo in it.
        NotebookClient(nb, timeout=TIMEOUT, kernel_name="python3",
                       resources={"metadata": {"path": str(work)}}).execute()
    except CellExecutionError as exc:
        failures.append((path.name, str(exc).strip().splitlines()[-1]))
        print("FAIL %s" % path.name, flush=True)
        print(str(exc)[-2000:], flush=True)
        continue
    finally:
        shutil.rmtree(work, ignore_errors=True)

    clone = [c for c in nb.cells if c.cell_type == "code"]
    landed = "".join(o.get("text", "") for c in clone
                     for o in c.get("outputs", []) if o.output_type == "stream")
    if "gsplat_edu from:" not in landed:
        failures.append((path.name, "setup cell did not report gsplat_edu"))
        print("FAIL %s: setup cell produced no gsplat_edu path" % path.name)
        continue
    print("ok   %s (%.0f s)" % (path.name, time.time() - started), flush=True)

for name in skipped:
    print("skip %s (needs a GPU runtime)" % name)

if failures:
    print()
    sys.exit("%d edition(s) failed:\n%s"
             % (len(failures), "\n".join("  %s: %s" % f for f in failures)))
print("\nall checked editions ran from a clean machine")
