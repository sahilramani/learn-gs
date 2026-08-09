"""Generate Colab editions of the notebooks from the same sources.

Usage:
    python tools/build_colab.py           # write colab/*.ipynb
    python tools/build_colab.py --check   # fail if colab/ is out of date

Colab can only open notebooks from GitHub, gist, or Drive; it cannot load
an arbitrary URL. So these editions live in the repo rather than in the
Pages artifact. They are built from notebooks_src/ and carry no outputs,
which keeps them a few tens of KB instead of duplicating the megabytes of
embedded figures in notebooks/.

A Colab VM starts with one .ipynb and nothing else, so each edition gets a
bootstrap cell that clones the repo and installs the package. The
notebooks resolve paths against the working directory (ROOT =
pathlib.Path(".").resolve()) and read cuda/*.cu off disk, so the bootstrap
must chdir into the clone, not just install.

Outputs are byte-stable: cell ids are assigned by position so --check
compares content rather than churn.
"""
import sys
import pathlib
import jupytext
import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks_src"
DST = ROOT / "colab"

REPO_URL = "https://github.com/sahilramani/learn-gs.git"
BRANCH = "main"

# Notebooks whose kernel cells guard on torch.cuda.is_available() and skip
# themselves when it is false. That degrade keeps the CPU build green, but
# on Colab it reads as a notebook that ran and did nothing, so these get a
# preflight cell that stops with instructions instead. metadata.accelerator
# asks Colab for a GPU runtime; the preflight is what does not depend on
# Colab honouring it.
NEEDS_GPU = ("10", "11", "12")

# Notebooks that spend minutes in a training loop on CPU by design. Saying
# so up front stops it reading as a hang.
SLOW = {"08": "600 gradient steps", "09": "800 training iterations"}

INTRO = """\
# {title}

Colab edition of `notebooks_src/{stem}.py` from
[sahilramani/learn-gs]({repo_html}). Outputs are stripped here; run the
cells to produce them. Edits live in this tab only. To keep them, use
File > Save a copy in Drive.

Run the cells in order, or use Runtime > Run all. Setup clones the repo and
installs the package, which the rest of the notebook needs on disk.{gpu}{slow}\
"""

GPU_NOTE = """

This notebook compiles CUDA kernels, so it needs a GPU runtime. The
preflight cell checks for one and stops with instructions if it is
missing. Switching the runtime type restarts the VM and discards the
clone, so run everything again afterwards.\
"""

SLOW_NOTE = """

One cell here runs {what} on the CPU and takes a few minutes. That cost is
the point: it is the measurement the CUDA phase argues against.\
"""

PREFLIGHT = '''\
# Checked before the clone, so a wrong runtime costs seconds rather than a
# full install. The notebook's own kernel cells guard on the same condition
# and skip quietly; here that would look like success.
import shutil

import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "This notebook needs a GPU runtime. Set Runtime > Change runtime "
        "type > GPU, then Runtime > Run all. Colab's free T4 runs every "
        "cell in this notebook.")

nvcc = shutil.which("nvcc") or shutil.which("nvcc", path="/usr/local/cuda/bin")
if nvcc is None:
    raise RuntimeError(
        "A GPU is attached but nvcc is not on PATH, so the kernels cannot "
        "be compiled. Try Runtime > Disconnect and delete runtime, then "
        "reconnect with a GPU runtime.")

print("torch", torch.__version__)
print("gpu  ", torch.cuda.get_device_name(0))
print("nvcc ", nvcc)
'''

BOOTSTRAP = '''\
# Colab setup. Clones the repo and installs the package in editable mode.
# The pip line deliberately omits requirements.txt: that file lists torch
# unpinned, and reinstalling torch here can replace the CUDA-matched build
# Colab ships with one that has no GPU support.
import importlib
import os
import pathlib
import subprocess
import sys

# The environment lookups exist so CI can point this at the branch under
# test and prove the setup still works before it reaches anyone. On Colab
# neither is set and the defaults are what runs.
REPO = os.environ.get("LEARN_GS_REPO", "{repo_url}")
BRANCH = os.environ.get("LEARN_GS_REF", "{branch}")

base = pathlib.Path("/content") if pathlib.Path("/content").is_dir() else pathlib.Path.cwd()
root = base / "learn-gs"
if not root.is_dir():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH,
                    REPO, str(root)], check=True)
os.chdir(root)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".",
                "plyfile", "ninja"], check=True)

# An editable install drops a .pth file that only takes effect when the
# interpreter starts, so the kernel already running this cell cannot see
# gsplat_edu yet. Put the package on the path directly.
src = str(root / "src")
if src not in sys.path:
    sys.path.insert(0, src)
importlib.invalidate_caches()

import gsplat_edu
print("working directory:", os.getcwd())
print("gsplat_edu from:", gsplat_edu.__file__)
'''


def build(src):
    """Return the Colab edition of one source file as a notebook object."""
    stem = src.stem
    nb = jupytext.read(src)

    # Title: first markdown heading in the source, falling back to the stem.
    title = stem
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            head = [ln for ln in cell.source.splitlines() if ln.startswith("# ")]
            if head:
                title = head[0][2:].strip()
                break

    repo_html = REPO_URL[:-len(".git")]
    needs_gpu = stem[:2] in NEEDS_GPU
    gpu = GPU_NOTE if needs_gpu else ""
    slow = SLOW_NOTE.format(what=SLOW[stem[:2]]) if stem[:2] in SLOW else ""
    intro = INTRO.format(title=title, stem=stem, repo_html=repo_html,
                         gpu=gpu, slow=slow)
    setup = BOOTSTRAP.format(repo_url=REPO_URL, branch=BRANCH)

    head = [nbformat.v4.new_markdown_cell(intro)]
    if needs_gpu:
        head.append(nbformat.v4.new_code_cell(PREFLIGHT))
    head.append(nbformat.v4.new_code_cell(setup))
    nb.cells = head + list(nb.cells)

    nb.metadata["kernelspec"] = {"name": "python3",
                                 "display_name": "Python 3"}
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    if stem[:2] in NEEDS_GPU:
        nb.metadata["accelerator"] = "GPU"
    # jupytext's round-trip metadata points at the source path and is noise
    # in a file nobody round-trips.
    nb.metadata.pop("jupytext", None)

    for i, cell in enumerate(nb.cells):
        cell.id = "cell-%03d" % i
        cell.metadata = {}
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return nb


check = "--check" in sys.argv[1:]
sources = sorted(SRC.glob("*.py"))
if not sources:
    sys.exit("no sources in %s" % SRC)

DST.mkdir(exist_ok=True)
stale = []
for src in sources:
    nb = build(src)
    out = DST / (src.stem + ".ipynb")
    text = nbformat.writes(nb, version=4) + "\n"
    if check:
        if not out.exists() or out.read_text() != text:
            stale.append(out.relative_to(ROOT))
    else:
        out.write_text(text)
        print("wrote", out.relative_to(ROOT))

known = {src.stem + ".ipynb" for src in sources}
for extra in sorted(DST.glob("*.ipynb")):
    if extra.name not in known:
        if check:
            stale.append(extra.relative_to(ROOT))
        else:
            extra.unlink()
            print("removed", extra.relative_to(ROOT))

if check and stale:
    sys.exit("colab/ is out of date; run tools/build_colab.py and commit:\n" +
             "\n".join("  " + str(p) for p in stale))
if check:
    print("colab/ is up to date (%d notebooks)" % len(sources))
