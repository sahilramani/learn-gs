"""Execute notebook sources and write executed .ipynb files.

Usage:
    python tools/build_notebooks.py          # build all
    python tools/build_notebooks.py 06       # build sources matching "06"

Sources are jupytext percent files in notebooks_src/. Output is executed
.ipynb (figures embedded, asserts run) in notebooks/. Never hand-edit the
.ipynb files; edit the source and rebuild. A notebook that raises fails the
build, so the inline asserts are the test suite.

Also enforces the project style rule that sources are pure ASCII.
"""
import os
import sys
import pathlib
import jupytext
import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks_src"
DST = ROOT / "notebooks"

# Per-cell ceiling. The default suits a developer machine, where the slowest
# notebook runs in minutes. Shared CI runners are far slower at the dense
# float64 training in 08 and 09, so CI raises this rather than reading a
# slow runner as a broken notebook.
TIMEOUT = int(os.environ.get("NB_TIMEOUT", "1500"))

pattern = sys.argv[1] if len(sys.argv) > 1 else ""
sources = sorted(p for p in SRC.glob("*.py") if pattern in p.name)
if not sources:
    sys.exit(f"no sources matching '{pattern}' in {SRC}")

for src in sources:
    bad = [(i + 1, ch) for i, line in enumerate(src.read_text().splitlines())
           for ch in line if ord(ch) > 126]
    if bad:
        sys.exit(f"{src.name}: non-ASCII character {bad[0][1]!r} on line {bad[0][0]}")

for src in sources:
    nb = jupytext.read(src)
    print("executing", src.name, flush=True)
    NotebookClient(nb, timeout=TIMEOUT, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    out = DST / (src.stem + ".ipynb")
    nbformat.write(nb, out)
    print("wrote", out.relative_to(ROOT), flush=True)
