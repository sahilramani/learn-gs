"""Apply markdown-only source edits to the built notebooks, no execution.

Usage:
    python tools/update_markdown.py           # all notebooks
    python tools/update_markdown.py 02 04     # only matching sources

tools/build_notebooks.py re-executes, which is right when code changes and
wrong when only prose did. Notebooks 10-12 carry results from an NVIDIA
machine; re-executing them on a CPU box replaces real measurements with the
skip path, so a typo fix in a markdown cell must not go through the
executing build.

This refuses to run unless every code cell is byte-identical between the
source and the built notebook. That is the whole safety argument: if no
code changed, the existing outputs are still the outputs that code
produces, and carrying them across is honest. If any code did change, the
notebook needs a real build and this says so instead of preserving stale
results next to new code.
"""
import sys
import pathlib
import jupytext
import nbformat

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks_src"
DST = ROOT / "notebooks"


def code_cells(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


pattern = sys.argv[1:] or [""]
sources = sorted(p for p in SRC.glob("*.py")
                 if any(pat in p.name for pat in pattern))
if not sources:
    sys.exit("no sources matching %s in %s" % (pattern, SRC))

for src in sources:
    bad = [(i + 1, ch) for i, line in enumerate(src.read_text().splitlines())
           for ch in line if ord(ch) > 126]
    if bad:
        sys.exit("%s: non-ASCII character %r on line %d"
                 % (src.name, bad[0][1], bad[0][0]))

changed, untouched = [], []
for src in sources:
    built = DST / (src.stem + ".ipynb")
    if not built.exists():
        sys.exit("%s has never been built; run tools/build_notebooks.py"
                 % src.name)

    fresh = jupytext.read(src)
    old = nbformat.read(built, as_version=4)

    if code_cells(fresh) != code_cells(old):
        sys.exit(
            "%s: code cells differ from the built notebook, so its outputs "
            "are stale.\nThis tool only carries markdown across. Run\n"
            "  python tools/build_notebooks.py %s\ninstead, on a machine "
            "that can produce that notebook's results." % (src.name, src.stem[:2]))

    # Same code, so the recorded outputs still belong to it. Move them onto
    # the freshly parsed notebook, which carries the new markdown.
    outputs = [(c.outputs, c.execution_count)
               for c in old.cells if c.cell_type == "code"]
    for cell, (outs, count) in zip(
            [c for c in fresh.cells if c.cell_type == "code"], outputs):
        cell.outputs = outs
        cell.execution_count = count
    fresh.metadata = old.metadata

    before = [c.source for c in old.cells if c.cell_type == "markdown"]
    after = [c.source for c in fresh.cells if c.cell_type == "markdown"]
    if before == after:
        untouched.append(src.stem)
        continue

    nbformat.write(fresh, built)
    n = sum(1 for a, b in zip(before, after) if a != b)
    changed.append((src.stem, n, len(after) - len(before)))
    print("updated %s (%d markdown cell%s changed%s)"
          % (built.relative_to(ROOT), n, "" if n == 1 else "s",
             ", %+d cells" % (len(after) - len(before))
             if len(after) != len(before) else ""))

if untouched:
    print("unchanged: %s" % ", ".join(untouched))
if not changed:
    print("nothing to do")
