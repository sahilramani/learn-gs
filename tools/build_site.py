"""Render the executed notebooks into a static site for GitHub Pages.

Usage:
    python tools/build_site.py            # write site/
    python tools/build_site.py --clean    # remove site/ first

Reads the committed notebooks/*.ipynb as they are. Nothing is executed
here: those files ship executed, so the figures and printed numbers on the
site are the ones the build produced, including the GPU results in 10-12
that a CPU machine cannot reproduce. Running tools/build_notebooks.py is
what re-executes them, and that is a separate decision.

The index is generated from the curriculum tables in README.md so the
notebook list has one source of truth. A notebook with no README row is an
error rather than a silent omission.
"""
import html
import re
import shutil
import sys
import pathlib
import nbformat
from nbconvert import HTMLExporter

ROOT = pathlib.Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks"
SITE = ROOT / "site"
OUT = SITE / "notebooks"

REPO_HTML = "https://github.com/sahilramani/learn-gs"
COLAB = "https://colab.research.google.com/github/sahilramani/learn-gs/blob/main"

ROW = re.compile(r"^\|\s*`([0-9]{2}_[a-z0-9_]+)`\s*\|\s*(.*\S)\s*\|\s*$")
HEADING = re.compile(r"^### (Phase [A-Z][^\[]*?)\s*(?:\[done\])?\s*$")

STYLE = """\
:root { color-scheme: light dark; --fg: #16181d; --bg: #ffffff;
        --muted: #5b6270; --line: #e3e6ec; --link: #1a4fd6; --card: #fafbfc; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #e6e8ec; --bg: #111318; --muted: #99a0ad;
          --line: #272b33; --link: #7aa2ff; --card: #171a20; } }
* { box-sizing: border-box; }
body { margin: 0 auto; padding: 3rem 1.25rem 5rem; max-width: 52rem;
       background: var(--bg); color: var(--fg); line-height: 1.6;
       font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                    Helvetica, Arial, sans-serif; }
h1 { font-size: 1.9rem; margin: 0 0 .4rem; letter-spacing: -.01em; }
h2 { font-size: 1.15rem; margin: 2.6rem 0 .9rem; padding-bottom: .4rem;
     border-bottom: 1px solid var(--line); }
p { margin: .7rem 0; }
a { color: var(--link); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
       font-size: .88em; }
.lede { color: var(--muted); }
.nb { border: 1px solid var(--line); border-radius: 10px; padding: .9rem 1.1rem;
      margin: .7rem 0; background: var(--card); }
.nb h3 { margin: 0 0 .35rem; font-size: 1rem; }
.nb h3 a { text-decoration: none; }
.nb p { margin: 0; color: var(--muted); font-size: .93rem; }
.runs { margin-top: .6rem; font-size: .85rem; }
.runs a { display: inline-block; margin-right: .9rem; }
footer { margin-top: 3.5rem; padding-top: 1rem; border-top: 1px solid var(--line);
         color: var(--muted); font-size: .88rem; }
"""

PAGE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaussian Splatting from Scratch</title>
<meta name="description" content="An educational build of a 3D Gaussian
Splatting renderer and trainer, notebook by notebook.">
<style>
{style}</style>
</head>
<body>
<h1>Gaussian Splatting from Scratch</h1>
<p class="lede">An educational build of a 3D Gaussian Splatting renderer and
trainer. Starts from "project a point," ends at a CUDA tile rasterizer with a
training loop.</p>
<p>One rule governs the whole project: nothing is introduced until a problem in
front of us forces it. Every primitive, formula, and optimization exists because
the previous version broke, and each notebook shows the break before the fix.</p>
<p>Every notebook below ships executed, so the figures and measured numbers are
there without running anything. <em>Run in Colab</em> opens a private copy on a
Google VM; it cannot affect this site or the repository.</p>
{body}
<footer>
Source: <a href="{repo}">{repo}</a>. Phases A-C run on CPU; phase D needs an
NVIDIA GPU and the CUDA toolkit, and the shipped phase D notebooks were built
on an RTX 5080.
</footer>
</body>
</html>
"""


def inline(text):
    """Markdown fragment from a README table cell to HTML."""
    text = text.replace(r"\|", "|")
    text = html.escape(text, quote=False)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", text)


def curriculum():
    """[(phase, [(stem, description), ...]), ...] parsed from README.md."""
    phases, phase, rows = [], None, []
    for line in (ROOT / "README.md").read_text().splitlines():
        head = HEADING.match(line)
        if head:
            if phase and rows:
                phases.append((phase, rows))
            phase, rows = head.group(1).strip(), []
            continue
        row = ROW.match(line)
        if row and phase:
            rows.append((row.group(1), row.group(2)))
    if phase and rows:
        phases.append((phase, rows))
    return phases


if "--clean" in sys.argv[1:] and SITE.exists():
    shutil.rmtree(SITE)
OUT.mkdir(parents=True, exist_ok=True)

sources = sorted(NBDIR.glob("*.ipynb"))
if not sources:
    sys.exit("no notebooks in %s; run tools/build_notebooks.py first" % NBDIR)

exporter = HTMLExporter()
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True

BACKLINK = """\
<div style="max-width:1050px;margin:1.2rem auto 0;padding:0 1rem;
font:14px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif">
<a href="../index.html">All notebooks</a> &middot;
<a href="{colab}/colab/{stem}.ipynb">Run this notebook in Colab</a>
</div>
"""

for src in sources:
    nb = nbformat.read(src, as_version=4)
    # Without a name in resources every page ends up titled "Notebook".
    body, _ = exporter.from_notebook_node(
        nb, resources={"metadata": {"name": src.stem}})
    nav = BACKLINK.format(colab=COLAB, stem=src.stem)
    body, hit = re.subn(r"<body[^>]*>", lambda m: m.group(0) + "\n" + nav,
                        body, count=1)
    if not hit:
        sys.exit("no <body> in the rendered %s" % src.stem)
    dst = OUT / (src.stem + ".html")
    dst.write_text(body)
    print("rendered", dst.relative_to(ROOT), "(%d KB)" % (len(body) // 1024))

described = curriculum()
listed = {stem for _, rows in described for stem, _ in rows}
missing = sorted(s.stem for s in sources if s.stem not in listed)
if missing:
    sys.exit("no README curriculum row for: %s" % ", ".join(missing))

sections = []
for phase, rows in described:
    cards = []
    for stem, desc in rows:
        if not (OUT / (stem + ".html")).exists():
            sys.exit("README lists %s but notebooks/%s.ipynb is missing"
                     % (stem, stem))
        cards.append(
            '<div class="nb">\n'
            '<h3><a href="notebooks/{stem}.html">{stem}</a></h3>\n'
            '<p>{desc}</p>\n'
            '<div class="runs">'
            '<a href="notebooks/{stem}.html">Read</a>'
            '<a href="{colab}/colab/{stem}.ipynb">Run in Colab</a>'
            '</div>\n</div>'.format(stem=stem, desc=inline(desc), colab=COLAB))
    sections.append("<h2>%s</h2>\n%s" % (html.escape(phase), "\n".join(cards)))

(SITE / "index.html").write_text(
    PAGE.format(style=STYLE, body="\n".join(sections), repo=REPO_HTML))
(SITE / ".nojekyll").write_text("")
print("wrote", (SITE / "index.html").relative_to(ROOT),
      "with %d notebooks" % len(listed))
