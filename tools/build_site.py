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

Styling comes from sahilramani.com itself. This publishes as a project
page under that domain, so the pages link its stylesheet and reuse its
classes (.top, .page-hero, .chip, footer) rather than restating the
palette. A restyle over there lands here without a rebuild. The only CSS
below is what that sheet has no reason to carry: the notebook list, and
the JupyterLab variables that recolour nbconvert's output.
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
HOME = "https://www.sahilramani.com"

ROW = re.compile(r"^\|\s*`([0-9]{2}_[a-z0-9_]+)`\s*\|\s*(.*\S)\s*\|\s*$")
HEADING = re.compile(r"^### (Phase [A-Z][^\[]*?)\s*(?:\[done\])?\s*$")

HOME_CSS = HOME + "/assets/css/main.css"

HEAD = """\
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=Space+Grotesk:wght@300;400;500;600;700&\
family=JetBrains+Mono:wght@400;500;600&\
family=Instrument+Serif:ital@0;1&display=swap">
<link rel="stylesheet" href="{css}">
<style>
{extra}</style>\
"""

TOPBAR = """\
<div class="top">
  <a class="brand" href="{home}/"><b>&#9673;</b> sahil_ramani
    <span class="brand-meta">/ learn-gs</span></a>
  <div class="topnav">
    <a href="{home}/">home</a>
    <a href="{home}/posts/">writing</a>
    <a href="{home}/#projects">projects</a>
    <a href="{root}" class="active">notebooks</a>
    <a href="{home}/about/">about</a>
  </div>
  <div class="status"><span class="dot"></span> {status}</div>
</div>
"""

FOOTER = """\
<footer>
  <div>gaussian splatting from scratch &middot;
    <a href="{home}/">back to sahilramani.com</a></div>
  <div><a href="{repo}">source on github</a> &middot; phases A-C cpu &middot;
    phase D built on an rtx 5080</div>
</footer>
"""

# Only what the site stylesheet has no reason to carry.
INDEX_CSS = """\
.wrap{max-width:1100px;margin:0 auto;padding:56px 60px 88px}
.note{color:var(--ink2);font-size:16px;line-height:1.7;max-width:760px}
.note em{color:var(--ink);font-style:italic}
.wrap h2{font-family:var(--sans);font-size:26px;font-weight:500;
  letter-spacing:-0.02em;margin:56px 0 20px;padding-top:20px;
  border-top:1px solid var(--line)}
.wrap h2::before{content:'// curriculum';display:block;font-family:var(--mono);
  font-size:10px;font-weight:600;letter-spacing:0.15em;text-transform:uppercase;
  color:var(--g);margin-bottom:8px}
.nb{border:1px solid var(--line2);background:var(--bg2);padding:20px 22px;
  margin:12px 0;transition:border-color .14s}
.nb:hover{border-color:var(--g)}
.nb-id{font-family:var(--mono);font-size:10px;letter-spacing:0.12em;
  text-transform:uppercase;color:var(--g);margin-bottom:8px}
.nb-title{font-size:19px;font-weight:500;letter-spacing:-0.01em;margin-bottom:8px}
.nb p{color:var(--ink2);font-size:14.5px;line-height:1.65}
.nb code{font-family:var(--mono);font-size:0.88em;background:var(--bg);
  border:1px solid var(--line2);padding:1px 5px;border-radius:2px;color:var(--ink)}
.runs{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}
.runs a{font-family:var(--mono);font-size:10px;letter-spacing:0.1em;
  text-transform:uppercase;padding:7px 13px;border:1px solid var(--line2);
  color:var(--ink2);transition:color .12s,border-color .12s,background .12s}
.runs a:hover{color:var(--g);border-color:var(--g)}
.runs a.solid{background:var(--g);border-color:var(--g);color:#000;font-weight:600}
.runs a.solid:hover{color:#000;box-shadow:0 0 16px var(--g)}
@media(max-width:820px){.wrap{padding:36px 18px 64px}}
"""

PAGE = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaussian Splatting from Scratch</title>
<meta name="description" content="An educational build of a 3D Gaussian
Splatting renderer and trainer, notebook by notebook, from projecting a
point to a CUDA tile rasterizer with training.">
{head}
</head>
<body>
{topbar}
<section class="page-hero"><div class="page-hero-inner">
<div class="eyebrow">field_index // gaussian splatting</div>
<h1 class="page-hero-title">Gaussian Splatting <em>from scratch</em></h1>
<p class="page-hero-sub">Twelve notebooks from <b>project a point</b> to a
<b>CUDA tile rasterizer</b> with a training loop. Nothing is introduced
until a problem in front of you forces it, so every notebook shows the
break before the fix.</p>
<div class="page-hero-meta">
<span class="chip">12 notebooks</span>
<span class="chip">numpy &rsaquo; torch &rsaquo; cuda</span>
<span class="chip">shipped executed</span>
<span class="chip"><a href="{repo}">github &#8599;</a></span>
</div>
</div></section>
<div class="wrap">
<p class="note">Every notebook here ships executed, so the figures and
measured numbers are on the page without running anything. <em>Colab</em>
opens a private copy on a Google VM that cannot affect this site or the
repository; notebooks 10-12 ask it for a GPU and stop with instructions if
they do not get one.</p>
{body}
</div>
{footer}
</body>
</html>
"""

# nbconvert's dark lab theme, repainted with the site's own tokens. The lab
# template is driven entirely by --jp-* variables, so overriding them
# recolours markdown, code cells, and outputs in one place.
NB_CSS = """\
body{background:var(--bg);color:var(--ink);font-family:var(--sans)}
.jp-Notebook{background:var(--bg)!important;padding:36px 24px 80px!important;
  max-width:1080px;margin:0 auto}
#notebook-container{background:var(--bg)!important;box-shadow:none!important}
.nbnav{max-width:1080px;margin:0 auto;padding:22px 24px 0;
  font-family:var(--mono);font-size:11px;letter-spacing:0.08em;
  text-transform:uppercase;display:flex;gap:16px;flex-wrap:wrap}
.nbnav a{color:var(--g)}
.nbnav a:hover{opacity:.75}
.nbnav .sep{color:var(--line2)}
:root{
  --jp-layout-color0:var(--bg); --jp-layout-color1:var(--bg);
  --jp-layout-color2:var(--bg2); --jp-layout-color3:var(--line);
  --jp-content-font-color0:var(--ink); --jp-content-font-color1:var(--ink);
  --jp-content-font-color2:var(--ink2); --jp-content-font-color3:var(--ink2);
  --jp-border-color0:var(--line); --jp-border-color1:var(--line);
  --jp-border-color2:var(--line2); --jp-border-color3:var(--line2);
  --jp-cell-editor-background:var(--bg2);
  --jp-cell-editor-border-color:var(--line2);
  --jp-brand-color1:var(--g); --jp-brand-color2:var(--g);
  --jp-content-font-family:var(--sans);
  --jp-code-font-family:var(--mono);
  --jp-content-font-size1:16px; --jp-code-font-size:13px;
}
.jp-RenderedHTMLCommon{font-family:var(--sans);font-size:16px;line-height:1.75}
.jp-RenderedHTMLCommon a{color:var(--g);
  border-bottom:1px solid var(--line2)}
.jp-RenderedHTMLCommon a:hover{border-bottom-color:var(--g)}
.jp-RenderedHTMLCommon h1,.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,.jp-RenderedHTMLCommon h4{
  font-family:var(--sans);letter-spacing:-0.02em;color:var(--ink)}
/* Inline code only. Without the :not(pre), a fenced block's inner <code>
   picks up this border and padding, and since it is an inline element it
   draws one box per wrapped line and shifts the first line right. */
.jp-RenderedHTMLCommon :not(pre) > code{font-family:var(--mono);
  background:var(--bg2);border:1px solid var(--line2);border-radius:2px;
  padding:1px 5px}
.jp-RenderedHTMLCommon pre{font-family:var(--mono);background:#0a0a0c;
  border:1px solid var(--line2);border-radius:6px;padding:16px 18px;
  margin:24px 0;overflow-x:auto;font-size:13.5px;line-height:1.7;
  color:var(--ink)}
.jp-RenderedHTMLCommon pre code{font-family:inherit;font-size:inherit;
  background:none;border:0;padding:0;color:inherit}
.jp-RenderedHTMLCommon li::marker{color:var(--g)}
.jp-RenderedHTMLCommon .MJXc-display{margin:22px 0;overflow-x:auto;
  overflow-y:hidden}
.jp-InputArea-editor{border:1px solid var(--line2)!important;border-radius:6px;
  background:var(--bg2)!important}
.jp-OutputArea-output{background:transparent!important;color:var(--ink)}
/* Figures are matplotlib PNGs on a white canvas. Framing them keeps that
   from reading as a broken image on a near-black page. */
.jp-OutputArea-output img{background:#fff;border:1px solid var(--line2);
  border-radius:4px;padding:6px;max-width:100%;height:auto}
.jp-InputPrompt,.jp-OutputPrompt{color:var(--ink2)}
"""

NBNAV = """\
<div class="nbnav">
<a href="../index.html">&larr; all notebooks</a>
<span class="sep">/</span>
<a href="{colab}/colab/{stem}.ipynb">run in colab &#8599;</a>
<span class="sep">/</span>
<a href="{repo}/blob/main/notebooks_src/{stem}.py">source &#8599;</a>
</div>
"""

# The curriculum is an order, not a menu, so every notebook page carries the
# two steps either side of it. The end of a notebook is where the reader
# decides what to do next, so this repeats after the content.
STEPS = """\
<div class="nbsteps">
<div class="nbstep">{prev}</div>
<div class="nbstep next">{next}</div>
</div>
"""

STEP_LINK = """\
<a href="{stem}.html"><span class="dir">{dir}</span>
<span class="name">{name}</span></a>\
"""

STEP_END = """\
<a href="../index.html"><span class="dir">{dir}</span>
<span class="name">{name}</span></a>\
"""

STEPS_CSS = """\
.nbsteps{max-width:1080px;margin:0 auto;padding:8px 24px 56px;display:grid;
  grid-template-columns:1fr 1fr;gap:12px}
.nbstep a{display:block;height:100%;border:1px solid var(--line2);
  background:var(--bg2);padding:16px 18px;transition:border-color .14s}
.nbstep a:hover{border-color:var(--g)}
.nbstep.next{text-align:right}
.nbstep .dir{display:block;font-family:var(--mono);font-size:10px;
  letter-spacing:0.12em;text-transform:uppercase;color:var(--g);
  margin-bottom:6px}
.nbstep .name{font-size:15px;color:var(--ink)}
@media(max-width:640px){.nbsteps{grid-template-columns:1fr}
  .nbstep.next{text-align:left}}
"""


# Links that leave the domain open in a new tab, matching the convention in
# the blog's _layouts/field-project.html. Done as a pass over the finished
# page rather than in each template, so a link added later cannot miss it,
# and so links inside the notebook prose are covered too.
SAME_SITE = ("https://www.sahilramani.com", "http://www.sahilramani.com",
             "https://sahilramani.com", "http://sahilramani.com")

ANCHOR = re.compile(r"<a\b[^>]*>")


# Two things wrong with nbconvert's MathJax settings for this site.
#
# It turns on automatic linebreaking, which wraps every expression in a
# full-width table cell so long equations can break. That is right for
# display math and wrong for inline math: an inline $z$ takes a line of its
# own and splits the sentence around it. MathJax marks those cells
# display:table-cell !important, so CSS cannot win. Equations here are short
# enough that losing automatic breaking costs nothing, and .MJXc-display
# scrolls instead.
#
# And MathJax matches its x-height to the surrounding text, which lands
# noticeably smaller than the 16px prose: subscripts and the bars in
# fractions get thin enough to squint at. scale is a percentage.
MATH_SCALE = 125

COMMONHTML = re.compile(
    r"CommonHTML:\s*\{\s*linebreaks:\s*\{\s*automatic:\s*true\s*\}\s*\}")

REPLACEMENT = ("CommonHTML: {\n"
               "                    scale: %d,\n"
               "                    linebreaks: { automatic: false }\n"
               "                }" % MATH_SCALE)


def tune_mathjax(page):
    page, hit = COMMONHTML.subn(lambda m: REPLACEMENT, page, count=1)
    if not hit:
        sys.exit("could not find MathJax's CommonHTML block; nbconvert's "
                 "template changed, so inline math will break across lines "
                 "and equations will render at the default size")
    return page


def external_tabs(page):
    def fix(match):
        tag = match.group(0)
        href = re.search(r'href="([^"]*)"', tag)
        if not href or "target=" in tag:
            return tag
        url = href.group(1)
        if not url.startswith(("http://", "https://")):
            return tag
        if url.startswith(SAME_SITE):
            return tag
        return tag[:-1].rstrip() + ' target="_blank" rel="noreferrer">'
    return ANCHOR.sub(fix, page)


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

exporter = HTMLExporter(theme="dark")
exporter.exclude_input_prompt = True
exporter.exclude_output_prompt = True

nb_topbar = TOPBAR.format(home=HOME, root="../index.html", status="notebook")
nb_footer = FOOTER.format(home=HOME, repo=REPO_HTML)

order = [p.stem for p in sources]


def step(stem, delta):
    """The prev or next card for one notebook, or the index at either end."""
    i = order.index(stem) + delta
    label = "previous" if delta < 0 else "next"
    if 0 <= i < len(order):
        neighbour = order[i]
        num, name = neighbour.split("_", 1)
        return STEP_LINK.format(stem=neighbour, dir=label,
                                name="%s. %s" % (num, name.replace("_", " ")))
    end = "start of the curriculum" if delta < 0 else "end of the curriculum"
    return STEP_END.format(dir=label, name=end)


for src in sources:
    nb = nbformat.read(src, as_version=4)
    # Without a name in resources every page ends up titled "Notebook".
    body, _ = exporter.from_notebook_node(
        nb, resources={"metadata": {"name": src.stem}})

    head = HEAD.format(css=HOME_CSS, extra=NB_CSS + STEPS_CSS)
    body, hit = re.subn(r"</head>", lambda m: head + "\n</head>",
                        body, count=1)
    if not hit:
        sys.exit("no </head> in the rendered %s" % src.stem)

    steps = STEPS.format(prev=step(src.stem, -1), next=step(src.stem, 1))
    nav = (nb_topbar + NBNAV.format(colab=COLAB, stem=src.stem, repo=REPO_HTML)
           + steps)
    body, hit = re.subn(r"<body[^>]*>", lambda m: m.group(0) + "\n" + nav,
                        body, count=1)
    if not hit:
        sys.exit("no <body> in the rendered %s" % src.stem)

    body, hit = re.subn(r"</body>", lambda m: steps + nb_footer + "</body>",
                        body, count=1)
    if not hit:
        sys.exit("no </body> in the rendered %s" % src.stem)

    dst = OUT / (src.stem + ".html")
    dst.write_text(external_tabs(tune_mathjax(body)))
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
        num, name = stem.split("_", 1)
        cards.append(
            '<div class="nb">\n'
            '<div class="nb-id">notebook {num}</div>\n'
            '<div class="nb-title">{name}</div>\n'
            '<p>{desc}</p>\n'
            '<div class="runs">'
            '<a class="solid" href="notebooks/{stem}.html">Read</a>'
            '<a href="{colab}/colab/{stem}.ipynb">Colab</a>'
            '<a href="{repo}/blob/main/notebooks_src/{stem}.py">Source</a>'
            '</div>\n</div>'.format(
                num=num, name=name.replace("_", " "), desc=inline(desc),
                stem=stem, colab=COLAB, repo=REPO_HTML))
    sections.append("<h2>%s</h2>\n%s" % (html.escape(phase), "\n".join(cards)))

(SITE / "index.html").write_text(external_tabs(PAGE.format(
    head=HEAD.format(css=HOME_CSS, extra=INDEX_CSS),
    topbar=TOPBAR.format(home=HOME, root="index.html", status="12 notebooks"),
    footer=FOOTER.format(home=HOME, repo=REPO_HTML),
    body="\n".join(sections), repo=REPO_HTML)))
(SITE / ".nojekyll").write_text("")
print("wrote", (SITE / "index.html").relative_to(ROOT),
      "with %d notebooks" % len(listed))
