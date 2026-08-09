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

Styling tracks the field theme on sahilramani.com, where this publishes as
a project page. The tokens below are copied from that site's
_sass/field/_base.scss; if it restyles, they have to be recopied, since
two Pages builds in two repositories cannot share a stylesheet.
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

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">\n'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Space+Grotesk:wght@300;400;500;600;700&'
         'family=JetBrains+Mono:wght@400;500;600&'
         'family=Instrument+Serif:ital@0;1&display=swap">')

TOKENS = """\
:root{
  --bg:#08080a; --bg2:#0d0d10; --ink:#f0eee4; --ink2:#9a9a92;
  --line:#1c1c20; --line2:#2a2a30;
  --g:#00ff88; --m:#ff0066;
  --sans:'Space Grotesk',sans-serif;
  --mono:'JetBrains Mono',monospace;
  --serif:'Instrument Serif',serif;
}
"""

TOPBAR_CSS = """\
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body{background:var(--bg);color:var(--ink);font-family:var(--sans);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
::selection{background:var(--g);color:#000}
a{color:inherit;text-decoration:none}
.top{display:flex;align-items:center;justify-content:space-between;
  padding:14px 28px;border-bottom:1px solid var(--line);font-family:var(--mono);
  font-size:11px;letter-spacing:0.04em;background:rgba(8,8,10,0.85);
  backdrop-filter:blur(8px);position:sticky;top:0;z-index:10}
.brand{display:flex;align-items:center;gap:10px;font-weight:600;color:var(--ink)}
.brand b{color:var(--g)}
.brand-meta{color:var(--ink2);font-weight:400}
.topnav{display:flex;gap:0}
.topnav a{padding:6px 14px;color:var(--ink2);border-radius:3px;
  transition:color .12s,background .12s}
.topnav a:hover{color:var(--ink)}
.topnav a.active{color:var(--ink);background:var(--line)}
footer{border-top:1px solid var(--line);padding:28px;display:flex;
  flex-wrap:wrap;gap:10px;justify-content:space-between;font-family:var(--mono);
  font-size:11px;color:var(--ink2);letter-spacing:0.05em}
footer a{color:var(--g)}
@media(max-width:820px){
  .top{padding:12px 16px;flex-wrap:wrap;gap:8px}
  .topnav{order:3;width:100%;overflow-x:auto}
  .status{display:none}
}
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
  <div class="status">{status}</div>
</div>
"""

FOOTER = """\
<footer>
  <div>gaussian splatting from scratch &middot;
    <a href="{home}/">sahilramani.com</a></div>
  <div><a href="{repo}">source on github</a> &middot; phases A-C cpu &middot;
    phase D built on an rtx 5080</div>
</footer>
"""

INDEX_CSS = """\
.hero{position:relative;border-bottom:1px solid var(--line);
  padding:88px 60px 56px;background:linear-gradient(180deg,#0a0a0d,#08080a)}
.hero-inner{max-width:1100px;margin:0 auto}
.eyebrow{font-family:var(--mono);font-size:10px;letter-spacing:0.15em;
  text-transform:uppercase;color:var(--g);margin-bottom:24px;
  display:flex;align-items:center;gap:10px}
.eyebrow::before{content:'';width:32px;height:1px;background:var(--g)}
h1{font-size:clamp(40px,7vw,84px);line-height:0.95;letter-spacing:-0.04em;
  font-weight:300;margin-bottom:20px}
h1 em{font-family:var(--serif);font-style:italic;color:var(--g);font-weight:400}
.sub{font-family:var(--serif);font-style:italic;font-size:22px;color:var(--ink2);
  line-height:1.45;max-width:760px}
.sub b{color:var(--ink);font-style:normal;font-family:var(--sans);font-weight:500}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin-top:26px}
.chip{font-family:var(--mono);font-size:10px;letter-spacing:0.08em;
  text-transform:uppercase;color:var(--ink2);padding:6px 12px;
  border:1px solid var(--line2)}
.chip a{color:var(--g)}
.wrap{max-width:1100px;margin:0 auto;padding:56px 60px 88px}
.note{color:var(--ink2);font-size:15px;line-height:1.7;margin-bottom:8px;
  max-width:760px}
.note a{color:var(--g);border-bottom:1px solid rgba(0,255,136,0.25)}
h2{font-family:var(--sans);font-size:26px;font-weight:500;letter-spacing:-0.02em;
  margin:56px 0 20px;padding-top:20px;border-top:1px solid var(--line)}
h2::before{content:'// curriculum';display:block;font-family:var(--mono);font-size:10px;
  font-weight:600;letter-spacing:0.15em;text-transform:uppercase;color:var(--g);
  margin-bottom:8px}
.nb{border:1px solid var(--line2);background:var(--bg2);padding:20px 22px;
  margin:12px 0;transition:border-color .14s}
.nb:hover{border-color:var(--g)}
.nb-id{font-family:var(--mono);font-size:10px;letter-spacing:0.12em;
  text-transform:uppercase;color:var(--g);margin-bottom:8px}
.nb-title{font-size:19px;font-weight:500;letter-spacing:-0.01em;margin-bottom:8px}
.nb p{color:var(--ink2);font-size:14.5px;line-height:1.65}
.nb code{font-family:var(--mono);font-size:0.88em;background:#0a0a0c;
  border:1px solid var(--line2);padding:1px 5px;border-radius:2px;color:var(--ink)}
.runs{margin-top:14px;display:flex;gap:8px;flex-wrap:wrap}
.runs a{font-family:var(--mono);font-size:10px;letter-spacing:0.1em;
  text-transform:uppercase;padding:7px 13px;border:1px solid var(--line2);
  color:var(--ink2);transition:color .12s,border-color .12s,background .12s}
.runs a:hover{color:var(--g);border-color:var(--g)}
.runs a.solid{background:var(--g);border-color:var(--g);color:#000;font-weight:600}
.runs a.solid:hover{color:#000;box-shadow:0 0 16px rgba(0,255,136,0.45)}
@media(max-width:820px){
  .hero{padding:56px 18px 40px}
  .wrap{padding:36px 18px 64px}
}
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
{fonts}
<style>
{tokens}{topbar_css}{index_css}</style>
</head>
<body>
{topbar}
<div class="hero"><div class="hero-inner">
<div class="eyebrow">field_index // gaussian splatting</div>
<h1>Gaussian Splatting <em>from scratch</em></h1>
<p class="sub">Twelve notebooks from <b>project a point</b> to a
<b>CUDA tile rasterizer</b> with a training loop. Nothing is introduced
until a problem in front of you forces it, so every notebook shows the
break before the fix.</p>
<div class="chips">
<span class="chip">12 notebooks</span>
<span class="chip">numpy &rsaquo; torch &rsaquo; cuda</span>
<span class="chip">shipped executed</span>
<span class="chip"><a href="{repo}">github &#8599;</a></span>
</div>
</div></div>
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
/* JupyterLab's dark theme repainted in the field palette. Everything the
   lab template emits is driven by these variables, so overriding them
   recolours markdown, code cells, and outputs in one place. */
:root{
  --jp-layout-color0:#08080a; --jp-layout-color1:#08080a;
  --jp-layout-color2:#0d0d10; --jp-layout-color3:#1c1c20;
  --jp-content-font-color0:#f0eee4; --jp-content-font-color1:#f0eee4;
  --jp-content-font-color2:#9a9a92; --jp-content-font-color3:#9a9a92;
  --jp-border-color0:#1c1c20; --jp-border-color1:#1c1c20;
  --jp-border-color2:#2a2a30; --jp-border-color3:#2a2a30;
  --jp-cell-editor-background:#0a0a0c; --jp-cell-editor-border-color:#2a2a30;
  --jp-brand-color1:#00ff88; --jp-brand-color2:#00ff88;
  --jp-content-font-family:'Space Grotesk',sans-serif;
  --jp-code-font-family:'JetBrains Mono',monospace;
  --jp-content-font-size1:16px; --jp-code-font-size:13px;
}
.jp-RenderedHTMLCommon{font-family:var(--sans);font-size:16px;line-height:1.75}
.jp-RenderedHTMLCommon a{color:var(--g);
  border-bottom:1px solid rgba(0,255,136,0.25)}
.jp-RenderedHTMLCommon h1,.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,.jp-RenderedHTMLCommon h4{
  font-family:var(--sans);letter-spacing:-0.02em;color:var(--ink)}
.jp-RenderedHTMLCommon code{font-family:var(--mono);background:#0a0a0c;
  border:1px solid var(--line2);border-radius:2px;padding:1px 5px}
.jp-RenderedHTMLCommon li::marker{color:var(--g)}
.jp-InputArea-editor{border:1px solid var(--line2)!important;border-radius:6px;
  background:#0a0a0c!important}
.jp-OutputArea-output{background:transparent!important;color:var(--ink)}
/* Figures are matplotlib PNGs with a white canvas. Framing them keeps that
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

for src in sources:
    nb = nbformat.read(src, as_version=4)
    # Without a name in resources every page ends up titled "Notebook".
    body, _ = exporter.from_notebook_node(
        nb, resources={"metadata": {"name": src.stem}})

    extra = (FONTS + "\n<style>\n" + TOKENS + TOPBAR_CSS + NB_CSS + "</style>")
    body, hit = re.subn(r"</head>", lambda m: extra + "\n</head>",
                        body, count=1)
    if not hit:
        sys.exit("no </head> in the rendered %s" % src.stem)

    nav = nb_topbar + NBNAV.format(colab=COLAB, stem=src.stem, repo=REPO_HTML)
    body, hit = re.subn(r"<body[^>]*>", lambda m: m.group(0) + "\n" + nav,
                        body, count=1)
    if not hit:
        sys.exit("no <body> in the rendered %s" % src.stem)

    body, hit = re.subn(r"</body>", lambda m: nb_footer + "</body>",
                        body, count=1)
    if not hit:
        sys.exit("no </body> in the rendered %s" % src.stem)

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

(SITE / "index.html").write_text(PAGE.format(
    fonts=FONTS, tokens=TOKENS, topbar_css=TOPBAR_CSS, index_css=INDEX_CSS,
    topbar=TOPBAR.format(home=HOME, root="index.html", status="12 notebooks"),
    footer=FOOTER.format(home=HOME, repo=REPO_HTML),
    body="\n".join(sections), repo=REPO_HTML))
(SITE / ".nojekyll").write_text("")
print("wrote", (SITE / "index.html").relative_to(ROOT),
      "with %d notebooks" % len(listed))
