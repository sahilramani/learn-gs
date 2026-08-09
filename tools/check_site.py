"""Check the built site before it ships, and the live site after.

Usage:
    python tools/check_site.py                 # check site/
    python tools/check_site.py --live URL      # additionally check a deploy

Three things can break this site without any build failing.

The theme is not in this repository. The pages link the stylesheet on
sahilramani.com and reuse its class names, so a restyle over there can
strip the styling here silently. The contract below names what is being
relied on, and a missing selector fails loudly. A stylesheet that will not
load at all is reported but not failed, since that is usually the network
rather than a change worth blocking on.

The Colab and source links point into GitHub by path. Rename a notebook
source and those become 404s that nothing local would notice.

And a deploy can succeed while serving the wrong thing, which is what
--live is for: it runs after deployment and asserts the page that is
actually being served is the page that was built.
"""
import re
import sys
import time
import pathlib
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"

HOME = "https://www.sahilramani.com"
HOME_CSS = HOME + "/assets/css/main.css"

# Classes and tokens the generated pages take from the site stylesheet
# rather than defining. If one disappears, the pages lose that styling with
# no other symptom, so the check has to be explicit about the list.
THEME_CONTRACT = ["--bg", "--bg2", "--ink", "--ink2", "--line", "--line2",
                  "--g", "--sans", "--mono", "--serif",
                  ".top", ".topnav", ".brand", ".status",
                  ".page-hero", ".page-hero-title", ".page-hero-sub",
                  ".eyebrow", ".chip", "footer"]

HREF = re.compile(r'href="([^"]+)"')

problems = []
notes = []


def check_built():
    index = SITE / "index.html"
    if not index.exists():
        sys.exit("no built site; run tools/build_site.py first")

    pages = sorted(SITE.rglob("*.html"))
    notebooks = sorted((SITE / "notebooks").glob("*.html"))
    if not notebooks:
        problems.append("site/notebooks/ has no rendered notebooks")

    for page in pages:
        rel = page.relative_to(SITE)
        for href in HREF.findall(page.read_text()):
            if href.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = (page.parent / href.split("#")[0]).resolve()
            if not target.exists():
                problems.append("%s links to missing %s" % (rel, href))

    # Every Colab and source link names a file that has to exist in the repo.
    for page in pages:
        text = page.read_text()
        for stem in set(re.findall(r"/colab/([0-9]{2}_[a-z0-9_]+)\.ipynb", text)):
            if not (ROOT / "colab" / (stem + ".ipynb")).exists():
                problems.append("%s links colab/%s.ipynb, which does not exist"
                                % (page.relative_to(SITE), stem))
        for stem in set(re.findall(
                r"/notebooks_src/([0-9]{2}_[a-z0-9_]+)\.py", text)):
            if not (ROOT / "notebooks_src" / (stem + ".py")).exists():
                problems.append(
                    "%s links notebooks_src/%s.py, which does not exist"
                    % (page.relative_to(SITE), stem))

    for page in pages:
        if HOME_CSS not in page.read_text():
            problems.append("%s does not link the site stylesheet"
                            % page.relative_to(SITE))

    check_cross_links(pages, notebooks)
    print("checked %d pages (%d notebooks)" % (len(pages), len(notebooks)))


def check_cross_links(pages, notebooks):
    """Every page reachable from every other, and the order intact.

    Broken-link checking only catches links that point at nothing. It says
    nothing about a link that was never emitted, which is the failure mode
    when a template changes: the page still builds, still validates, and
    quietly becomes a dead end.
    """
    for page in pages:
        text = page.read_text()
        if HOME + "/" not in text:
            problems.append("%s has no link back to sahilramani.com"
                            % page.relative_to(SITE))

    order = [p.stem for p in notebooks]
    for i, page in enumerate(notebooks):
        text = page.read_text()
        rel = page.relative_to(SITE)

        if 'href="../index.html"' not in text:
            problems.append("%s does not link back to the index" % rel)
        if "/colab/%s.ipynb" % page.stem not in text:
            problems.append("%s does not link its Colab edition" % rel)
        if "/notebooks_src/%s.py" % page.stem not in text:
            problems.append("%s does not link its source" % rel)

        # Ends of the curriculum fall back to the index, which is already
        # asserted above, so only interior neighbours are checked by name.
        if i > 0 and 'href="%s.html"' % order[i - 1] not in text:
            problems.append("%s does not link back to %s"
                            % (rel, order[i - 1]))
        if i + 1 < len(order) and 'href="%s.html"' % order[i + 1] not in text:
            problems.append("%s does not link on to %s" % (rel, order[i + 1]))

    index = (SITE / "index.html").read_text()
    for page in notebooks:
        if 'href="notebooks/%s.html"' % page.stem not in index:
            problems.append("the index does not link %s" % page.stem)

    if not problems:
        print("cross-links complete (%d notebooks, both directions)"
              % len(notebooks))


def check_theme():
    try:
        with urllib.request.urlopen(HOME_CSS, timeout=20) as resp:
            css = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        notes.append("could not fetch %s (%s); theme contract unchecked"
                     % (HOME_CSS, exc))
        return
    missing = [sel for sel in THEME_CONTRACT if sel not in css]
    if missing:
        problems.append(
            "the stylesheet on sahilramani.com no longer defines: %s. The "
            "pages here reuse those, so they have lost that styling."
            % ", ".join(missing))
    else:
        print("theme contract holds (%d selectors and tokens)"
              % len(THEME_CONTRACT))


def check_live(url):
    if not url.endswith("/"):
        url += "/"

    # A deploy reports success before every edge has the new bytes, so a
    # single miss here would be a flake rather than a finding.
    status, page, err = None, "", None
    for attempt in range(6):
        if attempt:
            time.sleep(10)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                status = resp.status
                page = resp.read().decode("utf-8", "replace")
            if status == 200:
                break
        except (urllib.error.URLError, TimeoutError) as exc:
            err = exc
    if status != 200:
        problems.append("live site %s did not serve 200 after 6 tries (%s)"
                        % (url, err or status))
        return

    built = (SITE / "index.html").read_text()
    want_cards = built.count('class="nb"')
    got_cards = page.count('class="nb"')
    if got_cards != want_cards:
        problems.append("live index shows %d notebook cards, built shows %d"
                        % (got_cards, want_cards))
    if "<title>Gaussian Splatting from Scratch</title>" not in page:
        problems.append("live index is not the page this build produced")
    if HOME_CSS not in page:
        problems.append("live index does not link the site stylesheet")

    # One rendered notebook, to prove the subdirectory deployed too.
    stem = sorted(p.stem for p in (SITE / "notebooks").glob("*.html"))[0]
    nb_url = url + "notebooks/" + stem + ".html"
    try:
        with urllib.request.urlopen(nb_url, timeout=30) as resp:
            if resp.status != 200:
                problems.append("%s returned %s" % (nb_url, resp.status))
            else:
                print("live notebook ok: %s" % nb_url)
    except (urllib.error.URLError, TimeoutError) as exc:
        problems.append("%s did not load: %s" % (nb_url, exc))

    print("live index ok: %s (%d cards)" % (url, got_cards))


check_built()
check_theme()
if "--live" in sys.argv[1:]:
    i = sys.argv.index("--live")
    if i + 1 >= len(sys.argv):
        sys.exit("--live needs a URL")
    check_live(sys.argv[i + 1])

for note in notes:
    print("note:", note)
if problems:
    sys.exit("\n%d problem(s):\n%s"
             % (len(problems), "\n".join("  " + p for p in problems)))
print("site checks passed")
