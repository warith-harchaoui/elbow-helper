"""Compose the static, SFTP-uploadable elbow-helper web app into ``webapp/dist/``.

The interactive page (``webapp/gui.html``) runs the real package on an
in-browser Python engine: ``backend-pyodide.js`` boots Pyodide, registers a
tiny ``os_helper`` stub (the real one drags in psutil, a C extension absent
from Pyodide, for six trivial symbols), micropip-installs the wheel built
here and drives ``webapp/glue.py`` with JSON strings. Uploading the resulting
``dist/`` folder to any static host (e.g. SFTP to
https://deraison.ai/elbow-helper) yields a fully working app: no process to
run, nothing to maintain server-side.

What lands in ``dist/``:

- ``index.html``          the composed app page (gui.html + the SEO head)
- ``backend-pyodide.js``  the Pyodide transport
- ``py/os_helper_stub.py``browser stand-in for os-helper (see file)
- ``py/glue.py``          the endpoint logic: presets + analyze
- ``py/manifest.json``    the wheel list (just elbow_helper; numpy ships with
                          Pyodide, os_helper is the stub)
- ``wheels/*.whl``        the repo's own wheel
- ``i18n/<lang>.json``    the GUI string tables (webapp/i18n/), so
                          localization needs no Python at page load
- ``static/*``            icons + webmanifest + the Open Graph card + the
                          STATIC example figures the landing shows
- ``robots.txt`` / ``sitemap.xml`` / ``llms.txt`` / ``llms-full.txt`` /
  ``humans.txt``          SEO + GEO indexes (sprezzature-publish site_indexes)
- ``*.md``                the curated Markdown corpus (README/LISEZ-MOI,
                          EXAMPLES/EXEMPLES, LANDSCAPE/PAYSAGE) the indexes
                          cite, served raw
- ``landing.css``         the landing stylesheet, shared by both front doors
- ``index.php``           the PUBLIC landing page (lead-magnet gate): SEO
                          head, static example figures, professional-email
                          form. An open build has no PHP, so it ships the same
                          page as static ``index.html`` (from landing.html)
                          and moves the app to ``app.html``
- ``gate/*.php``, ``gate/free_domains.txt``, ``gate/track.js``, ``.htaccess``
                          the gate itself (see webapp/gate/auth.php):
                          magic-link auth, per-user activity logs,
                          generic-domain blocklist; the .htaccess routes
                          every app file through the gate
- ``private/``            runtime data (secret, leads, logs), pre-created
                          here with its "Require all denied" .htaccess

The gate needs the host to run PHP (deraison.ai does) and to honour
.htaccess rewrites; served without PHP the SAME dist/ degrades to an ungated
static app (index.html still works directly), which is also how the local
python -m http.server smoke tests keep passing.

Run from the repo root with the project env active::

    python webapp/build.py            # writes webapp/dist/
    python webapp/build.py --clean    # rebuild from scratch
    python webapp/build.py --no-gate  # open-access build: no PHP, no email

``--no-gate`` drops the lead-magnet layer (no ``index.php``, no ``gate/``, no
``.htaccess``, no ``private/``) and keeps everything else, the landing page
included: ``index.html`` is then the open landing (``webapp/landing.html`` —
same headline, same example gallery as the gated one, with a button where the
email form was) and the app answers at ``app.html``. The bundle is a plain
static folder any host serves as is, with the tool one click from the landing
and nothing asked of the visitor. That is the shape uploaded to the sev7n
public SFTP (see ``webapp/deploy_sev7n.py``).

The two deployments are two assemblies of the same source, so each gets its
own output folder and neither can overwrite the other::

    python webapp/build.py                       # dist/      → deraison.ai
    python webapp/build.py --no-gate --out \
        webapp/dist-open                         # dist-open/ → sev7n public

Shipping the wrong folder is the one silently public mistake available here:
an ungated bundle uploaded over the gated one serves the app to everybody.
``--out`` is what keeps the two apart on disk; both deploy scripts additionally
check the folder's shape before uploading a byte.

The Pyodide runtime itself is loaded from the jsDelivr CDN at page load (see
``backend-pyodide.js``): only the wheel built here ships in the folder.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Repo layout anchors: this file lives in <repo>/webapp/.
WEBAPP = Path(__file__).resolve().parent
REPO = WEBAPP.parent
# The default output folder; --out redirects it (see main), which is how the
# gated and open-access assemblies coexist without overwriting each other.
DIST = WEBAPP / "dist"

# Where the bundle is deployed; drives the canonical URL, the OG image URL and
# every absolute URL in sitemap.xml / llms.txt (override with --base-url).
BASE_URL = "https://deraison.ai/elbow-helper"

# The sprezzature-publish scripts used for the SEO/GEO artifacts.
_SPREZZATURE = Path.home() / ".claude" / "skills" / "sprezzature-publish" / "scripts"

# The landing page's static gallery: one figure id per card, each shipped as
# figures/<id>_en.svg + figures/<id>_fr.svg in the repo (pre-rendered by the
# examples), published as static/examples/<id>.<lang>.svg.
LANDING_FIGURES = ["kmeans", "queueing_latency", "cache_hit_rate", "bug_discovery_rate"]


def _run(cmd: list[str]) -> None:
    """Run a subprocess loudly; a non-zero exit aborts the build."""
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_wheel() -> list[str]:
    """Build the repo's own wheel into dist/wheels.

    Returns
    -------
    list[str]
        The wheel filenames for ``py/manifest.json`` (a single entry: numpy
        comes from the Pyodide distribution, os_helper is the stub).
    """
    wheels_dir = DIST / "wheels"
    # Start clean: a version bump would otherwise leave the previous wheel
    # behind and break the exactly-one pick below.
    if wheels_dir.exists():
        shutil.rmtree(wheels_dir)
    wheels_dir.mkdir(parents=True)
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "-w",
            str(wheels_dir),
            str(REPO),
        ]
    )

    found = sorted(p.name for p in wheels_dir.glob("elbow_helper-*.whl"))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one elbow_helper-* wheel, found {found}")
    return found


# The activity beacon gui.html loads; it only exists in a gated deployment,
# so an open-access build drops the tag rather than shipping a dead <script>.
TRACK_TAG = '<script src="./gate/track.js" defer></script>\n'


def _seo_head(base_url: str) -> str:
    """The deployment SEO/GEO block (canonical, OG/Twitter card, JSON-LD).

    A deployment concern, so it lives in the build rather than in the page
    sources: a local checkout must never claim the deraison.ai canonical.
    """
    head = (WEBAPP / "seo" / "head-seo.html").read_text(encoding="utf-8")
    return head.replace("{BASE}", base_url.rstrip("/"))


def _inject_head(html: str, head: str, source: str) -> str:
    """Replace the single ``<!--SEO_HEAD-->`` placeholder of *source*."""
    if html.count("<!--SEO_HEAD-->") != 1:
        raise SystemExit(f"{source} SEO_HEAD placeholder missing")
    return html.replace("<!--SEO_HEAD-->", head)


def compose_app(base_url: str, gated: bool = True) -> None:
    """Write the app page: gui.html plus the head its deployment needs.

    Gated, the app IS the site's single page (the landing is index.php, which
    PHP serves instead), so it carries the full SEO head. Open, the landing
    is a real page of its own and the indexed one, so the app answers at
    app.html with a canonical pointing back at it and a noindex: one address
    per piece of content, never two competing for the same search result.
    """
    html = (WEBAPP / "gui.html").read_text(encoding="utf-8")
    if gated:
        html = _inject_head(html, _seo_head(base_url), "gui.html")
        name = "index.html"
    else:
        if html.count(TRACK_TAG) != 1:
            raise SystemExit("gui.html activity-beacon tag missing")
        html = html.replace(TRACK_TAG, "")
        base = base_url.rstrip("/")
        head = (
            f'  <link rel="canonical" href="{base}/" />\n'
            '  <meta name="robots" content="noindex, follow" />\n'
        )
        html = _inject_head(html, head, "gui.html")
        name = "app.html"
    (DIST / name).write_text(html, encoding="utf-8")
    print(f"{name} composed")


def compose_landing(base_url: str) -> None:
    """Write dist/index.html for an open build: landing.html + the SEO head.

    The gated build has no use for this: its landing is index.php, installed
    by :func:`gate_assets` from the same design (see webapp/landing.css).
    """
    html = (WEBAPP / "landing.html").read_text(encoding="utf-8")
    html = _inject_head(html, _seo_head(base_url), "landing.html")
    (DIST / "index.html").write_text(html, encoding="utf-8")
    print("index.html composed (open landing)")


def copy_assets(wheel_names: list[str]) -> None:
    """Copy the transport, glue, strings, icons, figures; write the manifest."""
    (DIST / "py").mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEBAPP / "backend-pyodide.js", DIST / "backend-pyodide.js")
    # The landing stylesheet, shared by the open landing (landing.html) and
    # the gated one (gate/landing.php); both builds serve it at the root.
    shutil.copy2(WEBAPP / "landing.css", DIST / "landing.css")
    shutil.copy2(WEBAPP / "os_helper_stub.py", DIST / "py" / "os_helper_stub.py")
    shutil.copy2(WEBAPP / "glue.py", DIST / "py" / "glue.py")
    (DIST / "py" / "manifest.json").write_text(
        json.dumps({"wheels": wheel_names}, indent=1)
    )

    i18n_dst = DIST / "i18n"
    if i18n_dst.exists():
        shutil.rmtree(i18n_dst)
    shutil.copytree(WEBAPP / "i18n", i18n_dst)

    # Icons: the sprezzature favicon/PWA set generated from assets/logo.png
    # (webapp/seo/icons), the header logo and the Open Graph card.
    static_dst = DIST / "static"
    if static_dst.exists():
        shutil.rmtree(static_dst)
    static_dst.mkdir(parents=True)
    for icon_file in (WEBAPP / "seo" / "icons").iterdir():
        if icon_file.suffix in {".png", ".ico"}:
            shutil.copy2(icon_file, static_dst / icon_file.name)
    shutil.copy2(WEBAPP / "seo" / "og-card.png", static_dst / "og-card.png")
    shutil.copy2(WEBAPP / "seo" / "logo-header.png", static_dst / "logo-header.png")

    # One merged manifest: the sprezzature icon set + the app's identity, with
    # URLs relative to the manifest's own location (dist/static/), so the PWA
    # opens the app root, not the static folder.
    manifest = json.loads(
        (WEBAPP / "seo" / "icons" / "site.webmanifest").read_text(encoding="utf-8")
    )
    manifest["description"] = (
        "Noise-robust knee and elbow detection, entirely in your browser: "
        "a knee with its uncertainty or a frank abstention."
    )
    manifest["start_url"] = "../"
    manifest["scope"] = "../"
    for icon in manifest.get("icons", []):
        icon["src"] = "./" + icon["src"].lstrip("/")
    (static_dst / "site.webmanifest").write_text(json.dumps(manifest, indent=2))

    # The landing's static gallery: the repo's pre-rendered bilingual figures,
    # renamed to the <id>.<lang>.svg convention the page fetches.
    examples_dst = static_dst / "examples"
    examples_dst.mkdir(parents=True)
    for fig_id in LANDING_FIGURES:
        for lang in ("en", "fr"):
            src = REPO / "figures" / f"{fig_id}_{lang}.svg"
            if not src.exists():
                raise SystemExit(f"landing figure missing: {src}")
            shutil.copy2(src, examples_dst / f"{fig_id}.{lang}.svg")
    print("assets copied")


def gate_assets(base_url: str) -> None:
    """Install the lead-magnet gate: landing page, PHP endpoints, .htaccess.

    The interactive app stays exactly as composed by :func:`compose_index`;
    the gate wraps it at the HTTP layer (see ``webapp/gate/auth.php`` for the
    architecture). The landing page reuses the same SEO head block as the app,
    so gating changes what visitors can DO, not what crawlers can read.
    """
    gate_src = WEBAPP / "gate"
    gate_dst = DIST / "gate"
    gate_dst.mkdir(parents=True, exist_ok=True)
    for name in (
        "auth.php",
        "access.php",
        "login.php",
        "serve.php",
        "track.php",
        "track.js",
        "free_domains.txt",
        # The healing template: auth.php restores the root .htaccess from this
        # copy when an SFTP sync drops or overwrites the dotfile.
        "htaccess.dist",
    ):
        shutil.copy2(gate_src / name, gate_dst / name)

    # The public landing (dist/index.php) carries the deployment SEO head.
    seo_head = (WEBAPP / "seo" / "head-seo.html").read_text(encoding="utf-8")
    seo_head = seo_head.replace("{BASE}", base_url.rstrip("/"))
    landing = (gate_src / "landing.php").read_text(encoding="utf-8")
    if landing.count("<!--SEO_HEAD-->") != 1:
        raise SystemExit("landing.php SEO_HEAD placeholder missing")
    (DIST / "index.php").write_text(
        landing.replace("<!--SEO_HEAD-->", seo_head), encoding="utf-8"
    )

    shutil.copy2(gate_src / "htaccess.dist", DIST / ".htaccess")

    # Pre-create the runtime data directory already web-denied, so the deny
    # rule is in place from the very first upload (auth.php re-asserts it).
    private = DIST / "private"
    private.mkdir(exist_ok=True)
    (private / ".htaccess").write_text("Require all denied\n", encoding="utf-8")
    print("gate installed (index.php, gate/, .htaccess, private/)")


def site_indexes(base_url: str) -> None:
    """Emit robots.txt, sitemap.xml, llms.txt, llms-full.txt and humans.txt into dist/.

    Runs sprezzature-publish's stdlib-only ``site_indexes.py`` over a staged
    corpus: the built ``index.html`` (for the sitemap) plus the repo's curated
    Markdown (README/LISEZ-MOI, EXAMPLES/EXEMPLES, LANDSCAPE/PAYSAGE), which
    becomes the ``llms-full.txt`` full-text body that generative engines read.
    """
    script = _SPREZZATURE / "site_indexes.py"
    if not script.exists():
        print(f"note: {script} not found; skipping robots/sitemap/llms indexes")
        return
    corpus = [
        "README.md",
        "LISEZ-MOI.md",
        "EXAMPLES.md",
        "EXEMPLES.md",
        "LANDSCAPE.md",
        "PAYSAGE.md",
    ]
    # The corpus ships in dist/ too, so every URL the sitemap and llms.txt cite
    # actually resolves — and generative engines get raw Markdown to read.
    for name in corpus:
        shutil.copy2(REPO / name, DIST / name)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        shutil.copy2(DIST / "index.html", stage / "index.html")
        for name in corpus:
            shutil.copy2(REPO / name, stage / name)
        # humans.txt credits: the script reads AUTHORS when present.
        (stage / "AUTHORS").write_text(
            "Warith Harchaoui — https://deraison.ai\n", encoding="utf-8"
        )
        _run(
            [
                sys.executable,
                str(script),
                "--root",
                str(stage),
                "--base-url",
                base_url,
                "--out",
                str(stage),
                "--humans",
                "--name",
                "elbow-helper",
                "--summary",
                "elbow-helper locates the knee of a curve of diminishing returns "
                "and validates it: a knee with its uncertainty or a frank "
                "abstention, entirely in the visitor's browser (WebAssembly). "
                "By Warith Harchaoui (https://deraison.ai). Free, BSD 3-Clause.",
            ]
        )
        for name in (
            "robots.txt",
            "sitemap.xml",
            "llms.txt",
            "llms-full.txt",
            "humans.txt",
        ):
            if (stage / name).exists():
                shutil.copy2(stage / name, DIST / name)
                print(f"{name} written")


def main() -> None:
    """Build the bundle end to end; ``--clean`` wipes a previous build first."""
    # Every helper above writes into the module-level DIST, so redirecting the
    # build with --out is a single rebinding rather than an argument threaded
    # through each one.
    global DIST

    parser = argparse.ArgumentParser(
        description="Build the static elbow-helper web app."
    )
    parser.add_argument(
        "--clean", action="store_true", help="remove dist/ before building"
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help="deployment URL for canonical/OG/sitemap (default: %(default)s)",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="open-access build: skip the lead-magnet gate (no PHP, no email)",
    )
    parser.add_argument(
        "--out",
        default=str(DIST),
        help="output folder (default: %(default)s)",
    )
    args = parser.parse_args()
    DIST = Path(args.out).resolve()

    if args.clean and DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True, exist_ok=True)

    wheels = build_wheel()
    compose_app(args.base_url, gated=not args.no_gate)
    copy_assets(wheels)
    if args.no_gate:
        compose_landing(args.base_url)
        print("gate skipped (--no-gate): the landing opens the app directly")
    else:
        gate_assets(args.base_url)
    site_indexes(args.base_url)
    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    shape = "open-access" if args.no_gate else "gated"
    print(
        f"\n{DIST.name}/ ready, {shape} "
        f"({total / 1e6:.1f} MB before the CDN-served Pyodide runtime)."
    )
    print(f"Upload the CONTENTS of {DIST} to the web folder (e.g. /elbow-helper).")


if __name__ == "__main__":
    main()
