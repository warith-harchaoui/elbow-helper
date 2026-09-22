# webapp/ — the static build of the elbow-helper site

The same robust-knee pipeline the library ships, composed into a folder any
static host can serve: the engine runs in the visitor's browser (Pyodide /
WebAssembly), the diagnostic figure is the package's own hand-authored SVG
(`elbow_helper.plotting.render_svg`) and the whole page is bilingual (EN/FR,
client-side toggle). The interactive app sits behind a lead-magnet gate
(professional-email magic link, per-user activity logs), the same design as
standpoint's.

## Build and deploy

There are two deployments, so there are two builds of the same source — and
each writes its own folder, so neither can overwrite the other:

```bash
python webapp/build.py --clean                                  # webapp/dist/      gated, for deraison.ai
python webapp/build.py --clean --no-gate --out webapp/dist-open # webapp/dist-open/ open access, for sev7n
```

Then upload the CONTENTS of the matching folder to the target web folder:

```bash
sftp> put -r webapp/dist/* /path/to/htdocs/elbow-helper/
sftp> put webapp/dist/.htaccess /path/to/htdocs/elbow-helper/.htaccess
```

The second line is not redundant: a `*` glob skips dotfiles, so a recursive
put alone leaves `.htaccess` behind — and without it Apache serves the whole
app ungated. Send it explicitly, or upload the folder itself rather than its
glob.

The bundle is relocatable (relative URLs only), so it works at any mount
point, e.g. `https://deraison.ai/elbow-helper`. The Pyodide runtime and numpy
load from the jsDelivr CDN on first visit (browser-cached); everything else
ships in the folder.

## The open-access mirror (sev7n public SFTP)

A second deployment carries the same app with no gate at all: no email asked,
no PHP, no activity beacon. One command builds it and ships it:

```bash
python webapp/deploy_sev7n.py             # build into dist-open/, then upload
python webapp/deploy_sev7n.py --dry-run   # build and report, upload nothing
```

It lands under sev7n's `demo-client-public/` convention, readable by anyone
holding the link:
`https://sftp.s7n.app/private/warith.harchaoui/sev7n-avec-warith/demo-client-public/elbow-helper/`

The host (SFTPGo's HTTPS front) serves static files only, which is exactly
what `--no-gate` produces. The canonical URL still points at
`https://deraison.ai/elbow-helper`, so the mirror never competes with the home
site in a search index. Credentials come from `~/sev7n/settings.yaml` through
`sftp_helper.credentials`, never from the repo (see `~/sev7n/sftp.md`).

Shipping the wrong folder is the one silently public mistake available here:
an ungated bundle uploaded over the gated one serves the app to everybody. Two
things keep that from happening — the separate output folders, and a shape
check in `deploy_sev7n.py`, which refuses to upload a folder carrying gate
files (this host cannot execute PHP, so `index.php` would be served as source
text).

The reverse direction deserves the same care by hand: when uploading to
deraison.ai, check that `index.php`, `.htaccess` and `gate/` are present in
what you send, and never let another project's sync run over that folder —
that is how the gate was lost on 2026-09-20, which is why every gate endpoint
now restores the root `.htaccess` from `gate/htaccess.dist` on each request.
Leave the server's own `private/` alone as well: the HMAC secret, the leads
and the activity logs live there, and only there.

## Files

| File                  | Role                                                                        |
| --------------------- | --------------------------------------------------------------------------- |
| `build.py`            | Composes `dist/` from `gui.html` + these files                              |
| `gui.html`            | The app page: presets, paste-your-points, options, verdict, figure          |
| `backend-pyodide.js`  | The `window.backend` transport: Pyodide boot, wheel install, JSON bridge    |
| `glue.py`             | The in-Pyodide endpoint logic: presets (the repo's examples, same seeds) and analyze (verdict + SVG, one pipeline run thanks to a memoized `robust_knee`) |
| `os_helper_stub.py`   | Browser stand-in for `os-helper` (whose psutil dependency has no WASM build)|
| `i18n/<lang>.json`    | The GUI string tables (EN/FR), fetched statically at page load              |
| `seo/head-seo.html`   | Deployment head block: canonical, Open Graph/Twitter card, JSON-LD          |
| `seo/icons/`          | Favicon/PWA set generated from `assets/logo.png` (sprezzature-publish)      |
| `seo/make_og_card.py` | Regenerates `seo/og-card.png` (the 1200×630 Open Graph card)                |
| `seo/logo-header.png` | The header logo (56 px, resized from `assets/logo.png`)                     |
| `gate/`               | The lead-magnet gate: `auth.php` (HMAC magic links, file-based storage), `landing.php` → `dist/index.php`, `serve.php` (cookie-gated asset serving), `access.php`/`login.php`/`track.php`, `track.js` (activity beacon), `free_domains.txt` (generic-domain blocklist) |

The build also emits `robots.txt`, `sitemap.xml`, `llms.txt`, `llms-full.txt`
and `humans.txt` into `dist/` (sprezzature-publish `site_indexes.py`), plus
the curated Markdown corpus those indexes cite (README/LISEZ-MOI,
EXAMPLES/EXEMPLES, LANDSCAPE/PAYSAGE). Note for the deployer: crawlers only
honor a `robots.txt` served at the DOMAIN root, so reference
`https://deraison.ai/elbow-helper/sitemap.xml` from deraison.ai's own
`/robots.txt` (a `Sitemap:` line) for full effect.

## The gate

`dist/index.php` is the public landing (SEO head, static example figures, the
email form); `.htaccess` routes `index.html`, `backend-pyodide.js`, `py/`,
`wheels/` and `i18n/` through `gate/serve.php`, which requires the signed
`eh_auth` cookie. Runtime data lives in `dist/private/` (secret, leads,
per-user JSONL logs), web-denied twice over. Served without PHP the same
`dist/` degrades to an ungated static app, which is how the local smoke tests
run; `build.py --no-gate` makes that shape explicit by omitting the gate files
outright (that is what the sev7n mirror ships). Local gated testing:

```bash
GATE_DEV=1 php -S 127.0.0.1:8323 -t webapp/dist webapp/gate/dev_router.php
# magic links land in webapp/dist/private/outbox.jsonl instead of being mailed
```

`dist/` is generated (and gitignored); rebuild it after any change to the GUI,
the strings, or the library. Rebuild with `--clean` before deploying so no
local test data (leads, logs, outbox) ships to the host.
