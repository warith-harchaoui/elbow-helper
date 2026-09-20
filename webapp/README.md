# webapp/ — the static build of the elbow-helper site

The same robust-knee pipeline the library ships, composed into a folder any
static host can serve: the engine runs in the visitor's browser (Pyodide /
WebAssembly), the diagnostic figure is the package's own hand-authored SVG
(`elbow_helper.plotting.render_svg`) and the whole page is bilingual (EN/FR,
client-side toggle). The interactive app sits behind a lead-magnet gate
(professional-email magic link, per-user activity logs), the same design as
standpoint's.

## Build and deploy

```bash
python webapp/build.py            # writes webapp/dist/ (~1.2 MB)
# upload the CONTENTS of webapp/dist/ to the target web folder, e.g.:
#   sftp> put -r webapp/dist/* /path/to/htdocs/elbow-helper/
```

The bundle is relocatable (relative URLs only), so it works at any mount
point, e.g. `https://deraison.ai/elbow-helper`. The Pyodide runtime and numpy
load from the jsDelivr CDN on first visit (browser-cached); everything else
ships in the folder.

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
run. Local gated testing:

```bash
GATE_DEV=1 php -S 127.0.0.1:8323 -t webapp/dist webapp/gate/dev_router.php
# magic links land in webapp/dist/private/outbox.jsonl instead of being mailed
```

`dist/` is generated (and gitignored); rebuild it after any change to the GUI,
the strings, or the library. Rebuild with `--clean` before deploying so no
local test data (leads, logs, outbox) ships to the host.
