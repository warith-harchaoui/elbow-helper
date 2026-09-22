# webapp/ — the static build of the elbow-helper site

The same robust-knee pipeline the library ships, composed into a folder any
static host can serve: the engine runs in the visitor's browser (Pyodide /
WebAssembly), the diagnostic figure is the package's own hand-authored SVG
(`elbow_helper.plotting.render_svg`) and the whole page is bilingual (EN/FR,
client-side toggle).

There is one deployment: **https://deraison.ai/elbow-helper**. It is open —
a visitor lands on a page that explains the tool, clicks once and computes.
No account, no email, nothing to sign.

## Build and deploy

One source, one build, one folder:

```bash
python webapp/build.py --clean     # writes webapp/dist/
```

`index.html` is the landing (from `landing.html`: what the tool does, the
example gallery, one button) and `app.html` is the app itself, one click
away. Upload the CONTENTS of `webapp/dist/` to the web folder by hand:

```bash
sftp> put -r webapp/dist/* /path/to/htdocs/elbow-helper/
```

The upload is manual on purpose. There is no deploy script and no stored
credentials for deraison.ai in this repo — nothing is missing, nothing to
look for.

The bundle is relocatable (relative URLs only), so it works at any mount
point. The Pyodide runtime and numpy load from the jsDelivr CDN on first
visit (browser-cached); everything else ships in the folder.

If the previous, gated deployment is still on the server, uploading over it
changes nothing by itself: its `.htaccess` carries `DirectoryIndex index.php`
and routes the app through `gate/serve.php`. Delete the leftovers in this
order, since every gate endpoint rewrites `.htaccess` from
`gate/htaccess.dist` on each request and would undo an earlier deletion:

1. `gate/` — this is what restores the rest
2. `.htaccess`
3. `index.php`
4. `private/` — download it first: the collected leads and the activity logs
   live there, and only there

## Files

| File                  | Role                                                                        |
| --------------------- | --------------------------------------------------------------------------- |
| `build.py`            | Composes the bundle from `gui.html`, `landing.html` and these files         |
| `gui.html`            | The app page: presets, paste-your-points, options, verdict, figure. Becomes `app.html` |
| `backend-pyodide.js`  | The `window.backend` transport: Pyodide boot, wheel install, JSON bridge    |
| `glue.py`             | The in-Pyodide endpoint logic: presets (the repo's examples, same seeds) and analyze (verdict + SVG, one pipeline run thanks to a memoized `robust_knee`) |
| `os_helper_stub.py`   | Browser stand-in for `os-helper` (whose psutil dependency has no WASM build)|
| `i18n/<lang>.json`    | The GUI string tables (EN/FR), fetched statically at page load              |
| `seo/head-seo.html`   | Deployment head block: canonical, Open Graph/Twitter card, JSON-LD          |
| `seo/icons/`          | Favicon/PWA set generated from `assets/logo.png` (sprezzature-publish)      |
| `seo/make_og_card.py` | Regenerates `seo/og-card.png` (the 1200×630 Open Graph card)                |
| `seo/logo-header.png` | The header logo (56 px, resized from `assets/logo.png`)                     |
| `landing.html`        | The landing (becomes `index.html`): what the tool does, the example gallery, the button that opens the app |
| `landing.css`         | The landing stylesheet                                                      |
| `gate/`               | RETIRED. The professional-email gate the site used to sit behind; `build.py --gate` still assembles it, nothing else references it |

The build also emits `robots.txt`, `sitemap.xml`, `llms.txt`, `llms-full.txt`
and `humans.txt` into `dist/` (sprezzature-publish `site_indexes.py`), plus
the curated Markdown corpus those indexes cite (README/LISEZ-MOI,
EXAMPLES/EXEMPLES, LANDSCAPE/PAYSAGE). Note for the deployer: crawlers only
honor a `robots.txt` served at the DOMAIN root, so reference
`https://deraison.ai/elbow-helper/sitemap.xml` from deraison.ai's own
`/robots.txt` (a `Sitemap:` line) for full effect.

## The retired gate

The site used to ask for a professional email and mail back a magic link. It
no longer does: the tool is free and BSD-licensed, `pip install elbow-helper`
gives the whole engine anyway, and the page's own metadata promises no
account — so the wall filtered nothing but goodwill.

The code is still in `gate/` and `build.py --gate` still assembles it
(`index.php` landing, `.htaccess` routing every app file through
`gate/serve.php`, runtime data under `private/`). Nothing in a normal build
touches it. Local testing of that shape:

```bash
python webapp/build.py --clean --gate
GATE_DEV=1 php -S 127.0.0.1:8323 -t webapp/dist webapp/gate/dev_router.php
# magic links land in webapp/dist/private/outbox.jsonl instead of being mailed
```

`dist/` is generated (and gitignored); rebuild it after any change to the GUI,
the strings, or the library. Rebuild with `--clean` before deploying so no
local test data ships to the host.
