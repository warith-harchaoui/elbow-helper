// backend-pyodide.js — the static build's `window.backend`: every data
// operation of the elbow-helper GUI runs on an in-browser Python engine
// (Pyodide / WebAssembly) instead of a server.
//
// Boot sequence (lazy, single-flight, retried on failure):
//   1. load the pinned Pyodide runtime from the jsDelivr CDN
//   2. loadPackage(["numpy", "micropip"]) — the numeric stack ships with Pyodide
//   3. register the os_helper stub in sys.modules (py/os_helper_stub.py):
//      the real os-helper drags in psutil, a C extension absent from Pyodide,
//      for six trivial symbols elbow_helper actually uses
//   4. micropip-install the repo's own wheel (py/manifest.json lists it),
//      deps=False — numpy is already there, os_helper is the stub
//   5. write py/glue.py into site-packages and import it; from then on every
//      call is one JSON string in, one JSON string out
//
// Two indicators, two jobs: the page's header activity ring (top right) says
// THAT work is in flight — the page wraps these methods with it, and boot()
// lights it for the boot itself — while the fixed bottom-right badge narrates
// WHICH boot step is running.

(function () {
  "use strict";

  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.29.0/full/pyodide.js";
  const PYODIDE_PACKAGES = ["numpy", "micropip"];

  // Bundle-relative URL resolution: the app must work at any mount point
  // (e.g. https://deraison.ai/elbow-helper), so never start a path with "/".
  const rel = (p) => new URL(p, document.baseURI).href;

  let py = null;
  let bootPromise = null;

  function badge(text, done) {
    let el = document.getElementById("engineBadge");
    if (!el) {
      el = document.createElement("div");
      el.id = "engineBadge";
      el.setAttribute("role", "status");
      el.style.cssText =
        "position:fixed;right:.75rem;bottom:.75rem;z-index:50;" +
        "font:12px Roboto Mono,ui-monospace,monospace;padding:.35rem .6rem;" +
        "border-radius:.5rem;background:#171717;color:#fafafa;opacity:.85";
      document.body.appendChild(el);
    }
    el.textContent = text;
    if (done) setTimeout(() => el.remove(), 2500);
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = () => reject(new Error("failed to load " + src));
      document.head.appendChild(s);
    });
  }

  async function bootOnce() {
    badge("Python engine: loading runtime…");
    await loadScript(PYODIDE_URL);
    py = await loadPyodide();
    badge("Python engine: numeric stack…");
    await py.loadPackage(PYODIDE_PACKAGES);
    badge("Python engine: elbow-helper…");
    const manifest = await (await fetch(rel("py/manifest.json"))).json();
    py.globals.set("STUB_SRC", await (await fetch(rel("py/os_helper_stub.py"))).text());
    py.globals.set("GLUE_SRC", await (await fetch(rel("py/glue.py"))).text());
    py.globals.set("WHEEL_URLS", py.toPy(manifest.wheels.map((w) => rel("wheels/" + w))));
    await py.runPythonAsync(`
import pathlib, sysconfig
import micropip
_site = pathlib.Path(sysconfig.get_paths()["purelib"])
# The stub must shadow os_helper BEFORE elbow_helper becomes importable.
(_site / "os_helper.py").write_text(STUB_SRC)
exec(STUB_SRC)  # registers sys.modules["os_helper"] right away
for _url in WHEEL_URLS:
    await micropip.install(_url, deps=False)
(_site / "elbow_helper_glue.py").write_text(GLUE_SRC)
import elbow_helper_glue  # fails loudly here if anything is missing
`);
    window.__engineReady = true;
    badge("Python engine ready", true);
  }

  function boot() {
    if (!bootPromise) {
      // Light the page's header activity ring for the whole boot. The page
      // wraps its backend calls, but this boot starts on its own (below, on
      // DOMContentLoaded), so nothing else would account for the wait it
      // causes — and that wait is the longest one the visitor ever sees.
      if (window.ehBusy) window.ehBusy.start();
      bootPromise = bootOnce()
        .catch((err) => {
          bootPromise = null; // let the next call retry
          badge("Python engine failed: " + (err && err.message ? err.message : err), true);
          throw err;
        })
        .finally(() => {
          if (window.ehBusy) window.ehBusy.end();
        });
    }
    return bootPromise;
  }

  // One JSON envelope per call; Python never leaks proxies to JS.
  async function call(fn, args) {
    await boot();
    py.globals.set("_fn", fn);
    py.globals.set("_arg", JSON.stringify(args || {}));
    const out = JSON.parse(py.runPython("elbow_helper_glue.glue_call(_fn, _arg)"));
    if (out.error) throw new Error(out.error);
    return out.ok;
  }

  window.backend = {
    // Localized GUI strings: a static fetch, no Python engine involved, so
    // the page localizes instantly on load.
    i18n: async (lang) => {
      const res = await fetch(rel("i18n/" + (lang === "fr" ? "fr" : "en") + ".json"));
      if (!res.ok) throw new Error("i18n fetch failed: " + res.status);
      return res.json();
    },
    // The installed version, read from the wheel filename in the manifest
    // (elbow_helper-0.1.7-py3-none-any.whl), so the footer needs no boot.
    version: async () => {
      const res = await fetch(rel("py/manifest.json"));
      if (!res.ok) return "";
      const manifest = await res.json();
      const m = /^elbow_helper-([^-]+)-/.exec((manifest.wheels || [])[0] || "");
      return m ? m[1] : "";
    },
    preset: (id) => call("preset", { id }),
    analyze: (req) => call("analyze", req),
  };

  // Background boot: start downloading the engine while the visitor reads the
  // page, so the first Analyze is usually instant. Failures stay silent here;
  // the first real call will retry and surface the error.
  document.addEventListener("DOMContentLoaded", () => { boot().catch(() => {}); });
})();
