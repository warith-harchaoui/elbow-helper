// track.js — per-user activity beacon, referenced by the app page. Sends one
// tiny same-origin POST per meaningful action to gate/track.php, which
// attributes it to the authenticated email.
//
// The same dist/ must keep working WITHOUT the gate (python -m http.server
// smoke tests, any PHP-less host): the first beacon doubles as a capability
// probe — if track.php answers like a static file server (404/405/501),
// tracking disables itself for the rest of the session, so the only trace is
// that single probe line in the network log.
(function () {
  "use strict";
  var enabled = null; // null: unknown yet; decided by the first (probe) beacon
  function beacon(event, detail) {
    if (enabled === false) return;
    var payload = JSON.stringify({ event: event, detail: detail || null });
    try {
      if (enabled === null) {
        fetch("gate/track.php", { method: "POST", body: payload, keepalive: true })
          .then(function (res) { enabled = res.status < 400 || res.status === 403; })
          .catch(function () { enabled = false; });
      } else if (navigator.sendBeacon) {
        navigator.sendBeacon("gate/track.php", new Blob([payload], { type: "application/json" }));
      } else {
        fetch("gate/track.php", { method: "POST", body: payload, keepalive: true });
      }
    } catch (e) { enabled = false; /* tracking must never break the app */ }
  }
  document.addEventListener("DOMContentLoaded", function () {
    beacon("app_open", { lang: document.documentElement.lang });
  });
  // Event delegation over the app's stable control ids: the beacon names the
  // action after the control, so the log reads like a session storyboard.
  var WATCHED = "#run, #upload, #dlSvg, #dlPng, #langToggle, #themeToggle, [data-preset]";
  document.addEventListener("click", function (ev) {
    var el = ev.target && ev.target.closest ? ev.target.closest(WATCHED) : null;
    if (!el) return;
    if (el.hasAttribute("data-preset")) {
      beacon("preset", { id: el.getAttribute("data-preset") });
    } else {
      beacon(el.id);
    }
  });
})();
