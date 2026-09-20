<?php
// landing.php — the public face of the gated deployment (becomes dist/index.php).
//
// Carries the full SEO/GEO head plus STATIC, non-editable example figures, so
// crawlers and generative engines keep seeing what elbow-helper does even
// though the interactive app sits behind the professional-email gate. A
// returning visitor with a valid cookie skips this page entirely.
//
// Fully bilingual, with the SAME top-right controls as the app itself (flag
// language toggle, sun/moon theme toggle, GitHub link in a sticky header) and
// the SAME localStorage keys (eh-lang, eh-theme): the language and theme a
// visitor picks here follow them into the app after they sign in.

declare(strict_types=1);

require __DIR__ . '/gate/auth.php';

if (current_email() !== null) {
    header('Location: index.html');
    exit;
}

// Server-side initial language for the first paint and no-JS visitors:
// explicit ?lang= wins, else the browser's Accept-Language. localStorage
// (the durable choice, shared with the app) takes over before first paint.
$lang = (string) ($_GET['lang'] ?? '');
if ($lang !== 'fr' && $lang !== 'en') {
    $accept = strtolower((string) ($_SERVER['HTTP_ACCEPT_LANGUAGE'] ?? ''));
    $lang = str_starts_with($accept, 'fr') ? 'fr' : 'en';
}
$title = $lang === 'fr' ? 'elbow-helper : le coude est-il réel ?' : 'elbow-helper: is that elbow real?';
$meta = $lang === 'fr'
    ? "Détection de coude robuste au bruit, dans votre navigateur : un coude assorti de son incertitude ou une abstention franche. Rien n'est envoyé."
    : 'Noise-robust knee and elbow detection in your browser: a knee with its uncertainty or a frank abstention. Nothing is uploaded.';
$prefill = (string) ($_GET['email'] ?? '');
$flag = (string) ($_GET['link'] ?? ($_GET['login'] ?? ''));
?>
<!doctype html>
<html lang="<?= $lang ?>">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="description" content="<?= htmlspecialchars($meta, ENT_QUOTES) ?>" />
  <title><?= htmlspecialchars($title, ENT_QUOTES) ?></title>
<!--SEO_HEAD-->
  <link rel="icon" href="./static/favicon.ico" sizes="any" />
  <link rel="icon" type="image/png" sizes="32x32" href="./static/favicon-32.png" />
  <link rel="icon" type="image/png" sizes="16x16" href="./static/favicon-16.png" />
  <link rel="apple-touch-icon" sizes="180x180" href="./static/apple-touch-icon.png" />
  <link rel="manifest" href="./static/site.webmanifest" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;800&family=Roboto+Serif:wght@400;500;600&family=Roboto+Mono&display=swap" rel="stylesheet" />
  <script>
    // Pre-paint: apply the persisted theme and language (the app's own
    // localStorage keys) before any CSS resolves, so there is no flash.
    (function () {
      var theme = localStorage.getItem("eh-theme");
      var dark = theme ? theme === "dark"
        : window.matchMedia("(prefers-color-scheme: dark)").matches;
      if (dark) document.documentElement.classList.add("dark");
      var lang = localStorage.getItem("eh-lang");
      if (lang === "fr" || lang === "en") document.documentElement.lang = lang;
    })();
  </script>
  <style>
    /* Same typographic system as the app: Roboto for chrome, Roboto Serif for
       the headline, Roboto Mono for the eyebrow. Good Colors blue as accent.
       Dark mode is a .dark class on <html> (not a media query), exactly like
       the app, so the shared theme toggle rules both pages. */
    :root { --accent: #007AFF; --accent-soft: #CCE4FF; }
    * { box-sizing: border-box; margin: 0; }
    body {
      font-family: Roboto, system-ui, -apple-system, Helvetica, Arial, sans-serif;
      background: #f5f5f4; color: #1c1917; line-height: 1.55;
    }
    .headline { font-family: "Roboto Serif", Georgia, serif; }
    .eyebrow { font-family: "Roboto Mono", ui-monospace, SFMono-Regular, monospace;
      font-size: .8rem; letter-spacing: .08em; color: #78716c; }
    /* Sticky header, mirrored from the app: brand left, controls top right. */
    .site-header { position: sticky; top: 0; z-index: 40;
      -webkit-backdrop-filter: blur(8px); backdrop-filter: blur(8px);
      background: rgba(245,245,244,.85); border-bottom: 1px solid #e7e5e4; }
    .dark .site-header { background: rgba(11,11,12,.8); border-bottom-color: #262626; }
    .nav { max-width: 60rem; margin: 0 auto; padding: .55rem 1.25rem;
      display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
    .brand { display: flex; align-items: center; gap: .5rem; font-weight: 600;
      letter-spacing: -.01em; }
    .brand img { width: 1.75rem; height: 1.75rem; border-radius: .25rem; }
    .controls { display: flex; align-items: center; gap: .25rem; }
    .gh-link { display: inline-flex; align-items: center; gap: .35rem;
      font-size: .875rem; color: #44403c; text-decoration: none;
      padding: .35rem .6rem; border-radius: .5rem; }
    .gh-link:hover { background: #e7e5e4; }
    .dark .gh-link { color: #d4d4d4; }
    .dark .gh-link:hover { background: #262626; }
    .toggle-btn { font-size: 1.15rem; line-height: 1; width: 2.25rem; height: 2.25rem;
      display: inline-flex; align-items: center; justify-content: center;
      border: 0; background: transparent; border-radius: .5rem; cursor: pointer; }
    .toggle-btn:hover { background: #e7e5e4; }
    .dark .toggle-btn:hover { background: #262626; }
    .toggle-btn:focus-visible, .gh-link:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    main { max-width: 60rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
    h1 { font-size: clamp(2.2rem, 6vw, 3.4rem); font-weight: 600; margin: .3rem 0 .8rem; }
    .guide { color: #44403c; max-width: 44rem; }
    section.card { background: #fff; border: 1px solid #e7e5e4; border-radius: 1rem;
      padding: 1.5rem; margin-top: 2rem; }
    .gallery { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
      gap: 1rem; margin-top: 1rem; }
    figure { border: 1px solid #e7e5e4; border-radius: .75rem; overflow: hidden; background: #fff; }
    figure img { display: block; width: 100%; height: auto; }
    figcaption { padding: .5rem .8rem; font-size: .85rem; font-weight: 500;
      border-top: 3px solid var(--tint, var(--accent)); }
    figure button { display: block; width: 100%; padding: 0; border: 0;
      background: transparent; cursor: zoom-in; }
    figure button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    #lightbox[hidden] { display: none; }
    #lightbox { position: fixed; inset: 0; z-index: 60; cursor: zoom-out;
      background: rgba(12,10,9,.88); display: flex;
      align-items: center; justify-content: center; padding: 2rem; }
    #lightbox img { max-width: 100%; max-height: 100%; background: #fff;
      border-radius: .75rem; }
    form { display: grid; gap: .9rem; max-width: 30rem; margin-top: 1rem; }
    label { font-weight: 500; }
    input[type=email] { padding: .65rem .8rem; border: 1px solid #d6d3d1;
      border-radius: .6rem; font-size: 1rem; width: 100%; }
    input[type=email]:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
    .consent { display: flex; gap: .55rem; align-items: flex-start; font-size: .85rem;
      color: #57534e; font-weight: 400; }
    .consent input { margin-top: .25rem; }
    button[type=submit] { background: #1c1917; color: #fff; border: 0;
      border-radius: .6rem; padding: .7rem 1.2rem; font-size: 1rem; font-weight: 600;
      cursor: pointer; justify-self: start; }
    button[type=submit]:hover { background: #292524; }
    button[type=submit]:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
    .msg { border-radius: .6rem; padding: .7rem .9rem; font-size: .95rem; display: none; }
    .msg.ok { background: #D4F5D9; color: #14532d; display: block; }
    .msg.err { background: #FFD8D6; color: #7f1d1d; display: block; }
    .hp { position: absolute; left: -9999px; }
    footer { border-top: 1px solid #e7e5e4; margin-top: 3rem; padding-top: 1rem;
      font-size: .85rem; color: #78716c; }
    footer a { color: inherit; }
    .dark body, .dark { background: #0c0a09; }
    .dark body { color: #e7e5e4; }
    .dark .guide { color: #d6d3d1; }
    .dark section.card, .dark figure { background: #1c1917; border-color: #292524; }
    .dark input[type=email] { background: #0c0a09; border-color: #44403c; color: #e7e5e4; }
    .dark button[type=submit] { background: #e7e5e4; color: #1c1917; }
    .dark button[type=submit]:hover { background: #fff; }
    .dark .msg.ok { background: #14532d; color: #D4F5D9; }
    .dark .msg.err { background: #7f1d1d; color: #FFD8D6; }
    /* The muted grays readable on white fail on the dark cards (consent,
       notes, eyebrow ~2.8:1); lift them to a legible stone. */
    .dark .consent { color: #a8a29e; }
    .dark .note, .dark .eyebrow, .dark footer { color: #a8a29e; }
    .dark .gh-link { color: #d4d4d4; }
    .dark footer { border-color: #292524; }
    /* The example SVGs carry their own prefers-color-scheme block; on this
       page the theme is a class, so keep the previews on white cards in both
       themes (the figures are self-contained documents, light by design). */
  </style>
</head>
<body>
<header class="site-header">
  <nav class="nav" aria-label="Primary">
    <div class="brand">
      <img src="./static/logo-header.png" alt="elbow-helper logo" width="28" height="28" />
      <span>elbow-helper</span>
    </div>
    <!-- Same top-right controls as the app: GitHub, language flag, theme. -->
    <div class="controls">
      <a id="ghLink" class="gh-link" href="https://github.com/warith-harchaoui/elbow-helper"
         target="_blank" rel="noopener" data-i18n="github">⭐️ on GitHub</a>
      <button id="langToggle" class="toggle-btn" type="button" aria-label="Switch language">🇬🇧</button>
      <button id="themeToggle" class="toggle-btn" type="button" aria-label="Switch theme">🌛</button>
    </div>
  </nav>
</header>
<main>
  <p class="eyebrow">elbow-helper</p>
  <h1 class="headline" data-i18n="title"></h1>
  <p class="guide" data-i18n="tagline"></p>

  <section class="card" aria-labelledby="access-h">
    <h2 id="access-h" data-i18n="form_title"></h2>
    <p data-i18n="form_sub"></p>
    <div class="msg" id="formMsg" role="status" aria-live="polite"></div>
    <form id="accessForm" novalidate>
      <div>
        <label for="email" data-i18n="email_label"></label>
        <input type="email" id="email" name="email" required autocomplete="email"
               value="<?= htmlspecialchars($prefill, ENT_QUOTES) ?>" />
      </div>
      <!-- Honeypot: humans never see it, naive bots fill it. -->
      <div class="hp" aria-hidden="true">
        <label for="website">Website</label>
        <input type="text" id="website" name="website" tabindex="-1" autocomplete="off" />
      </div>
      <label class="consent">
        <input type="checkbox" id="consent" name="consent" required />
        <span data-i18n="consent"></span>
      </label>
      <button type="submit" data-i18n="submit"></button>
    </form>
  </section>

  <section class="card" aria-labelledby="examples-h">
    <h2 id="examples-h" data-i18n="examples_title"></h2>
    <div class="gallery" id="gallery"></div>
  </section>

  <!-- Fullscreen viewer for the example figures: click a card to open, click
       anywhere / Escape to close. A plain overlay, no dialog element (no
       focus trap needed for a single close action). -->
  <div id="lightbox" hidden>
    <img id="lightboxImg" src="" alt="" />
  </div>

  <footer>
    BSD 3-Clause License ·
    <a href="https://deraison.ai" rel="author">Warith Harchaoui</a>
  </footer>
</main>

<script>
"use strict";
// Bilingual chrome, same mechanism (and same localStorage keys) as the app:
// one strings table per language, swapped in place by the flag toggle.
const STRINGS = {
  en: {
    doc_title: "elbow-helper: is that elbow real?",
    meta_description: "Noise-robust knee and elbow detection in your browser: a knee with its uncertainty or a frank abstention. Nothing is uploaded.",
    github: "⭐️ on GitHub",
    lang_aria: "Switch language",
    theme_to_dark_aria: "Switch to dark theme",
    theme_to_light_aria: "Switch to light theme",
    title: "Is that elbow real?",
    tagline: "Detect where the curve bends",
    form_title: "Get access with your work email",
    form_sub: "We send you a sign-in link. One click and the interactive app is yours for 30 days on this device.",
    email_label: "Work email (no generic addresses such as gmail.com)",
    consent: "I agree that my email address is stored. Removal on request: warith@deraison.ai.",
    submit: "Email me the access link",
    examples_title: "What you will get",
    zoom_aria: "View full screen",
    msg_sent: "Link sent. Check your inbox (and spam folder).",
    msg_generic: "Please use your professional email address, not a generic one (gmail, yahoo, orange…).",
    msg_invalid: "That email address does not look valid.",
    msg_nodomain: "That domain does not seem to receive email.",
    msg_consent: "Please tick the consent box.",
    msg_ratelimit: "Too many requests; please try again later.",
    msg_sendfail: "The email could not be sent. Please try again, or write to warith@deraison.ai.",
    msg_link_expired: "That access link has expired. Enter your email to receive a fresh one.",
    msg_link_invalid: "That access link is not valid. Enter your email to receive a new one.",
    msg_login_required: "Please enter your work email to access the app.",
    examples: {
      kmeans: "The k-means elbow",
      queueing_latency: "Queueing latency",
      cache_hit_rate: "Cache sizing",
      bug_discovery_rate: "When to stop testing"
    }
  },
  fr: {
    doc_title: "elbow-helper : le coude est-il réel ?",
    meta_description: "Détection de coude robuste au bruit, dans votre navigateur : un coude assorti de son incertitude ou une abstention franche. Rien n'est envoyé.",
    github: "⭐️ sur GitHub",
    lang_aria: "Changer de langue",
    theme_to_dark_aria: "Passer au thème sombre",
    theme_to_light_aria: "Passer au thème clair",
    title: "Le coude est-il réel ?",
    tagline: "Détecter là où la courbe plie",
    form_title: "Accédez avec votre email professionnel",
    form_sub: "Nous vous envoyons un lien de connexion. Un clic et l'application interactive est à vous pour 30 jours sur cet appareil.",
    email_label: "Email professionnel (pas d'adresse générique type gmail.com)",
    consent: "J'accepte que mon adresse email soit enregistrée. Suppression sur demande : warith@deraison.ai.",
    submit: "Recevoir le lien d'accès",
    examples_title: "Ce que vous obtiendrez",
    zoom_aria: "Voir en plein écran",
    msg_sent: "Lien envoyé. Vérifiez votre boîte (et les spams).",
    msg_generic: "Merci d'utiliser votre email professionnel, pas une adresse générique (gmail, yahoo, orange…).",
    msg_invalid: "Cette adresse email ne semble pas valide.",
    msg_nodomain: "Ce domaine ne semble pas recevoir d'emails.",
    msg_consent: "Merci de cocher la case de consentement.",
    msg_ratelimit: "Trop de demandes ; réessayez plus tard.",
    msg_sendfail: "L'email n'a pas pu partir. Réessayez, ou écrivez à warith@deraison.ai.",
    msg_link_expired: "Ce lien d'accès a expiré. Saisissez votre email pour en recevoir un nouveau.",
    msg_link_invalid: "Ce lien d'accès n'est pas valide. Saisissez votre email pour en recevoir un nouveau.",
    msg_login_required: "Saisissez votre email professionnel pour accéder à l'application.",
    examples: {
      kmeans: "Le coude de k-means",
      queueing_latency: "Latence de file d'attente",
      cache_hit_rate: "Dimensionner un cache",
      bug_discovery_rate: "Quand arrêter les tests"
    }
  }
};
// One Good Colors tint per dataset, mirroring the app's example chips.
const TINTS = {
  kmeans: "#28CD41",
  queueing_latency: "#AF52DE",
  cache_hit_rate: "#5AC8FA",
  bug_discovery_rate: "#FF2D55"
};

// Language and theme: initialized by the pre-paint script in <head> from the
// app's own localStorage keys; the toggles below keep both pages in sync.
let lang = document.documentElement.lang === "fr" ? "fr" : "en";

function isDark() { return document.documentElement.classList.contains("dark"); }

function apply() {
  const t = STRINGS[lang];
  document.documentElement.lang = lang;
  document.title = t.doc_title;
  document.querySelector('meta[name="description"]').setAttribute("content", t.meta_description);
  document.querySelectorAll("[data-i18n]").forEach(el => {
    el.textContent = t[el.dataset.i18n];
  });
  // Same convention as the app: the flag shows the CURRENT language, the sun/
  // moon shows the theme you would switch to, and the aria labels say so.
  const langBtn = document.getElementById("langToggle");
  langBtn.textContent = lang === "fr" ? "🇫🇷" : "🇬🇧";
  langBtn.setAttribute("aria-label", t.lang_aria);
  const themeBtn = document.getElementById("themeToggle");
  themeBtn.textContent = isDark() ? "🌞" : "🌛";
  themeBtn.setAttribute("aria-label", isDark() ? t.theme_to_light_aria : t.theme_to_dark_aria);
  const gallery = document.getElementById("gallery");
  gallery.innerHTML = "";
  for (const id of Object.keys(t.examples)) {
    const fig = document.createElement("figure");
    fig.style.setProperty("--tint", TINTS[id]);
    const img = document.createElement("img");
    img.src = `static/examples/${id}.${lang}.svg`;
    img.alt = t.examples[id];
    img.loading = "lazy";
    // A missing render hides its card rather than showing a broken image.
    img.onerror = () => fig.remove();
    const zoom = document.createElement("button");
    zoom.type = "button";
    zoom.setAttribute("aria-label", t.examples[id] + " — " + t.zoom_aria);
    zoom.addEventListener("click", () => openLightbox(img.src, img.alt));
    zoom.append(img);
    const cap = document.createElement("figcaption");
    cap.textContent = t.examples[id];
    fig.append(zoom, cap);
    gallery.append(fig);
  }
}

// Fullscreen viewer: any click closes it, Escape too.
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightboxImg");
function openLightbox(src, alt) {
  lightboxImg.src = src;
  lightboxImg.alt = alt;
  lightbox.hidden = false;
}
function closeLightbox() {
  lightbox.hidden = true;
  lightboxImg.src = "";
}
lightbox.addEventListener("click", closeLightbox);
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && !lightbox.hidden) closeLightbox();
});

function showMsg(kind, text) {
  const box = document.getElementById("formMsg");
  box.className = "msg " + kind;
  box.textContent = text;
}

document.getElementById("langToggle").addEventListener("click", () => {
  lang = lang === "fr" ? "en" : "fr";
  localStorage.setItem("eh-lang", lang);
  apply();
});

document.getElementById("themeToggle").addEventListener("click", () => {
  document.documentElement.classList.toggle("dark");
  localStorage.setItem("eh-theme", isDark() ? "dark" : "light");
  apply();
});

document.getElementById("accessForm").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const t = STRINGS[lang];
  const form = ev.target;
  if (!form.consent.checked) { showMsg("err", t.msg_consent); return; }
  const data = new FormData(form);
  data.set("lang", lang);
  try {
    const res = await fetch("gate/access.php", { method: "POST", body: data });
    const out = await res.json();
    if (out.ok) {
      showMsg("ok", t.msg_sent);
      form.reset();
    } else {
      showMsg("err", t["msg_" + out.error] || t.msg_sendfail);
    }
  } catch (e) {
    showMsg("err", t.msg_sendfail);
  }
});

// Flags carried back by login.php / serve.php redirects.
const FLAG = <?= json_encode($flag) ?>;
if (FLAG === "expired") showMsg("err", STRINGS[lang].msg_link_expired);
else if (FLAG === "invalid") showMsg("err", STRINGS[lang].msg_link_invalid);
else if (FLAG === "required") showMsg("err", STRINGS[lang].msg_login_required);

apply();
</script>
</body>
</html>
