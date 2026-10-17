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
  <!-- The look is shared with the open landing (webapp/landing.html);
       build.py copies webapp/landing.css next to this page. -->
  <link rel="stylesheet" href="./landing.css" />
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
      <button id="langToggle" class="toggle-btn" type="button" aria-label="Switch language">🇫🇷</button>
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
    consent: "I agree that my email address is stored. Removal on request: warithmetics@deraison.ai.",
    submit: "Email me the access link",
    examples_title: "What you will get",
    zoom_aria: "View full screen",
    msg_sent: "Link sent. Check your inbox (and spam folder).",
    msg_generic: "Please use your professional email address, not a generic one (gmail, yahoo, orange…).",
    msg_invalid: "That email address does not look valid.",
    msg_nodomain: "That domain does not seem to receive email.",
    msg_consent: "Please tick the consent box.",
    msg_ratelimit: "Too many requests; please try again later.",
    msg_sendfail: "The email could not be sent. Please try again, or write to warithmetics@deraison.ai.",
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
    consent: "J'accepte que mon adresse email soit enregistrée. Suppression sur demande : warithmetics@deraison.ai.",
    submit: "Recevoir le lien d'accès",
    examples_title: "Ce que vous obtiendrez",
    zoom_aria: "Voir en plein écran",
    msg_sent: "Lien envoyé. Vérifiez votre boîte (et les spams).",
    msg_generic: "Merci d'utiliser votre email professionnel, pas une adresse générique (gmail, yahoo, orange…).",
    msg_invalid: "Cette adresse email ne semble pas valide.",
    msg_nodomain: "Ce domaine ne semble pas recevoir d'emails.",
    msg_consent: "Merci de cocher la case de consentement.",
    msg_ratelimit: "Trop de demandes ; réessayez plus tard.",
    msg_sendfail: "L'email n'a pas pu partir. Réessayez, ou écrivez à warithmetics@deraison.ai.",
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
  // Same convention as the app: the flag shows the language you would
  // SWITCH TO, the sun/moon the theme you would switch to.
  const langBtn = document.getElementById("langToggle");
  langBtn.textContent = lang === "fr" ? "🇬🇧" : "🇫🇷";
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
