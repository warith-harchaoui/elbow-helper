<?php
// serve.php — the only path to the app's files. The bundle .htaccess rewrites
// every protected URL (index.html, backend-pyodide.js, py/, wheels/, i18n/)
// here, so Apache never serves them directly: no cookie, no file — guessing
// an asset URL yields the same 403 as the front door.

declare(strict_types=1);

require __DIR__ . '/auth.php';

// The protected tail of the URL, matched with a charset too narrow to smuggle
// query strings or encodings in; realpath below re-checks against traversal.
$path = (string) parse_url($_SERVER['REQUEST_URI'] ?? '', PHP_URL_PATH);
$ok = preg_match(
    '#/(index\.html|backend-pyodide\.js|(?:py|wheels|i18n)/[A-Za-z0-9._-]+)$#',
    $path,
    $m
);
if (!$ok) {
    http_response_code(404);
    exit;
}
$rel = $m[1];

$email = current_email();
if ($email === null) {
    if ($rel === 'index.html') {
        // A person following an old bookmark: send them to the landing form.
        header('Location: ' . gate_base_url() . '/?login=required');
    } else {
        http_response_code(403);
    }
    exit;
}

$root = realpath(gate_root());
$file = realpath(gate_root() . '/' . $rel);
if ($file === false || $root === false || !str_starts_with($file, $root . DIRECTORY_SEPARATOR)) {
    http_response_code(404);
    exit;
}

// Entering the app page is the session-level activity signal; individual
// assets would only repeat it (the in-page beacon covers the fine grain).
if ($rel === 'index.html') {
    log_event($email, 'app_page');
}

$types = [
    'html' => 'text/html; charset=UTF-8',
    'js' => 'application/javascript; charset=UTF-8',
    'json' => 'application/json; charset=UTF-8',
    'py' => 'text/x-python; charset=UTF-8',
    'whl' => 'application/octet-stream',
];
$ext = strtolower(pathinfo($file, PATHINFO_EXTENSION));
header('Content-Type: ' . ($types[$ext] ?? 'application/octet-stream'));
header('Content-Length: ' . (string) filesize($file));
// Private: cacheable by the authenticated browser (the wheel is not tiny),
// never by shared caches that would bypass the gate.
header('Cache-Control: private, max-age=3600');
readfile($file);
