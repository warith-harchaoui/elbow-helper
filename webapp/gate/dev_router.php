<?php
// dev_router.php — local stand-in for the deployed .htaccess, for `php -S`.
// Reproduces the same public/protected split so the whole gate flow (form,
// magic link, cookie, protected assets, beacon) is testable headlessly:
//
//   GATE_DEV=1 php -S 127.0.0.1:8323 -t webapp/dist webapp/gate/dev_router.php
//
// GATE_DEV=1 writes magic links to private/outbox.jsonl instead of mailing.

declare(strict_types=1);

$path = (string) parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH);
$root = (string) $_SERVER['DOCUMENT_ROOT'];

if (preg_match('#^/private(/|$)#', $path) || $path === '/gate/auth.php') {
    http_response_code(403);
    return true;
}
if (preg_match('#^/(index\.html$|backend-pyodide\.js$|(?:py|wheels|i18n)/)#', $path)) {
    require $root . '/gate/serve.php';
    return true;
}
if ($path === '/' || $path === '/index.php') {
    require $root . '/index.php';
    return true;
}
return false; // everything else: the built-in server's static handling
