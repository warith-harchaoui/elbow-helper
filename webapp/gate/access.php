<?php
// access.php — POST endpoint of the landing form: validate the professional
// email, record the lead, and send the magic link. Responds in JSON; every
// refusal returns an error CODE the landing translates (the endpoint itself
// stays language-neutral).

declare(strict_types=1);

require __DIR__ . '/auth.php';

header('Content-Type: application/json; charset=UTF-8');

/** Emit the JSON response and stop. */
function respond(bool $ok, ?string $error = null, int $status = 200): void
{
    http_response_code($status);
    echo json_encode(['ok' => $ok, 'error' => $error]);
    exit;
}

if (($_SERVER['REQUEST_METHOD'] ?? '') !== 'POST') {
    respond(false, 'method', 405);
}

// Honeypot: the hidden "website" field is filled by naive bots only. Answer
// success so the bot moves on, but do nothing.
if (trim((string) ($_POST['website'] ?? '')) !== '') {
    respond(true);
}

$email = strtolower(trim((string) ($_POST['email'] ?? '')));
$lang = ($_POST['lang'] ?? 'en') === 'fr' ? 'fr' : 'en';

if ((string) ($_POST['consent'] ?? '') === '') {
    respond(false, 'consent', 422);
}
if ($email === '' || filter_var($email, FILTER_VALIDATE_EMAIL) === false) {
    respond(false, 'invalid', 422);
}
$domain = substr($email, strrpos($email, '@') + 1);
if (is_generic_domain($domain)) {
    respond(false, 'generic', 422);
}
// The domain must actually receive mail (MX, or A as the RFC fallback).
// Skipped in dev mode where the sandbox may not resolve DNS.
if (!dev_mode() && !checkdnsrr($domain, 'MX') && !checkdnsrr($domain, 'A')) {
    respond(false, 'nodomain', 422);
}
// Rate limits: a link is cheap but not free (mail quota, log noise).
if (!rate_limit_ok('ip:' . client_ip(), 10, 3600) || !rate_limit_ok('em:' . $email, 3, 86400)) {
    respond(false, 'ratelimit', 429);
}

// The lead line is the marketing deliverable: keep it append-only and complete.
file_put_contents(
    private_dir() . '/leads.jsonl',
    json_encode([
        'ts' => gmdate('c'),
        'email' => $email,
        'lang' => $lang,
        'ip' => client_ip(),
        'ua' => (string) ($_SERVER['HTTP_USER_AGENT'] ?? ''),
        'referer' => (string) ($_SERVER['HTTP_REFERER'] ?? ''),
    ], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES) . "\n",
    FILE_APPEND | LOCK_EX
);
log_event($email, 'access_request', ['lang' => $lang]);

if (!send_magic_link($email, $lang)) {
    respond(false, 'sendfail', 502);
}
respond(true);
