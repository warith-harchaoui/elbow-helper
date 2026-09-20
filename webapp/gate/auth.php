<?php
// auth.php — shared library of the elbow-helper lead-magnet gate.
//
// The gate turns the static deployment into an email-gated lead magnet: the
// interactive app (index.html + engine files) is only served to visitors who
// received a magic link at a PROFESSIONAL email address (generic/free/
// disposable domains are refused). Everything is file-based (no database):
//
//   private/secret.php     HMAC secret, generated on first request (0600)
//   private/leads.jsonl    one line per access request (email, ts, ip, ua)
//   private/logs/<em>.jsonl per-user activity log (login, app_open, generate…)
//   private/rl/            rate-limit counters (one small file per key)
//   private/outbox.jsonl   DEV MODE ONLY: magic links written here instead of
//                          being emailed (enable with the GATE_DEV=1 env var)
//
// private/ is denied to the web both by its own .htaccess and by a rewrite
// rule in the bundle root .htaccess; the secret is additionally a .php file
// so a misconfigured host would execute it (empty output) rather than leak it.
//
// PHP 8.1 compatible (the deraison.ai host runs 8.1; do not use 8.2+ syntax).

declare(strict_types=1);

const GATE_COOKIE = 'eh_auth';
const COOKIE_DAYS = 30;      // authenticated-session lifetime
const LINK_HOURS = 72;       // magic-link validity window
const OWNER_EMAIL = 'warith@deraison.ai';
const FROM_EMAIL = 'no-reply@deraison.ai';

/** Filesystem root of the deployed bundle (the directory holding index.php). */
function gate_root(): string
{
    return dirname(__DIR__);
}

/**
 * Self-heal the bundle root .htaccess (the gate's single point of failure).
 *
 * SFTP syncs routinely drop or overwrite dotfiles: lose this file and Apache
 * serves the whole app ungated (observed in production on 2026-09-20, when a
 * foreign project's sync replaced both index.php and .htaccess). So every
 * gate endpoint re-asserts it: if the root .htaccess is missing or no longer
 * contains our marker rule, it is rewritten from gate/htaccess.dist and the
 * incident is logged to private/heal.log. Best-effort: an unwritable root
 * must never break the endpoint that called us.
 */
function ensure_htaccess(): void
{
    $target = gate_root() . '/.htaccess';
    $marker = 'gate/serve.php';
    if (is_file($target) && str_contains((string) file_get_contents($target), $marker)) {
        return;
    }
    $template = __DIR__ . '/htaccess.dist';
    if (!is_file($template)) {
        return;
    }
    if (@file_put_contents($target, file_get_contents($template), LOCK_EX) !== false) {
        @file_put_contents(
            private_dir() . '/heal.log',
            gmdate('c') . " root .htaccess restored from gate/htaccess.dist\n",
            FILE_APPEND | LOCK_EX
        );
    }
}

/** The private data directory, created (and web-denied) on first use. */
function private_dir(): string
{
    $dir = gate_root() . '/private';
    if (!is_dir($dir)) {
        mkdir($dir, 0700, true);
    }
    $ht = $dir . '/.htaccess';
    if (!file_exists($ht)) {
        file_put_contents($ht, "Require all denied\n", LOCK_EX);
    }
    return $dir;
}

/** The per-deployment HMAC secret, generated once server-side (never in git). */
function gate_secret(): string
{
    $file = private_dir() . '/secret.php';
    if (!file_exists($file)) {
        $secret = bin2hex(random_bytes(32));
        file_put_contents($file, "<?php return '" . $secret . "';\n", LOCK_EX);
        @chmod($file, 0600);
    }
    return (string) require $file;
}

/** Dev mode (local testing without a mail server): GATE_DEV=1 in the env. */
function dev_mode(): bool
{
    return getenv('GATE_DEV') === '1';
}

/** URL-safe base64, used for embedding emails in links and cookies. */
function b64url_encode(string $raw): string
{
    return rtrim(strtr(base64_encode($raw), '+/', '-_'), '=');
}

/** Inverse of b64url_encode; returns '' on malformed input. */
function b64url_decode(string $enc): string
{
    $decoded = base64_decode(strtr($enc, '-_', '+/'), true);
    return $decoded === false ? '' : $decoded;
}

/** HMAC binding an email to an expiry timestamp (used by links AND cookies). */
function gate_token(string $email, int $expiry): string
{
    return hash_hmac('sha256', $email . '|' . $expiry, gate_secret());
}

/** Absolute URL of the bundle root (scheme + host + mount path, no trailing /). */
function gate_base_url(): string
{
    $https = (!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS'] !== 'off')
        || (($_SERVER['HTTP_X_FORWARDED_PROTO'] ?? '') === 'https');
    $scheme = $https ? 'https' : 'http';
    $host = $_SERVER['HTTP_HOST'] ?? 'localhost';
    // The gate scripts live at <mount>/gate/<script>.php. Direct hits carry
    // that in REQUEST_URI; Apache-rewritten hits (serve.php) carry it in
    // SCRIPT_NAME instead. Neither (the dev router) means mounted at ''.
    $requested = (string) parse_url($_SERVER['REQUEST_URI'] ?? '/', PHP_URL_PATH);
    $script = (string) ($_SERVER['SCRIPT_NAME'] ?? '');
    $mount = '';
    foreach ([$requested, $script] as $path) {
        $pos = strpos($path, '/gate/');
        if ($pos !== false) {
            $mount = substr($path, 0, $pos);
            break;
        }
    }
    return $scheme . '://' . $host . $mount;
}

/** The visitor's email if their auth cookie is valid and unexpired, else null. */
function current_email(): ?string
{
    $raw = $_COOKIE[GATE_COOKIE] ?? '';
    $parts = explode('.', $raw);
    if (count($parts) !== 3) {
        return null;
    }
    [$enc, $expiry, $mac] = $parts;
    $email = b64url_decode($enc);
    if ($email === '' || !ctype_digit($expiry) || (int) $expiry < time()) {
        return null;
    }
    if (!hash_equals(gate_token($email, (int) $expiry), $mac)) {
        return null;
    }
    return $email;
}

/** Issue the auth cookie for an email (called by login.php after link check). */
function set_auth_cookie(string $email): void
{
    $expiry = time() + COOKIE_DAYS * 86400;
    $value = b64url_encode($email) . '.' . $expiry . '.' . gate_token($email, $expiry);
    // Path '/' rather than the mount path: the cookie must survive both the
    // Apache deployment under /elbow-helper/ and the local dev server at /.
    setcookie(GATE_COOKIE, $value, [
        'expires' => $expiry,
        'path' => '/',
        'secure' => str_starts_with(gate_base_url(), 'https'),
        'httponly' => true,
        'samesite' => 'Lax',
    ]);
}

/** Best-effort client IP (shared hosting sits behind no trusted proxy chain). */
function client_ip(): string
{
    return (string) ($_SERVER['REMOTE_ADDR'] ?? 'unknown');
}

/** Append one activity line to the per-user JSONL log (the precise audit trail). */
function log_event(string $email, string $event, array $detail = []): void
{
    $dir = private_dir() . '/logs';
    if (!is_dir($dir)) {
        mkdir($dir, 0700, true);
    }
    $safe = preg_replace('/[^a-z0-9@._+-]/', '_', strtolower($email));
    $line = json_encode([
        'ts' => gmdate('c'),
        'event' => $event,
        'detail' => $detail === [] ? null : $detail,
        'ip' => client_ip(),
    ], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    file_put_contents($dir . '/' . $safe . '.jsonl', $line . "\n", FILE_APPEND | LOCK_EX);
}

/** True when the email's domain is on the vendored generic/disposable blocklist. */
function is_generic_domain(string $domain): bool
{
    static $set = null;
    if ($set === null) {
        $lines = file(__DIR__ . '/free_domains.txt', FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
        $set = $lines === false ? [] : array_fill_keys($lines, true);
    }
    return isset($set[strtolower($domain)]);
}

/**
 * Sliding-window rate limiter backed by one small file per key.
 * Returns true when the caller is still under `$max` events per `$window` s.
 */
function rate_limit_ok(string $key, int $max, int $window): bool
{
    $dir = private_dir() . '/rl';
    if (!is_dir($dir)) {
        mkdir($dir, 0700, true);
    }
    $file = $dir . '/' . hash('sha256', $key) . '.json';
    $now = time();
    $times = [];
    if (is_file($file)) {
        $times = json_decode((string) file_get_contents($file), true) ?: [];
    }
    $times = array_values(array_filter($times, fn ($t) => $t > $now - $window));
    if (count($times) >= $max) {
        return false;
    }
    $times[] = $now;
    file_put_contents($file, json_encode($times), LOCK_EX);
    return true;
}

/**
 * Email the magic link (and notify the owner). In dev mode the message is
 * appended to private/outbox.jsonl instead, so the flow is testable offline.
 */
function send_magic_link(string $email, string $lang): bool
{
    $expiry = time() + LINK_HOURS * 3600;
    $link = gate_base_url() . '/gate/login.php?e=' . b64url_encode($email)
        . '&x=' . $expiry . '&t=' . gate_token($email, $expiry);
    $fr = $lang === 'fr';
    $subject = $fr ? 'Votre accès à elbow-helper' : 'Your access to elbow-helper';
    $hours = (string) LINK_HOURS;
    $body = $fr
        ? "Bonjour,\n\nVoici votre lien d'accès à elbow-helper (valable {$hours} h) :\n\n{$link}\n\n"
            . "Un clic vous connecte pour " . COOKIE_DAYS . " jours sur cet appareil.\n\n"
            . "Warith Harchaoui — https://deraison.ai"
        : "Hello,\n\nHere is your access link to elbow-helper (valid for {$hours} h):\n\n{$link}\n\n"
            . "One click signs you in for " . COOKIE_DAYS . " days on this device.\n\n"
            . "Warith Harchaoui — https://deraison.ai";
    if (dev_mode()) {
        file_put_contents(
            private_dir() . '/outbox.jsonl',
            json_encode(['ts' => gmdate('c'), 'to' => $email, 'link' => $link]) . "\n",
            FILE_APPEND | LOCK_EX
        );
        return true;
    }
    $headers = 'From: Elbow Helper <' . FROM_EMAIL . ">\r\n"
        . 'Reply-To: ' . OWNER_EMAIL . "\r\n"
        . "Content-Type: text/plain; charset=UTF-8\r\n";
    $sent = mail($email, $subject, $body, $headers);
    // Owner notification: one line per new lead, best-effort (never blocks access).
    @mail(OWNER_EMAIL, 'elbow-helper lead: ' . $email, "New access request from {$email}\n", $headers);
    return $sent;
}

// Every endpoint that includes this library re-asserts the root .htaccess:
// the lock must survive dotfile-dropping SFTP syncs (see ensure_htaccess).
ensure_htaccess();
