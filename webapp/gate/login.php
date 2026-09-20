<?php
// login.php — the magic-link target. A valid, unexpired link sets the signed
// auth cookie (30 days) and lands the visitor straight in the app; anything
// else bounces back to the landing page with an error flag it can translate.

declare(strict_types=1);

require __DIR__ . '/auth.php';

$email = b64url_decode((string) ($_GET['e'] ?? ''));
$expiry = (string) ($_GET['x'] ?? '');
$mac = (string) ($_GET['t'] ?? '');

$valid = $email !== ''
    && filter_var($email, FILTER_VALIDATE_EMAIL) !== false
    && ctype_digit($expiry)
    && hash_equals(gate_token($email, (int) $expiry), $mac);

if (!$valid) {
    header('Location: ' . gate_base_url() . '/?link=invalid');
    exit;
}
if ((int) $expiry < time()) {
    // Expired is the one recoverable failure: the landing pre-fills the email
    // so one click requests a fresh link.
    header('Location: ' . gate_base_url() . '/?link=expired&email=' . rawurlencode($email));
    exit;
}

set_auth_cookie($email);
log_event($email, 'login');
header('Location: ' . gate_base_url() . '/index.html');
