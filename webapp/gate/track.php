<?php
// track.php — in-app activity beacon. The app page (track.js) posts one small
// JSON object per meaningful action; each is appended to the authenticated
// user's JSONL log. Anonymous posts are dropped: no cookie, no log line.

declare(strict_types=1);

require __DIR__ . '/auth.php';

if (($_SERVER['REQUEST_METHOD'] ?? '') !== 'POST') {
    http_response_code(405);
    exit;
}
$email = current_email();
if ($email === null) {
    http_response_code(403);
    exit;
}

$body = json_decode((string) file_get_contents('php://input'), true);
$event = is_array($body) ? (string) ($body['event'] ?? '') : '';
if (!preg_match('/^[a-z][a-z0-9_:-]{0,63}$/', $event)) {
    http_response_code(422);
    exit;
}
// The detail payload is caller-supplied: keep it, but bounded.
$detail = [];
if (is_array($body) && is_array($body['detail'] ?? null)) {
    foreach (array_slice($body['detail'], 0, 8, true) as $key => $value) {
        if (is_scalar($value)) {
            $detail[substr((string) $key, 0, 32)] = substr((string) $value, 0, 200);
        }
    }
}

log_event($email, $event, $detail);
http_response_code(204);
