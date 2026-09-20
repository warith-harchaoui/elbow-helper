"""Refresh gate/free_domains.txt from its two upstream open-source lists.

"Is this a professional email?" is decided by DATA, not clever code: the
gate refuses any domain present in the community-maintained reference lists
that every off-the-shelf checker (npm ``free-email-domains``, pip
``disposable-email-domains``, the Composer equivalents) wraps:

- Kikobeats/free-email-domains — webmail / free providers (gmail.com,
  yahoo.*, orange.fr, laposte.net, protonmail.com, …), HubSpot-derived;
- disposable-email-domains/disposable-email-domains — throwaway providers
  (mailinator.com, yopmail.com, …), the blocklist Django and friends use.

The deployment is plain SFTP + PHP (no Composer on the host), so instead of
installing a package that embeds this data, we vendor the merged data itself
and keep THIS script as the documented, repeatable way to refresh it::

    python webapp/gate/update_blocklist.py   # rewrites gate/free_domains.txt

Then rebuild (``python webapp/build.py``) and re-upload. The file stays a
plain sorted list of lowercase domains, one per line: ``auth.php`` loads it
with ``file()`` and answers membership in O(1) via a hash set.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "free_domains.txt"

# The two reference lists every ready-made "is this a company email" package
# ultimately wraps. Pinned to their default branches: the data moves, the
# format (JSON array / one-domain-per-line) has been stable for years.
FREE_URL = (
    "https://raw.githubusercontent.com/Kikobeats/free-email-domains/master/domains.json"
)
DISPOSABLE_URL = (
    "https://raw.githubusercontent.com/disposable-email-domains/"
    "disposable-email-domains/main/disposable_email_blocklist.conf"
)

# If an upstream ever vanished or truncated, refuse to ship a weaker gate.
MIN_DOMAINS = 10_000
MUST_CONTAIN = ("gmail.com", "yahoo.fr", "orange.fr", "laposte.net", "mailinator.com")


def fetch(url: str) -> str:
    """Download one upstream list as text (10 s cap; raises on HTTP errors)."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read().decode("utf-8")


def main() -> None:
    """Merge both upstreams into free_domains.txt, with sanity checks."""
    free = json.loads(fetch(FREE_URL))
    disposable = [
        line.strip()
        for line in fetch(DISPOSABLE_URL).splitlines()
        if line.strip() and not line.startswith("#")
    ]
    merged = sorted({domain.strip().lower() for domain in [*free, *disposable]})

    if len(merged) < MIN_DOMAINS:
        raise SystemExit(
            f"only {len(merged)} domains fetched; upstream looks broken, aborting"
        )
    for probe in MUST_CONTAIN:
        if probe not in merged:
            raise SystemExit(f"{probe} missing from the merged list; aborting")

    OUT.write_text("\n".join(merged) + "\n", encoding="utf-8")
    print(
        f"{OUT.name}: {len(merged)} domains ({len(free)} free + {len(disposable)} disposable)"
    )


if __name__ == "__main__":
    main()
