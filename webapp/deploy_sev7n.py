"""Deploy the OPEN-ACCESS build of the web app to sev7n's public SFTP.

Two things distinguish this deployment from the deraison.ai one:

- **No gate.** The bundle is built with ``--no-gate``: no ``index.php``, no
  ``gate/``, no ``.htaccess``, no ``private/``, no activity beacon. Nobody is
  asked for an email address; ``index.html`` is the entry point and every
  visitor lands straight in the app.
- **No PHP.** The host is SFTPGo's HTTPS front (``sftp.s7n.app``), which
  serves static files and nothing else. That is exactly what an ungated
  ``dist/`` needs, since the engine runs in the visitor's browser.

The canonical URL still points at ``https://deraison.ai/elbow-helper`` (the
build default): this copy is a mirror, so it must not compete with the home
site in a search index.

Credentials are never written here. They are read from ``~/sev7n/settings.yaml``
by ``sftp_helper.credentials`` — the same file ``sftp-helper --config`` takes,
kept at permissions 600 (see ``~/sev7n/sftp.md``).

Where it lands, and why the two paths differ: the SFTP account is chrooted, so
the folder is ``/private/sev7n-avec-warith/demo-client-public/elbow-helper``
over SFTP while the public reader sees it under ``sftp_https``, whose path
carries the account name. ``demo-client-public/`` is the established
convention for "anyone with the link may read this".

Usage from the repo root, with the project env active::

    python webapp/deploy_sev7n.py             # build, then upload
    python webapp/deploy_sev7n.py --dry-run   # build and print, upload nothing
    python webapp/deploy_sev7n.py --skip-build  # upload the current dist/ as is

Beware that a plain run leaves ``webapp/dist/`` in its UNGATED shape. Rebuild
with ``python webapp/build.py --clean`` before deploying to deraison.ai again.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import sftp_helper as sftph

WEBAPP = Path(__file__).resolve().parent
DIST = WEBAPP / "dist"

# The credentials file (never a copy of it, never its contents inlined here).
CONFIG = Path.home() / "sev7n" / "settings.yaml"

# The chrooted SFTP path, and the prefix of it that `sftp_https` already covers.
REMOTE_DIR = "/private/sev7n-avec-warith/demo-client-public/elbow-helper"
REMOTE_ROOT = "/private/sev7n-avec-warith"


def public_url(credentials: dict) -> str:
    """Return the HTTPS URL a reader opens for :data:`REMOTE_DIR`."""
    base = credentials["sftp_https"].rstrip("/")
    return base + REMOTE_DIR[len(REMOTE_ROOT) :] + "/"


def main() -> None:
    """Build the ungated bundle, then ship it to the public SFTP."""
    parser = argparse.ArgumentParser(
        description="Deploy the ungated web app to sev7n's public SFTP."
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="upload the current dist/ as is"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="build and report, upload nothing"
    )
    args = parser.parse_args()

    if not args.skip_build:
        subprocess.run(
            [sys.executable, str(WEBAPP / "build.py"), "--clean", "--no-gate"],
            check=True,
        )

    # A gate file in dist/ means the bundle was built without --no-gate; the
    # host cannot run PHP, so index.php would be served as source text.
    stowaways = sorted(
        p.relative_to(DIST).as_posix()
        for p in DIST.rglob("*")
        if p.suffix == ".php" or p.name == ".htaccess"
    )
    if stowaways:
        raise SystemExit(
            f"dist/ still carries gate files {stowaways}; "
            "rebuild with: python webapp/build.py --clean --no-gate"
        )

    credentials = sftph.credentials(str(CONFIG))
    url = public_url(credentials)
    total = sum(f.stat().st_size for f in DIST.rglob("*") if f.is_file())
    print(f"\n{DIST} ({total / 1e6:.1f} MB) → {REMOTE_DIR}")

    if args.dry_run:
        print("dry run: nothing uploaded")
    else:
        sftph.upload(str(DIST), credentials, sftp_address=REMOTE_DIR)
        print("uploaded")

    print(f"public URL: {url}")


if __name__ == "__main__":
    main()
