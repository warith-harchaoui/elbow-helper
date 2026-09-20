"""Compose the Open Graph / Twitter card (1200x630) for the static deployment.

One reproducible image, sprezzature look: the real logo (assets/logo.png) on
the left, the monospace eyebrow + serif baseline + one plain-language line on
the right, paper background, neutral chrome. Bilingual card: the EN baseline
leads, the FR one follows in a muted tone, since one URL serves both languages.

Run from the repo root (fonts are looked up in the user's font library, with
system fallbacks)::

    python webapp/seo/make_og_card.py     # writes webapp/seo/og-card.png

The PNG is committed (build.py just copies it into dist/static/), so the card
never depends on the build machine's fonts.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "og-card.png"

W, H = 1200, 630
PAPER, INK, MUTED = "#FAFAFA", "#171717", "#6b7280"


def font(candidates: list[str], size: int) -> ImageFont.FreeTypeFont:
    """First loadable font among `candidates` (home fonts, then system), at `size`."""
    roots = [Path.home() / "Library/Fonts", Path("/System/Library/Fonts/Supplemental")]
    for name in candidates:
        for root in roots:
            p = root / name
            if p.exists():
                return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)  # last resort; keeps the script runnable


def main() -> None:
    """Draw and save the card."""
    img = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(img)

    # The logo, resized to a square panel on the left, vertically centered.
    logo = Image.open(REPO / "assets" / "logo.png").convert("RGBA")
    side = 380
    logo = logo.resize((side, side), Image.LANCZOS)
    img.paste(logo, (90, (H - side) // 2), logo)

    x = 550
    max_w = W - x - 60  # right margin

    def fit(candidates: list[str], size: int, text: str) -> ImageFont.FreeTypeFont:
        """Shrink `size` until `text` fits in the column (headline must not clip)."""
        f = font(candidates, size)
        while size > 20 and draw.textlength(text, font=f) > max_w:
            size -= 2
            f = font(candidates, size)
        return f

    serif = fit(
        ["RobotoSerif-Clean.ttf", "Georgia Bold.ttf"], 74, "Is that elbow real?"
    )
    serif_small = fit(
        ["RobotoSerif-Clean.ttf", "Georgia.ttf"], 44, "Le coude est-il réel ?"
    )
    mono = font(["RobotoMono-Regular.ttf", "Menlo.ttc", "Courier New.ttf"], 30)
    sans = font(["Roboto-Regular.ttf", "Helvetica.ttc"], 32)
    draw.text((x, 150), "elbow-helper", font=mono, fill=MUTED)
    draw.text((x, 205), "Is that elbow real?", font=serif, fill=INK)
    draw.text((x, 310), "Le coude est-il réel ?", font=serif_small, fill=MUTED)
    draw.text(
        (x, 410),
        "A knee with its uncertainty\nor a frank abstention.\nRuns entirely in your browser.",
        font=sans,
        fill=INK,
        spacing=14,
    )

    img.save(OUT, "PNG", optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
