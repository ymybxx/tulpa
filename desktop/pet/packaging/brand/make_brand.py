"""Draws the tulpa mark (headphones + crescent ahoge forming a t, with a golden star) and its app/menu-bar variants.

Regenerate: `python3 make_brand.py`, then rasterize the SVGs with Electron's Chromium
(`../../node_modules/electron/dist/Electron.app/Contents/MacOS/Electron render.cjs jobs.json`),
render tulpa-app-icon.svg (64 px and up) and tulpa-app-icon-small.svg (16 and 32 px) into a
`tulpa.iconset` and run `iconutil -c icns tulpa.iconset -o tulpa.icns`. The menu-bar PNG is embedded in electron/main.ts.
"""

import pathlib

HERE = pathlib.Path(__file__).parent
BLUE, GOLD = "#74B4F4", "#F7C548"


def star(cx, cy, r, fill):
    k = r * 0.2
    return (
        f'<path d="M{cx} {cy - r} C{cx + k} {cy - k} {cx + k} {cy - k} {cx + r} {cy} C{cx + k} {cy + k} {cx + k} {cy + k} {cx} {cy + r} '
        f'C{cx - k} {cy + k} {cx - k} {cy + k} {cx - r} {cy} C{cx - k} {cy - k} {cx - k} {cy - k} {cx} {cy - r} Z" fill="{fill}"/>'
    )


CRESCENT = "M318 216 C262 188 232 140 250 95 C268 50 330 28 418 34 C356 44 300 72 290 118 C283 156 296 192 318 216 Z"
_uid = [0]


def mark(color=BLUE, accent=GOLD, bold=False):
    """The mark in a 540 x 700 box. The crescent cuts a thin gap through the band and the top of the t.

    `bold` is for 16-32 px: a thicker band, no gap (it would blur to grey) and a bigger star.
    """
    _uid[0] += 1
    m = f"cut{_uid[0]}"
    return (
        f'<defs><mask id="{m}" maskUnits="userSpaceOnUse" x="0" y="0" width="540" height="700">'
        f'<rect width="540" height="700" fill="#fff"/><path d="{CRESCENT}" fill="#000" stroke="#000" stroke-width="{0 if bold else 26}" stroke-linejoin="round"/></mask></defs>'
        f'<g mask="url(#{m})">'
        # headphone band
        f'<path d="M70 330 A190 150 0 0 1 450 330" fill="none" stroke="{color}" stroke-width="{62 if bold else 40}"/>'
        # the t: stem rising into a tip under the crescent, a rounded foot, and its crossbar
        f'<path d="M213 585 V304 C213 262 250 234 330 226 C304 246 288 270 285 304 V572 C285 598 300 612 330 612 H356 '
        f'C375 612 388 625 388 644 C388 664 372 684 340 684 H300 C245 684 213 650 213 585 Z" fill="{color}"/>'
        f'<rect x="145" y="360" width="230" height="64" rx="28" fill="{color}"/>'
        f"</g>"
        # ear cups: shell and pad
        f'<path d="M75 340 C40 345 25 375 25 411 C25 447 40 477 75 482 Z" fill="{color}"/>'
        f'<rect x="83" y="332" width="42" height="158" rx="16" fill="{color}"/>'
        f'<path d="M445 340 C480 345 495 375 495 411 C495 447 480 477 445 482 Z" fill="{color}"/>'
        f'<rect x="395" y="332" width="42" height="158" rx="16" fill="{color}"/>'
        # the crescent ahoge and its star
        f'<path d="{CRESCENT}" fill="{color}"/>'
        + (star(430, 110, 100, accent) if bold else star(438, 106, 70, accent))
    )


def svg(w, h, body, view="0 0 540 700"):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="{view}">{body}</svg>\n'


(HERE / "tulpa-mark.svg").write_text(svg(540, 700, mark()))
(HERE / "tulpa-mark-mono.svg").write_text(svg(540, 700, mark("#000000", "#000000")))


# macOS app icon: 1024 canvas, 824 body with a soft shadow, a deep sky ground and the mark in white.
# A white ground with a pale mark vanished at list sizes (System Settings shows it at 32 px).
def app_icon(bold):
    return (
        '<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5AA8F5"/><stop offset="1" stop-color="#2566C8"/></linearGradient>'
        '<filter id="s" x="-20%" y="-20%" width="140%" height="140%"><feDropShadow dx="0" dy="10" stdDeviation="14" flood-color="#1E3A5F" flood-opacity=".28"/></filter></defs>'
        '<rect x="100" y="100" width="824" height="824" rx="185" fill="url(#g)" filter="url(#s)"/>'
        f'<g transform="translate(512 520) scale({0.98 if bold else 0.95}) translate(-270 -350)">{mark("#FFFFFF", "#FFD34D", bold)}</g>'
    )


(HERE / "tulpa-app-icon.svg").write_text(
    svg(1024, 1024, app_icon(False), "0 0 1024 1024")
)
(HERE / "tulpa-app-icon-small.svg").write_text(
    svg(1024, 1024, app_icon(True), "0 0 1024 1024")
)
# menu bar: a template (black + alpha) image; 18 pt tall, drawn at @2x
(HERE / "tulpa-tray.svg").write_text(
    svg(
        36,
        36,
        f'<g transform="translate(18 18) scale(0.05) translate(-270 -350)">{mark("#000000", "#000000")}</g>',
        "0 0 36 36",
    )
)
print("written")
