"""Bounded, chronological contact sheets of complete observed desktop frames."""

from __future__ import annotations

import base64
import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from typing import cast

from PIL import Image, ImageDraw, ImageFont, ImageOps

from fish_memory.observations.environment_history_models import EnvironmentHistoryWindow

MAX_SHEET_PIXELS = 2_500_000
MAX_SHEET_BYTES = 1_200_000
_TILE_LONG_EDGE = 640
_LABEL_HEIGHT = 26
# For the pet: frames this alike are one moment (TA's own idle motion, a caret
# blink). Compared as small colour thumbnails; a subtitle change stays apart.
_SIMILAR_SIZE = (64, 36)
_SIMILAR_PIXEL_DELTA = 16
_SIMILAR_MAX_CHANGED = 0.005


@dataclass(frozen=True, slots=True)
class ContactSheet:
    jpeg_base64: str
    width: int
    height: int
    panels: tuple[dict[str, object], ...]


def build_contact_sheet(
    window: EnvironmentHistoryWindow, *, labels: bool = True, moments: bool = False
) -> ContactSheet:
    """Decode sequentially, deduplicate exact consecutive pixels, never crop.

    At most ten small RGB tiles and one bounded output canvas are retained.
    The source dimensions are checked before decoding to avoid trusting JPEG
    metadata supplied separately from the image itself. Summaries need the
    panel number and offset labels.

    `moments` is for the pet looking at the screen: nearly identical frames
    become one, and each remaining picture says how long ago it was ("now" for
    the last), so a row of look-alike screens reads as time passing rather
    than as a grid of cells. A single moment is sent as a plain screenshot.
    """
    if moments:
        labels = True
    label_height = _LABEL_HEIGHT if labels else 0
    if not window.frames:
        raise ValueError("a contact sheet requires frames")
    tiles: list[Image.Image] = []
    panels: list[dict[str, object]] = []
    previous_digest: bytes | None = None
    previous_thumb: Image.Image | None = None
    try:
        for frame in window.frames:
            with Image.open(BytesIO(frame.decode_jpeg())) as source:
                if source.format != "JPEG" or source.size != (
                    frame.pixel_width,
                    frame.pixel_height,
                ):
                    raise ValueError("frame JPEG dimensions do not match metadata")
                with source.convert("RGB") as rgb:
                    # Exact comparison preserves even a small subtitle change.
                    digest = hashlib.sha256(
                        f"{rgb.size}:{frame.display_id}:".encode() + rgb.tobytes()
                    ).digest()
                    offset = round(frame.observed_at - window.started_at, 3)
                    thumb = _thumbnail(rgb) if moments else None
                    if digest == previous_digest or (
                        thumb is not None
                        and previous_thumb is not None
                        and _alike(thumb, previous_thumb)
                    ):
                        panels[-1]["lastSeenSeconds"] = offset
                        if thumb is not None:
                            # Keep the latest look of the moment.
                            tiles[-1].close()
                            tiles[-1] = _tile(rgb)
                        continue
                    if thumb is not None:
                        previous_thumb = thumb
                    tiles.append(_tile(rgb))
                    panels.append(
                        {
                            "panel": len(panels) + 1,
                            "firstSeenSeconds": offset,
                            "lastSeenSeconds": offset,
                        }
                    )
                    previous_digest = digest
        if moments and len(tiles) == 1:
            label_height = 0
        columns = min(3, len(tiles))
        rows = math.ceil(len(tiles) / columns)
        cell_width = max(tile.width for tile in tiles)
        cell_height = max(tile.height for tile in tiles)
        # Reserve label space first; even square/portrait display changes fit.
        while (
            columns * cell_width * rows * (cell_height + label_height)
            > MAX_SHEET_PIXELS
        ):
            cell_width = max(1, math.floor(cell_width * 0.98))
            cell_height = max(1, math.floor(cell_height * 0.98))
        width, height = columns * cell_width, rows * (cell_height + label_height)
        with Image.new("RGB", (width, height), (20, 23, 28)) as canvas:
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default(size=18)
            for index, (tile, panel) in enumerate(zip(tiles, panels, strict=True)):
                x = index % columns * cell_width
                y = index // columns * (cell_height + label_height)
                if label_height and moments:
                    last_seen = cast(float, panel["lastSeenSeconds"])  # an offset
                    ago = window.ended_at - (window.started_at + last_seen)
                    label = (
                        "now"
                        if index == len(tiles) - 1
                        else f"{max(1, round(ago))}s ago"
                    )
                    draw.text((x + 6, y + 3), label, font=font, fill="white")
                elif label_height:
                    first = panel["firstSeenSeconds"]
                    last = panel["lastSeenSeconds"]
                    label = f"{index + 1:02d}  +{first:.2f}s"
                    if last != first:
                        label += f" / +{last:.2f}s"
                    draw.text((x + 6, y + 3), label, font=font, fill="white")
                with ImageOps.contain(tile, (cell_width, cell_height)) as fitted:
                    canvas.paste(
                        fitted,
                        (
                            x + (cell_width - fitted.width) // 2,
                            y + label_height + (cell_height - fitted.height) // 2,
                        ),
                    )
            for quality in (85, 75, 65, 55):
                output = BytesIO()
                canvas.save(output, format="JPEG", quality=quality)
                jpeg = output.getvalue()
                if len(jpeg) <= MAX_SHEET_BYTES:
                    return ContactSheet(
                        base64.b64encode(jpeg).decode("ascii"),
                        width,
                        height,
                        tuple(panels),
                    )
            raise ValueError("contact sheet exceeds its JPEG byte budget")
    finally:
        for tile in tiles:
            tile.close()


def _tile(rgb: Image.Image) -> Image.Image:
    return ImageOps.contain(
        rgb, (min(rgb.width, _TILE_LONG_EDGE), min(rgb.height, _TILE_LONG_EDGE))
    )


def _thumbnail(rgb: Image.Image) -> Image.Image:
    return rgb.resize(_SIMILAR_SIZE, Image.Resampling.BOX)


def _alike(first: Image.Image, second: Image.Image) -> bool:
    """True when too few thumbnail pixels changed to be a different moment."""
    # Pillow types getdata() as its opaque core sequence; it yields RGB tuples.
    first_pixels = cast(Iterable[tuple[int, ...]], first.getdata())
    second_pixels = cast(Iterable[tuple[int, ...]], second.getdata())
    changed = sum(
        1
        for a, b in zip(first_pixels, second_pixels, strict=True)
        if max(abs(x - y) for x, y in zip(a, b, strict=True)) > _SIMILAR_PIXEL_DELTA
    )
    return changed <= _SIMILAR_MAX_CHANGED * _SIMILAR_SIZE[0] * _SIMILAR_SIZE[1]
