from __future__ import annotations

import base64
from io import BytesIO
from uuid import UUID

import pytest
from PIL import Image
from pydantic import ValidationError

from fish_memory.observations.environment_history_contact_sheet import (
    MAX_SHEET_BYTES,
    MAX_SHEET_PIXELS,
    build_contact_sheet,
)
from fish_memory.observations.environment_history_models import EnvironmentHistoryWindow

START = 1_788_200_000.0


def frame(index, color="red", size=(320, 200)):
    output = BytesIO()
    with Image.new("RGB", size, color) as image:
        image.save(output, format="JPEG")
    jpeg = output.getvalue()
    return {
        "frameId": str(UUID(int=index + 10)),
        "revision": index + 1,
        "observedAt": START + index,
        "expiresAt": START + index + 3,
        "displayID": 1,
        "pixelWidth": size[0],
        "pixelHeight": size[1],
        "byteLength": len(jpeg),
        "mimeType": "image/jpeg",
        "jpegBase64": base64.b64encode(jpeg).decode(),
    }


def window(frames):
    return EnvironmentHistoryWindow.model_validate(
        {
            "captureSessionId": str(UUID(int=1)),
            "windowIndex": 0,
            "startedAt": START,
            "endedAt": START + 10,
            "frames": frames,
            "audioStatus": "unavailable",
            "audioCapturedSamples": 0,
        }
    )


def test_ten_frames_stay_in_temporal_order_and_one_bounded_jpeg():
    source = window([frame(i, (i * 25, 0, 0), size=(960, 960)) for i in range(10)])
    sheet = build_contact_sheet(source)
    assert len(sheet.panels) == 10
    assert [p["firstSeenSeconds"] for p in sheet.panels] == list(range(10))
    assert sheet.width * sheet.height <= MAX_SHEET_PIXELS
    jpeg = base64.b64decode(sheet.jpeg_base64)
    assert len(jpeg) <= MAX_SHEET_BYTES
    with Image.open(BytesIO(jpeg)) as image:
        assert image.size == (sheet.width, sheet.height)
        assert image.format == "JPEG"


def test_unlabeled_sheet_keeps_panels_and_drops_only_the_label_band():
    source = window([frame(0), frame(1, "blue"), frame(2, "green"), frame(3, "red")])
    labeled = build_contact_sheet(source)
    plain = build_contact_sheet(source, labels=False)
    assert plain.panels == labeled.panels
    assert plain.width == labeled.width
    assert plain.height == labeled.height - 2 * 26
    with Image.open(BytesIO(base64.b64decode(plain.jpeg_base64))) as image:
        # The first tile starts at the top edge instead of below a label.
        red, green, blue = image.convert("RGB").getpixel((160, 5))
        assert red > 200 and green < 60 and blue < 60


def test_only_consecutive_identical_frames_collapse_and_keep_both_times():
    sheet = build_contact_sheet(
        window([frame(0), frame(1), frame(3, "blue"), frame(5, "red"), frame(9, "red")])
    )
    assert sheet.panels == (
        {"panel": 1, "firstSeenSeconds": 0, "lastSeenSeconds": 1},
        {"panel": 2, "firstSeenSeconds": 3, "lastSeenSeconds": 3},
        {"panel": 3, "firstSeenSeconds": 5, "lastSeenSeconds": 9},
    )


def test_complete_frame_edges_are_kept_without_cropping_or_stretching():
    source = frame(0)
    output = BytesIO()
    with Image.new("RGB", (320, 200), "black") as image:
        image.paste("red", (0, 0, 30, 200))
        image.paste("blue", (290, 0, 320, 200))
        image.paste("green", (30, 170, 290, 200))
        image.save(output, format="JPEG", quality=95)
    jpeg = output.getvalue()
    source.update(byteLength=len(jpeg), jpegBase64=base64.b64encode(jpeg).decode())
    sheet = build_contact_sheet(window([source]))
    with Image.open(BytesIO(base64.b64decode(sheet.jpeg_base64))) as image:
        assert image.size == (320, 226)
        assert image.getpixel((10, 100))[0] > 200
        assert image.getpixel((310, 100))[2] > 200
        assert image.getpixel((150, 210))[1] > 100


def test_small_subtitle_change_is_not_discarded_as_a_similar_frame():
    first = frame(0, "black")
    second = frame(1, "black")
    output = BytesIO()
    with Image.new("RGB", (320, 200), "black") as image:
        image.paste("white", (159, 179, 161, 181))
        image.save(output, format="JPEG", quality=95)
    jpeg = output.getvalue()
    second.update(byteLength=len(jpeg), jpegBase64=base64.b64encode(jpeg).decode())
    assert len(build_contact_sheet(window([first, second])).panels) == 2


def test_bad_jpeg_or_false_dimensions_cannot_create_a_sheet():
    item = frame(0)
    item["pixelWidth"] = 1
    with pytest.raises(ValueError, match="dimensions"):
        build_contact_sheet(window([item]))
    item = frame(0)
    item.update(
        byteLength=6, jpegBase64=base64.b64encode(b"\xff\xd8\xff\x03\xff\xd9").decode()
    )
    with pytest.raises(OSError):
        build_contact_sheet(window([item]))


def test_eleventh_frame_is_rejected_at_the_window_boundary():
    with pytest.raises(ValidationError):
        window([frame(i) for i in range(11)])


def drawn_frame(index, *, box=None, strip=None, size=(960, 540)):
    """A settings-like screen: `box` is a small moving figure, `strip` a subtitle."""
    output = BytesIO()
    with Image.new("RGB", size, (40, 44, 52)) as image:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(image)
        draw.rectangle((525, 60, 930, 480), fill=(230, 220, 200))
        if box is not None:
            draw.ellipse(box, fill=(250, 180, 120))
        if strip is not None:
            draw.rectangle((75, 450 + strip, 885, 495 + strip), fill=(255, 255, 255))
        image.save(output, format="JPEG", quality=92)
    jpeg = output.getvalue()
    return {
        "frameId": str(UUID(int=index + 10)),
        "revision": index + 1,
        "observedAt": START + index,
        "expiresAt": START + index + 3,
        "displayID": 1,
        "pixelWidth": size[0],
        "pixelHeight": size[1],
        "byteLength": len(jpeg),
        "mimeType": "image/jpeg",
        "jpegBase64": base64.b64encode(jpeg).decode(),
    }


def test_for_the_pet_a_barely_moving_screen_is_one_plain_picture():
    # TA breathing on its own settings page: twelve frames, a few pixels apart.
    frames = [
        drawn_frame(i, box=(90, 150 + i % 3, 135, 195 + i % 3)) for i in range(10)
    ]
    sheet = build_contact_sheet(window(frames), moments=True)
    assert len(sheet.panels) == 1
    # One moment is just the screen, without a time band or a grid.
    assert (sheet.width, sheet.height) == (640, 360)
    # The audiovisual summaries still keep every frame that is not pixel-identical.
    assert len(build_contact_sheet(window(frames)).panels) == 10


def test_for_the_pet_a_new_subtitle_is_a_new_moment_labelled_by_time():
    frames = [
        drawn_frame(0),
        drawn_frame(1),
        drawn_frame(2, strip=0),
        drawn_frame(3, strip=0),
    ]
    sheet = build_contact_sheet(window(frames), moments=True)
    assert [panel["firstSeenSeconds"] for panel in sheet.panels] == [0, 2]
    # A band above each picture carries "…s ago" or "now".
    assert sheet.height == 360 + 26
