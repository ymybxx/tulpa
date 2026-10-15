import asyncio
from uuid import uuid4

import pytest

from fish_memory.companion.playback_caption import PlaybackCaption
from fish_memory.pet.protocol import parse_published_control


@pytest.mark.asyncio
async def test_background_caption_uses_device_progress_and_freezes_before_stop() -> (
    None
):
    current = ["你好，后半句还未播出。", 3, 2, "playing"]
    published = []
    caption = PlaybackCaption(
        uuid4(), uuid4(), uuid4(), lambda: tuple(current), published.append
    )
    assert published[-1]["visibleContent"] == "你好，"
    current[3] = "paused"
    await asyncio.sleep(0.12)
    assert published[-1]["status"] == "paused"
    caption.finish("interrupted")
    assert published[-1]["visibleContent"] == "你好"
    assert published[-1]["consumedOffset"] == 2
    count = len(published)
    current[:] = ["不该再变化", 0, 0, "playing"]
    await asyncio.sleep(0.12)
    assert len(published) == count
    parse_published_control({"type": "pet.playback.progress", "payload": published[-1]})
