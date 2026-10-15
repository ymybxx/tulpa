import asyncio
import base64
from datetime import UTC, datetime, timedelta
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from fish_memory.environment_frames import (
    EnvironmentFrameCache,
    EnvironmentFrameUploadRequest,
    EnvironmentPerceptionPolicy,
)
from fish_memory.experimental.perception import ExperimentalPerception, PerceptionInput
from fish_memory.llm.ark_search import _input_messages
from fish_memory.llm.client import _provider_messages

NOW = datetime(2026, 9, 19, 8, 40, tzinfo=UTC)


def fixture():
    clock = SimpleNamespace(now=NOW.timestamp())
    cache = EnvironmentFrameCache(now=lambda: clock.now)
    fish = uuid4()
    cache.set_policy(fish, EnvironmentPerceptionPolicy(False, True, True))
    capability = SimpleNamespace(image_input_capability="supported")
    return SimpleNamespace(
        clock=clock,
        cache=cache,
        fish=fish,
        capability=capability,
        perception=ExperimentalPerception(cache),
    )


def put(f, offset, color="red", *, revision=1):
    f.clock.now = NOW.timestamp() + offset
    output = BytesIO()
    with Image.new("RGB", (320, 200), color) as image:
        image.save(output, format="JPEG")
    jpeg = output.getvalue()
    frame = EnvironmentFrameUploadRequest(
        frameId=uuid4(),
        revision=revision,
        observedAt=f.clock.now,
        expiresAt=f.clock.now + 3,
        displayID=1,
        pixelWidth=320,
        pixelHeight=200,
        byteLength=len(jpeg),
        mimeType="image/jpeg",
        jpegBase64=base64.b64encode(jpeg).decode(),
    )
    assert f.cache.put(f.fish, frame, jpeg)
    return frame


@pytest.mark.asyncio
@pytest.mark.parametrize("audiovisual", [True, False])
@pytest.mark.parametrize(
    ("zone", "local_time"),
    [("UTC", "08:39:50+00:00"), ("Asia/Shanghai", "16:39:50+08:00")],
)
async def test_latest_ten_seconds_are_one_ordered_sheet_in_actual_wire_request(
    audiovisual,
    zone,
    local_time,
    system_timezone,
):
    system_timezone(zone)
    f = fixture()
    f.cache.set_policy(f.fish, EnvironmentPerceptionPolicy(False, True, audiovisual))
    put(f, -11, "black")
    colors = tuple((i * 25, 128, 255 - i * 25) for i in range(10))
    for i, color in enumerate(colors):
        put(f, i - 9.9, color, revision=i + 2)
    attachments = (await f.perception.capture(f.fish, NOW, model=f.capability)).images
    assert len(attachments) == 1
    attachment = attachments[0]
    # The pet reads what happened in order, never sensor window details.
    assert attachment["description"] == (
        "你最近几秒看到的屏幕，是同一块屏幕在不同时刻的样子，"
        "按时间先后排列，每张上方标着是几秒前，最后一张（now）是现在。"
    )
    assert local_time not in attachment["description"]
    assert datetime.fromisoformat(attachment["observed_at"]).timestamp() == int(
        NOW.timestamp() - 0.9
    )
    with Image.open(
        BytesIO(base64.b64decode(attachment["data_url"].split(",", 1)[1]))
    ) as image:
        # Each distinct moment carries a short "…s ago" / "now" band above it.
        assert image.size == (960, 4 * (200 + 26))
        for i, color in enumerate(colors):
            pixel = image.getpixel((i % 3 * 320 + 160, i // 3 * 226 + 26 + 100))
            assert all(
                abs(left - right) < 8 for left, right in zip(pixel, color, strict=True)
            )
    messages = [
        {"role": "system", "content": "你正在陪伴用户。"},
        {
            "role": "user",
            "content": "当前上下文",
            "environment_images": attachments,
        },
    ]
    wire = _provider_messages(messages)
    content = wire[-1]["content"]
    assert content[1]["text"] == ("你看到的屏幕画面：\n" + attachment["description"])
    assert content[2]["image_url"]["detail"] == "high"
    assert content[2]["image_url"]["url"] == attachment["data_url"]
    ark = _input_messages(messages)[-1]["content"]
    assert ark[1]["text"] == content[1]["text"]
    assert ark[2]["detail"] == "high"
    assert ark[2]["image_url"] == attachment["data_url"]


def test_screen_voice_history_keeps_experimental_input_at_ten_frames():
    f = fixture()
    f.cache.set_policy(f.fish, EnvironmentPerceptionPolicy(False, True, False))
    for index in range(12):
        put(f, index - 11, revision=index + 1)
    assert len(f.cache._history[f.fish]) == 12
    assert len(f.cache.recent_frames(f.fish, through_at=NOW.timestamp())) == 10


def test_recent_frames_sample_each_second_and_keep_the_newest_actual_capture():
    f = fixture()
    for index, offset in enumerate(
        (-10, -9.9, -7.9, -5.9, -3.9, -1.9, -0.9, -0.7, 0.1)
    ):
        put(f, offset, revision=index + 1)
    frames = f.cache.recent_frames(f.fish, through_at=NOW.timestamp())
    assert [round(frame.observed_at - NOW.timestamp(), 1) for frame in frames] == [
        -9.9,
        -7.9,
        -5.9,
        -3.9,
        -1.9,
        -0.7,
    ]
    assert f.cache.recent_frames(uuid4(), through_at=NOW.timestamp()) == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", ["unsupported"])
async def test_unavailable_image_capability_does_not_send_screen(capability):
    f = fixture()
    put(f, -1)
    f.capability.image_input_capability = capability
    assert (await f.perception.capture(f.fish, NOW, model=f.capability)).images == ()


@pytest.mark.asyncio
async def test_absent_expired_or_disabled_screen_does_not_fabricate_ten_frames():
    f = fixture()
    assert (await f.perception.capture(f.fish, NOW, model=f.capability)).images == ()
    put(f, -11)
    assert (await f.perception.capture(f.fish, NOW, model=f.capability)).images == ()
    put(f, -1, revision=2)
    images = (await f.perception.capture(f.fish, NOW, model=f.capability)).images
    assert len(images) == 1 and "帧" not in images[0]["description"]
    f.cache.set_policy(f.fish, EnvironmentPerceptionPolicy())
    assert (await f.perception.capture(f.fish, NOW, model=f.capability)).images == ()


@pytest.mark.asyncio
async def test_disabling_perception_during_composition_discards_the_image(monkeypatch):
    f = fixture()
    f.perception.audio = SimpleNamespace(heard_between=lambda *args: "旧声音")
    put(f, -1)
    to_thread = asyncio.to_thread

    async def compose_and_disable(*args, **kwargs):
        result = await to_thread(*args, **kwargs)
        f.cache.set_policy(f.fish, EnvironmentPerceptionPolicy())
        return result

    monkeypatch.setattr(asyncio, "to_thread", compose_and_disable)
    assert (
        await f.perception.capture(f.fish, NOW, model=f.capability) == PerceptionInput()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("screen", [False, True])
@pytest.mark.parametrize("audio", [False, True])
async def test_shared_input_freezes_heard_and_frames_from_the_same_interval(
    screen, audio
):
    f = fixture()
    f.cache.set_policy(f.fish, EnvironmentPerceptionPolicy(False, screen, audio))
    calls = []

    def heard(fish, since, through):
        calls.append((fish, since, through))
        return "17:34:01 - 17:34:02\n声源A：听到的原话。"

    f.perception.audio = SimpleNamespace(heard_between=heard)
    if screen:
        put(f, -1)
    captured = await f.perception.capture(f.fish, NOW, model=f.capability)
    assert bool(captured.images) == screen
    assert bool(captured.heard) == audio
    assert calls == ([(f.fish, NOW - timedelta(seconds=10), NOW)] if audio else [])
    message = {
        "role": "user",
        "content": "本轮触发：定时唤醒",
        "request_time": "当前时间：末尾",
    }
    captured.attach(message)
    for transport in (_provider_messages, _input_messages):
        wire = transport([message])[-1]["content"]
        text = (
            wire
            if isinstance(wire, str)
            else "\n".join(p.get("text", "") for p in wire)
        )
        assert text.endswith("当前时间：末尾")
        assert ("你听到的声音：" in text) == audio


@pytest.mark.asyncio
async def test_unknown_target_model_gets_images_without_an_extra_probe():
    f = fixture()
    put(f, -1)
    unknown = SimpleNamespace(image_input_capability="unknown")
    assert (await f.perception.capture(f.fish, NOW, model=unknown)).images
    f.perception.audio = SimpleNamespace(heard_between=lambda *args: "已转写的声音")
    unsupported = SimpleNamespace(image_input_capability="unsupported")
    result = await f.perception.capture(f.fish, NOW, model=unsupported)
    assert result == PerceptionInput(heard="已转写的声音")
