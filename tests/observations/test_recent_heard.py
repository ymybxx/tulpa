from datetime import timedelta
from types import SimpleNamespace
from uuid import UUID

import pytest

from fish_memory.observations.environment_history_audio import (
    SystemAudioSegment,
    SystemAudioTranscript,
)
from tests.observations.test_environment_history import (
    FISH,
    START,
    audible_window,
    settle,
    setup_service,
)


class Transcriber:
    async def transcribe(self, pcm):
        return SystemAudioTranscript(
            "跨界的旧话 本段原话 尾部原话",
            "Speech",
            (
                SystemAudioSegment(0.5, 1.5, "跨界的旧话", "声源A"),
                SystemAudioSegment(2, 4, "本段原话", "声源A"),
                SystemAudioSegment(8, 9, "尾部原话", None),
            ),
        )

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_ready_heard_preserves_speakers_and_only_uses_the_requested_interval():
    clock = SimpleNamespace(now=START + timedelta(seconds=10.5))
    service, _, _ = setup_service(transcriber=Transcriber(), now=lambda: clock.now)
    try:
        await service.enqueue(FISH, audible_window())
        await settle(service)
        since, through = START + timedelta(seconds=1), START + timedelta(seconds=11)
        text = service.heard_between(FISH, since, through)
        assert "声源A：本段原话" in text
        assert "[00:01.00—00:03.00]" in text
        assert "说话人未确定：尾部原话" in text
        assert "跨界的旧话" not in text
        assert "画面" not in text
        assert not service.heard_between(FISH, START, START + timedelta(seconds=10))
        assert not service.heard_between(UUID(int=999), since, through)

        # A newly started capture cannot inherit words from the previous one.
        next_window = audible_window(1, capture=UUID(int=333))
        await service.enqueue(FISH, next_window)
        assert not service.heard_between(FISH, since, through)
        service.set_enabled(FISH, False)
        assert not service.heard_between(FISH, since, through)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_unaligned_text_needs_its_whole_source_window_inside_the_range():
    class Plain(Transcriber):
        async def transcribe(self, pcm):
            return SystemAudioTranscript("没有逐句时间的原话", "Speech")

    service, _, _ = setup_service(
        transcriber=Plain(), now=lambda: START + timedelta(seconds=10.5)
    )
    try:
        await service.enqueue(FISH, audible_window())
        await settle(service)
        assert "没有逐句时间的原话" in service.heard_between(
            FISH, START, START + timedelta(seconds=11)
        )
        assert not service.heard_between(
            FISH, START + timedelta(seconds=2), START + timedelta(seconds=12)
        )
    finally:
        await service.close()
