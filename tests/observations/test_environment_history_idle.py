from __future__ import annotations

import base64
import hashlib
from uuid import UUID

import pytest
from pydantic import ValidationError

from fish_memory.observations.environment_history_audio import SystemAudioTranscript
from fish_memory.observations.environment_history_models import (
    EnvironmentHistoryWindow,
    SemanticDocument,
)
from fish_memory.observations.environment_history_service import (
    _quiet_visual_digest,
    _render_snapshot_context,
)
from tests.observations.test_environment_history import (
    CAPTURE,
    FISH,
    START,
    Transcriber,
    settle,
    setup_service,
)
from tests.test_environment_frames import JPEG, _request


def quiet_window(index=0, capture=CAPTURE):
    start = START.timestamp() + index * 10
    return EnvironmentHistoryWindow(
        capture_session_id=capture,
        window_index=index,
        started_at=start,
        ended_at=start + 10,
        frames=tuple(
            _request(observed_at=start + second, frame_id=index * 10 + second + 1)
            for second in range(1, 10)
        ),
        audio_status="complete",
        audio_captured_samples=160_000,
        audio_silence=True,
    )


class SilentTranscriber(Transcriber):
    async def transcribe(self, pcm):
        return SystemAudioTranscript("新对白" if pcm.strip(b"\0") else "", "silence")


@pytest.mark.parametrize(
    "patch",
    [
        {"audio_status": "partial"},
        {"audio_status": "unavailable"},
        {"audio_captured_samples": 0},
        {"audio_gap_reason": "missing"},
        {"audio_pcm_base64": base64.b64encode(bytes(320_000)).decode()},
        {"audio_silence": "true"},
    ],
)
def test_silence_marker_rejects_gaps_and_ambiguous_pcm(patch):
    body = quiet_window().model_dump()
    with pytest.raises(ValidationError):
        EnvironmentHistoryWindow.model_validate({**body, **patch})


def test_silent_window_uses_shared_pcm_and_preserves_retry_identity():
    first, second = quiet_window(), quiet_window(1)
    assert first.decode_pcm() is second.decode_pcm()
    assert len(first.decode_pcm()) == 320_000
    assert not first.decode_pcm().strip(b"\0")
    body = first.model_dump(mode="json")
    assert (
        EnvironmentHistoryWindow.model_validate(body).input_digest()
        == first.input_digest()
    )


def test_repeated_image_transport_preserves_each_capture_and_validates_references():
    body = quiet_window().model_dump(mode="json")
    digest = hashlib.sha256(JPEG).hexdigest()
    for index, frame in enumerate(body["frames"]):
        frame["content_hash"] = digest
        if index:
            frame.pop("jpeg_base64")
    compact = EnvironmentHistoryWindow.model_validate(body)
    assert len(compact.frames) == 9
    assert [f.observed_at for f in compact.frames] == [
        f.observed_at for f in quiet_window().frames
    ]
    assert all(f.decode_jpeg() == JPEG for f in compact.frames)
    assert (
        EnvironmentHistoryWindow.model_validate(compact.model_dump()).input_digest()
        == compact.input_digest()
    )
    body["frames"][1]["content_hash"] = "a" * 64
    with pytest.raises(ValidationError):
        EnvironmentHistoryWindow.model_validate(body)
    # A reference cannot resolve against another request or a different Fish.
    body["frames"][0].pop("jpeg_base64")
    with pytest.raises(ValidationError):
        EnvironmentHistoryWindow.model_validate(body)


@pytest.mark.asyncio
async def test_static_silent_minutes_reuse_one_observation_without_holes():
    service, repository, summarizer = setup_service(transcriber=SilentTranscriber())
    try:
        for index in range(12):
            await service.enqueue(FISH, quiet_window(index))
        await settle(service)
        leaves = sorted(
            (s for s in repository.values.values() if s.period_seconds == 10),
            key=lambda s: s.window_index,
        )
        assert [s.window_index for s in leaves] == list(range(12))
        assert len(summarizer.calls) == 1
        assert summarizer.merges == []
        assert all(s.document.unchanged_from == leaves[0].id for s in leaves[1:])
        assert all(not s.document.audio for s in leaves)
        assert {
            s.window_index for s in repository.values.values() if s.period_seconds == 60
        } == set()
        rendered = _render_snapshot_context(leaves)
        assert rendered.count("window 0") == 1
        assert rendered.count("画面保持不变，未听到新的声音") == 1
        context = await service.get_latest_context_summary(FISH)
        assert context is not None and "画面保持不变" in context.summary
        assert all(
            repository.evidence[(FISH, s.id)]["visualReuseFrom"] == str(leaves[0].id)
            for s in leaves[1:]
        )
        assert service.status(FISH)["failedWindows"] == 0
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_subtitle_change_audio_resumption_and_missing_frames_force_analysis():
    service, _, summarizer = setup_service(transcriber=SilentTranscriber())
    try:
        for index in range(8):
            window = quiet_window(index)
            if index == 2:
                changed = window.frames[4].model_copy(
                    update={
                        "jpeg_base64": base64.b64encode(
                            b"\xff\xd8\xffchanged\xff\xd9"
                        ).decode()
                    }
                )
                window = window.model_copy(
                    update={"frames": (*window.frames[:4], changed, *window.frames[5:])}
                )
            if index == 4:
                window = window.model_copy(
                    update={
                        "audio_silence": False,
                        "audio_pcm_base64": base64.b64encode(
                            b"\x01\x00" * 160_000
                        ).decode(),
                    }
                )
            if index == 6:
                window = window.model_copy(update={"frames": window.frames[:2]})
            await service.enqueue(FISH, window)
            await settle(service)
        assert [index for index, _ in summarizer.calls] == [0, 2, 3, 4, 5, 6, 7]
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_failed_baseline_and_new_capture_never_reuse_stale_observation():
    service, _, summarizer = setup_service(transcriber=SilentTranscriber())
    summarizer.fail.add(0)
    try:
        await service.enqueue(FISH, quiet_window())
        await settle(service)
        await service.enqueue(FISH, quiet_window(1))
        await settle(service)
        await service.enqueue(FISH, quiet_window(2, UUID(int=999)))
        await settle(service)
        assert [index for index, _ in summarizer.calls] == [0, 1, 2]
        service.set_enabled(FISH, False)
        assert not service._quiet_visual
        service.set_enabled(FISH, True)
        await service.enqueue(FISH, quiet_window(3, UUID(int=999)))
        await settle(service)
        assert len(summarizer.calls) == 4
    finally:
        await service.close()


def test_static_detection_rejects_missing_audio_and_sparse_observations():
    window = quiet_window()
    assert _quiet_visual_digest(window)
    assert (
        _quiet_visual_digest(window.model_copy(update={"frames": window.frames[3:]}))
        is None
    )
    assert (
        _quiet_visual_digest(window.model_copy(update={"audio_status": "partial"}))
        is None
    )
    assert (
        _quiet_visual_digest(window.model_copy(update={"audio_gap_reason": "gap"}))
        is None
    )
    with pytest.raises(ValidationError):
        SemanticDocument(summary_version=4, unchanged_from=UUID(int=1), audio="旧对白")
