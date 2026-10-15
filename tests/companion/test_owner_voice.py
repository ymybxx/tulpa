from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fish_memory.companion import owner_voice as owner


def pcm(seconds: float = 6, *, amplitude: float = 0.2) -> bytes:
    samples = np.arange(round(seconds * owner.SAMPLE_RATE), dtype=np.float32)
    return (np.sin(samples * 0.08) * amplitude * 32767).astype("<i2").tobytes()


@pytest.fixture
def model_bytes(monkeypatch: pytest.MonkeyPatch) -> bytes:
    content = b"test-pinned-model"
    monkeypatch.setattr(owner, "MODEL_SIZE", len(content))
    monkeypatch.setattr(owner, "MODEL_SHA256", hashlib.sha256(content).hexdigest())
    return content


@pytest.fixture
def service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_bytes: bytes
) -> owner.OwnerVoiceService:
    result = owner.OwnerVoiceService(tmp_path / "owner")
    result.directory.mkdir()
    result.model_path.write_bytes(model_bytes)

    class Manager:
        def __init__(self, dim: int) -> None:
            assert dim == owner.EMBEDDING_DIM

        def add(self, name, embeddings):
            assert name.startswith("owner-")
            self.embeddings = getattr(self, "embeddings", {})
            self.embeddings[name] = owner._normalized(np.mean(embeddings, axis=0))
            return True

        def score(self, name, embedding):
            return np.dot(self.embeddings[name], embedding)

    class Encoder:
        def __init__(self, path: Path) -> None:
            assert path == result.model_path
            self.sherpa = SimpleNamespace(SpeakerEmbeddingManager=Manager)

        def extract(self, samples):
            assert len(samples) >= owner.SAMPLE_RATE * owner.MIN_VERIFY_SECONDS
            vector = np.zeros(owner.EMBEDDING_DIM, dtype=np.float32)
            vector[0] = 1
            return vector

    class Vad:
        def __init__(self, mode):
            assert mode == 2

        def is_speech(self, frame, sample_rate):
            assert len(frame) == 960
            assert sample_rate == owner.SAMPLE_RATE
            return True

    monkeypatch.setattr(owner, "_Encoder", Encoder)
    monkeypatch.setattr(owner, "import_module", lambda name: SimpleNamespace(Vad=Vad))
    return result


def test_status_is_read_only_and_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        owner, "urlopen", lambda *args, **kwargs: pytest.fail("download")
    )
    service = owner.OwnerVoiceService(tmp_path / "absent")
    status = service.status()
    assert status["enrolled"] is False
    assert status["modelReady"] is False
    assert status["modelId"] == owner.MODEL_ID
    assert status["createdAt"] is None
    assert not service.directory.exists()
    with pytest.raises(owner.OwnerVoiceError, match="模型"):
        service.load_verifier(audio_backend="livekit")
    assert not service.directory.exists()


def test_prepare_verifies_pinned_bytes_and_uses_atomic_replace(
    tmp_path, monkeypatch, model_bytes
):
    service = owner.OwnerVoiceService(tmp_path / "owner")
    calls = []

    def download(request, timeout):
        assert request.full_url == owner.MODEL_URL
        assert timeout == 30
        calls.append("download")
        return io.BytesIO(model_bytes)

    original_replace = owner.os.replace

    def replace(source, destination):
        assert Path(source).read_bytes() == model_bytes
        if os.name != "nt":
            assert Path(source).stat().st_mode & 0o777 == 0o600
        assert destination == service.model_path
        calls.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(owner, "urlopen", download)
    monkeypatch.setattr(owner.os, "replace", replace)
    assert service.prepare()["modelReady"] is True
    assert calls == ["download", "replace"]
    assert list(service.directory.iterdir()) == [service.model_path]
    service.prepare()
    assert calls == ["download", "replace"]


def test_prepare_reports_progress_and_clears_parts_left_by_a_stopped_download(
    tmp_path, monkeypatch, model_bytes
):
    service = owner.OwnerVoiceService(tmp_path / "owner")
    service.directory.mkdir()
    (service.directory / ".owner-model-old").write_bytes(b"part")
    monkeypatch.setattr(owner, "urlopen", lambda *a, **kw: io.BytesIO(model_bytes))
    progress = []
    assert service.prepare(on_progress=lambda *value: progress.append(value))[
        "modelReady"
    ]
    assert progress[-1] == (len(model_bytes), len(model_bytes))
    assert list(service.directory.iterdir()) == [service.model_path]


def test_a_cancelled_download_stops_and_leaves_nothing_behind(
    tmp_path, monkeypatch, model_bytes
):
    import threading

    service = owner.OwnerVoiceService(tmp_path / "owner")
    cancel = threading.Event()
    cancel.set()
    sources = []

    def download(request, timeout):
        sources.append(request.full_url)
        return io.BytesIO(model_bytes)

    monkeypatch.setattr(owner, "urlopen", download)
    with pytest.raises(owner.OwnerVoiceError, match="取消"):
        service.prepare(cancel_event=cancel)
    # Cancelling does not go on to the next mirror.
    assert len(sources) == 1
    assert list(service.directory.iterdir()) == []


@pytest.mark.parametrize("payload", [b"short", b"a" * 17, b"a" * 18])
def test_prepare_rejects_wrong_hash_size_and_preserves_existing_files(
    service, monkeypatch, payload
):
    service.model_path.write_bytes(b"old-invalid-model")
    service.profile_path.write_text("old profile", encoding="utf-8")
    monkeypatch.setattr(owner, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    with pytest.raises(owner.OwnerVoiceError, match="模型"):
        service.prepare()
    assert service.model_path.read_bytes() == b"old-invalid-model"
    assert service.profile_path.read_text() == "old profile"
    assert sorted(path.name for path in service.directory.iterdir()) == sorted(
        [owner.MODEL_ID, owner.PROFILE_NAME]
    )


def test_enrollment_stores_only_normalized_embeddings_atomically(service):
    status = service.enroll(pcm(), audio_backend="livekit")
    assert status["enrolled"] is True
    assert status["audioBackend"] == "livekit"
    assert status["createdAt"]
    profile = json.loads(service.profile_path.read_text())
    assert set(profile) == {
        "version",
        "modelId",
        "modelSha256",
        "dimension",
        "sampleRate",
        "audioBackend",
        "enrollments",
    }
    assert status["enrollmentCount"] == 1
    [enrollment] = profile["enrollments"]
    assert set(enrollment) == {"createdAt", "source", "clip", "embeddings"}
    assert enrollment["source"] == "recorded"
    assert len(enrollment["embeddings"]) == 3
    assert all(
        np.linalg.norm(vector) == pytest.approx(1)
        for vector in enrollment["embeddings"]
    )
    assert owner.private_file_is_private(service.profile_path)
    assert sorted(path.name for path in service.directory.iterdir()) == sorted(
        [owner.MODEL_ID, owner.PROFILE_NAME, owner.CLIPS_DIRECTORY]
    )
    verifier = service.load_verifier(audio_backend="livekit")
    assert verifier.model_id == owner.MODEL_ID
    assert verifier.threshold == 0.5
    assert verifier.score_pcm(pcm(0.3)) == pytest.approx(1)


@pytest.mark.parametrize(
    "audio",
    [
        b"\0",
        pcm(5.9),
        pcm(30.01),
        bytes(6 * 32_000),
        pcm(6, amplitude=0.001),
        b"\xff\x7f" * (6 * 16_000),
    ],
    ids=["odd-bytes", "too-short", "too-long", "silent", "too-quiet", "clipped"],
)
def test_enrollment_quality_errors_preserve_existing_profile(service, audio):
    service.enroll(pcm(), audio_backend="livekit")
    old_profile = service.profile_path.read_bytes()
    with pytest.raises(owner.OwnerVoiceError):
        service.enroll(audio, audio_backend="livekit")
    assert service.profile_path.read_bytes() == old_profile


def test_enrollment_rejects_non_speech_and_strips_silence(service, monkeypatch):
    detector = SimpleNamespace(is_speech=lambda *args: False)
    monkeypatch.setattr(
        owner, "import_module", lambda name: SimpleNamespace(Vad=lambda mode: detector)
    )
    with pytest.raises(owner.OwnerVoiceError, match="有效人声不足"):
        service.enroll(pcm(), audio_backend="livekit")
    assert not service.profile_path.exists()
    detector.is_speech = lambda *args: True
    speech = owner._speech_samples(bytes(3 * 32_000) + pcm())
    assert 5.9 * 16_000 < len(speech) <= 6 * 16_000


def test_enrollment_rejects_inconsistent_segments_without_replacing_profile(
    service, monkeypatch
):
    service.enroll(pcm(), audio_backend="livekit")
    old_profile = service.profile_path.read_bytes()
    original = owner._Encoder

    class InconsistentEncoder(original):
        def __init__(self, path):
            super().__init__(path)
            self.index = 0

        def extract(self, samples):
            self.index += 1
            vector = np.zeros(owner.EMBEDDING_DIM, dtype=np.float32)
            vector[self.index] = 1
            return vector

    monkeypatch.setattr(owner, "_Encoder", InconsistentEncoder)
    with pytest.raises(owner.OwnerVoiceError, match="不一致"):
        service.enroll(pcm(), audio_backend="livekit")
    assert service.profile_path.read_bytes() == old_profile


def test_cancelled_enrollment_preserves_old_profile(service):
    service.enroll(pcm(), audio_backend="livekit")
    old_profile = service.profile_path.read_bytes()
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(owner.OwnerVoiceError, match="取消"):
        service.enroll(pcm(), audio_backend="livekit", cancel_event=cancelled)
    assert service.profile_path.read_bytes() == old_profile


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("version", 3),
        ("version", True),
        ("modelId", "old"),
        ("modelSha256", "old"),
        ("dimension", 256),
        ("sampleRate", 8000),
        ("createdAt", "invalid"),
        ("audioBackend", []),
        ("embeddings", [[0] * 192] * 3),
        ("embeddings", [[float("nan")] * 192] * 3),
        ("embeddings", [[True] * 192] * 3),
        ("embeddings", [[1] * 192] * 3),
    ],
)
def test_profile_model_format_and_vector_mismatch_fail_closed(service, key, value):
    service.enroll(pcm(), audio_backend="livekit")
    profile = json.loads(service.profile_path.read_text())
    target = (
        profile["enrollments"][0] if key in {"createdAt", "embeddings"} else profile
    )
    target[key] = value
    service.profile_path.write_text(json.dumps(profile))
    assert service.status()["enrolled"] is False
    with pytest.raises(owner.OwnerVoiceError):
        service.load_verifier(audio_backend="livekit")


def test_backend_model_changes_and_missing_profile_fail_closed(service):
    service.enroll(pcm(), audio_backend="livekit")
    assert service.status(audio_backend="sounddevice")["enrolled"] is False
    with pytest.raises(owner.OwnerVoiceError, match="后端"):
        service.load_verifier(audio_backend="sounddevice")
    service.model_path.write_bytes(b"changed-model")
    assert service.status()["enrolled"] is False
    assert service.status()["modelReady"] is False
    with pytest.raises(owner.OwnerVoiceError, match="模型"):
        service.load_verifier(audio_backend="livekit")


@pytest.mark.skipif(os.name == "nt", reason="Unix mode changes and symlinks")
def test_profile_permissions_and_symlinks_fail_closed(service, tmp_path):
    service.enroll(pcm(), audio_backend="livekit")
    service.profile_path.chmod(0o644)
    assert service.status()["enrolled"] is False
    service.profile_path.chmod(0o600)
    original = tmp_path / "original-profile.json"
    service.profile_path.rename(original)
    service.profile_path.symlink_to(original)
    assert service.status()["enrolled"] is False
    with pytest.raises(owner.OwnerVoiceError, match="不安全"):
        service.load_verifier(audio_backend="livekit")
    service.delete()
    assert original.exists()


def test_real_encoder_contract_finishes_input_before_readiness(monkeypatch):
    calls = []

    class Stream:
        def accept_waveform(self, sample_rate, samples):
            assert sample_rate == 16000
            calls.append("audio")

        def input_finished(self):
            calls.append("finished")

    class Extractor:
        dim = owner.EMBEDDING_DIM

        def create_stream(self):
            return Stream()

        def is_ready(self, stream):
            assert calls == ["audio", "finished"]
            calls.append("ready")
            return True

        def compute(self, stream):
            calls.append("compute")
            return [1.0] * owner.EMBEDDING_DIM

    monkeypatch.setattr(
        owner,
        "import_module",
        lambda name: SimpleNamespace(
            SpeakerEmbeddingExtractorConfig=lambda **kw: kw,
            SpeakerEmbeddingExtractor=lambda config: Extractor(),
        ),
    )
    vector = owner._Encoder(Path("unused-model")).extract(
        np.zeros(4800, dtype=np.float32)
    )
    assert calls == ["audio", "finished", "ready", "compute"]
    assert np.linalg.norm(vector) == pytest.approx(1)


def test_delete_removes_profile_only_and_is_idempotent(service, model_bytes):
    service.enroll(pcm(), audio_backend="livekit")
    for _ in range(2):
        status = service.delete()
        assert status["enrolled"] is False
        assert status["modelReady"] is True
    assert service.model_path.read_bytes() == model_bytes
    with pytest.raises(owner.OwnerVoiceError, match="尚未录入"):
        service.load_verifier(audio_backend="livekit")


def test_verification_too_short_or_silent_is_an_error(service):
    service.enroll(pcm(), audio_backend="livekit")
    verifier = service.load_verifier(audio_backend="livekit")
    with pytest.raises(owner.OwnerVoiceError, match="太短"):
        verifier.score_pcm(pcm(0.29))
    with pytest.raises(owner.OwnerVoiceError, match="音量"):
        verifier.score_pcm(bytes(32_000))
    assert verifier.score_pcm(pcm(0.3, amplitude=0.003)) == pytest.approx(1)


def test_first_format_profile_reads_as_one_enrollment(service):
    service.enroll(pcm(), audio_backend="livekit")
    profile = json.loads(service.profile_path.read_text())
    [enrollment] = profile.pop("enrollments")
    service.profile_path.write_text(json.dumps({**profile, "version": 1, **enrollment}))
    status = service.status(audio_backend="livekit")
    assert status["enrolled"] is True
    assert status["enrollmentCount"] == 1
    assert service.load_verifier(audio_backend="livekit").score_pcm(pcm(0.3)) == (
        pytest.approx(1)
    )


def test_append_keeps_earlier_voiceprints_and_the_closest_one_decides(
    service, monkeypatch
):
    directions = iter(range(owner.MAX_ENROLLMENTS + 1))
    current = [0]

    def extract(self, samples):
        vector = np.zeros(owner.EMBEDDING_DIM, dtype=np.float32)
        vector[current[0]] = 1
        return vector

    monkeypatch.setattr(owner._Encoder, "extract", extract)
    for index in directions:
        current[0] = index
        status = service.enroll(pcm(), audio_backend="livekit", append=index > 0)
    # The oldest (direction 0) dropped out past MAX_ENROLLMENTS.
    assert status["enrollmentCount"] == owner.MAX_ENROLLMENTS
    verifier = service.load_verifier(audio_backend="livekit")
    current[0] = 3
    assert verifier.score_pcm(pcm(0.3)) == pytest.approx(1)
    current[0] = 0
    assert verifier.score_pcm(pcm(0.3)) == pytest.approx(0)
    # A plain enrollment starts over with only the new voiceprint.
    assert service.enroll(pcm(), audio_backend="livekit")["enrollmentCount"] == 1


def test_added_near_misses_become_one_more_voiceprint(service):
    service.enroll(pcm(), audio_backend="livekit")
    status = service.add_speech(pcm(3), audio_backend="livekit")
    assert status["enrollmentCount"] == 2
    added = json.loads(service.profile_path.read_text())["enrollments"][-1]
    # Under 4.5 s of speech one voiceprint is taken from all of it.
    assert len(added["embeddings"]) == 1
    service.load_verifier(audio_backend="livekit")
    with pytest.raises(owner.OwnerVoiceError, match="太短"):
        service.add_speech(pcm(2.9), audio_backend="livekit")
    assert service.status()["enrollmentCount"] == 2


def test_added_sentences_make_room_before_recorded_voiceprints(service):
    service.enroll(pcm(), audio_backend="livekit")
    service.enroll(pcm(), audio_backend="livekit", append=True)
    for _ in range(owner.MAX_ENROLLMENTS + 3):
        service.add_speech(pcm(3), audio_backend="livekit")
    sources = [
        item["source"]
        for item in json.loads(service.profile_path.read_text())["enrollments"]
    ]
    assert len(sources) == owner.MAX_ENROLLMENTS
    assert sources[:2] == ["recorded", "recorded"]
    assert sources[2:] == ["added"] * (owner.MAX_ENROLLMENTS - 2)


def test_one_voiceprint_can_be_removed_and_the_last_removes_the_profile(service):
    service.enroll(pcm(), audio_backend="livekit")
    service.add_speech(pcm(3), audio_backend="livekit")
    listed = service.status()["enrollments"]
    assert [item["source"] for item in listed] == ["recorded", "added"]
    status = service.delete(listed[0]["createdAt"])
    assert status["enrollmentCount"] == 1
    assert [item["source"] for item in status["enrollments"]] == ["added"]
    with pytest.raises(owner.OwnerVoiceError, match="没有找到"):
        service.delete(listed[0]["createdAt"])
    assert service.delete(listed[1]["createdAt"])["enrolled"] is False
    assert not service.profile_path.exists()


def test_each_voiceprint_keeps_its_speech_to_listen_to_and_loses_it_on_removal(
    service,
):
    import wave

    service.enroll(pcm(), audio_backend="livekit")
    service.add_speech(pcm(3), audio_backend="livekit")
    listed = service.status()["enrollments"]
    clips = [service.clips_path / item["clip"] for item in listed]
    assert all(owner.private_file_is_private(path) for path in clips)
    with wave.open(str(clips[1])) as clip:
        assert (clip.getframerate(), clip.getnchannels()) == (owner.SAMPLE_RATE, 1)
        assert clip.getnframes() == 3 * owner.SAMPLE_RATE
    service.delete(listed[1]["createdAt"])
    assert [path.exists() for path in clips] == [True, False]
    service.delete()
    assert list(service.clips_path.iterdir()) == []


def test_clips_to_listen_to_have_no_splice_clicks_and_a_normal_level():
    tone = (np.sin(np.arange(16_000) * 0.2) * 0.03 * 32767).astype("<i2").tobytes()
    joined = owner._joined_utterances([tone, tone])
    # A 0.3 s pause between the two, each faded in and out.
    assert len(joined) == 2 * 16_000 + int(0.3 * owner.SAMPLE_RATE)
    assert np.abs(np.diff(joined)).max() < 0.02
    burst = joined.copy()
    burst[5_000:5_030] = 1.0  # a stray pop
    audible = owner._audible(burst)
    assert np.abs(audible).max() <= 0.9
    voiced = audible[1_000:15_000]
    assert 20 * np.log10(np.sqrt(np.mean(voiced * voiced))) == pytest.approx(-20, abs=3)
