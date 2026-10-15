"""Explicitly enrolled, local-only owner speaker verification.

CAM++ similarity is an experimental admission signal, not authentication or
speaker separation. No method downloads a model except ``prepare``; neither
enrollment nor verification writes or uploads captured audio.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import uuid
import wave
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

from fish_memory.model_catalog import speaker_embedding_urls
from fish_memory.runtime_platform import (
    private_file_is_private,
    private_file_permissions,
)

MODEL_ID = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
MODEL_URL = (
    "https://huggingface.co/csukuangfj/speaker-embedding-models/resolve/main/"
    + MODEL_ID
)
MODEL_SHA256 = "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11"
MODEL_SIZE = 28_281_138
EMBEDDING_DIM = 192
PROFILE_NAME = "owner-voice.json"
PROFILE_VERSION = 2
# Each enrollment keeps its own voiceprint, so a new microphone or room adds
# one instead of replacing the rest; the oldest drops out past this many.
MAX_ENROLLMENTS = 10
SAMPLE_RATE = 16_000
MIN_ENROLL_SECONDS = 6
MAX_ENROLL_SECONDS = 30
MIN_EFFECTIVE_ENROLL_SECONDS = 4.5
# Sentences TA did not recognize, added by the owner: measured on the owner's
# own recordings, 3 s of speech made a voiceprint as good as 4.5 s.
MIN_ADDED_SPEECH_SECONDS = 3.0
# Each voiceprint keeps the speech it came from, so the owner can listen before
# removing one. Local only, owner-readable, deleted with its voiceprint.
CLIPS_DIRECTORY = "clips"
MAX_CLIP_SECONDS = 20
_CLIP_NAME = re.compile(r"[0-9a-f]{32}\.wav")
MIN_VERIFY_SECONDS = 0.3
# Candidate from the sherpa speaker-identification example, not user-calibrated.
DEFAULT_THRESHOLD = 0.5
_BACKENDS = frozenset({"livekit", "sounddevice"})
_PROFILE_MAX_BYTES = 256 * 1024  # up to MAX_ENROLLMENTS x 3 voiceprints
_FRAME_SAMPLES = 480  # WebRTC VAD requires 10, 20, or 30 ms PCM16 frames.
_MIN_RMS = 10 ** (-45 / 20)
_MIN_VERIFY_RMS = 10 ** (-60 / 20)
_MAX_CLIPPED_FRACTION = 0.01


class OwnerVoiceError(RuntimeError):
    """Safe user-facing errors; never include captured audio or embeddings."""


def _backend(value: object) -> str:
    if not isinstance(value, str) or value not in _BACKENDS:
        raise OwnerVoiceError("不支持的主人声纹音频后端")
    return value


def _normalized(value: object) -> NDArray[np.float32]:
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise OwnerVoiceError("声纹向量格式无效，请重新录入") from error
    if vector.shape != (EMBEDDING_DIM,) or not np.isfinite(vector).all():
        raise OwnerVoiceError("声纹向量维度或数值无效，请重新录入")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise OwnerVoiceError("未提取到有效声纹，请重新录入")
    return vector / norm


def _pcm_samples(
    pcm: bytes, *, minimum_seconds: float, minimum_rms: float = _MIN_RMS
) -> NDArray[np.float32]:
    if not isinstance(pcm, bytes) or len(pcm) % 2:
        raise OwnerVoiceError("录音必须是 16 kHz 单声道 PCM16")
    if len(pcm) < SAMPLE_RATE * 2 * minimum_seconds:
        raise OwnerVoiceError("有效录音太短，请继续说话")
    if len(pcm) > SAMPLE_RATE * 2 * MAX_ENROLL_SECONDS:
        raise OwnerVoiceError("录音超过 30 秒上限")
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    if float(np.mean(np.abs(samples) >= 0.995)) > _MAX_CLIPPED_FRACTION:
        raise OwnerVoiceError("录音削顶失真过多，请降低输入音量后重试")
    if float(np.sqrt(np.mean(samples * samples))) < minimum_rms:
        raise OwnerVoiceError("录音音量过低或只有静音，请靠近麦克风后重试")
    return samples


def _speech_samples(pcm: bytes) -> NDArray[np.float32]:
    """Enrollment-only local speech/energy gate; discard silence in memory."""
    samples = _pcm_samples(pcm, minimum_seconds=MIN_ENROLL_SECONDS)
    try:
        detector = import_module("webrtcvad").Vad(2)
    except Exception as error:
        raise OwnerVoiceError("本地录音质检组件不可用") from error
    kept = []
    for offset in range(0, len(samples) - _FRAME_SAMPLES + 1, _FRAME_SAMPLES):
        frame = samples[offset : offset + _FRAME_SAMPLES]
        if float(np.sqrt(np.mean(frame * frame))) < _MIN_RMS:
            continue
        raw = pcm[offset * 2 : (offset + _FRAME_SAMPLES) * 2]
        if detector.is_speech(raw, SAMPLE_RATE):
            kept.append(frame)
    if sum(len(frame) for frame in kept) < SAMPLE_RATE * MIN_EFFECTIVE_ENROLL_SECONDS:
        raise OwnerVoiceError("有效人声不足，请在安静环境连续说几句话后重新录入")
    return np.concatenate(kept)


class _Encoder:
    def __init__(self, model_path: Path) -> None:
        try:
            self.sherpa = import_module("sherpa_onnx")
            config = self.sherpa.SpeakerEmbeddingExtractorConfig(
                model=str(model_path), num_threads=1, provider="cpu", debug=False
            )
            self.extractor = self.sherpa.SpeakerEmbeddingExtractor(config)
        except Exception as error:
            raise OwnerVoiceError("无法加载本地主人声纹模型") from error
        if self.extractor.dim != EMBEDDING_DIM:
            raise OwnerVoiceError("声纹模型维度不兼容，请重新准备模型并录入")
        self._lock = threading.Lock()

    def extract(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        with self._lock:
            stream = self.extractor.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            stream.input_finished()
            if not self.extractor.is_ready(stream):
                raise OwnerVoiceError("有效录音不足以提取声纹")
            try:
                return _normalized(self.extractor.compute(stream))
            except OwnerVoiceError:
                raise
            except Exception as error:
                raise OwnerVoiceError("本地主人声纹推理失败") from error


class OwnerVoiceVerifier:
    """Reusable CPU verifier; callers may execute score_pcm in a worker thread."""

    model_id = MODEL_ID

    def __init__(
        self,
        encoder: _Encoder,
        enrollments: list[list[NDArray[np.float32]]],
        *,
        threshold: float,
    ) -> None:
        self.threshold = threshold
        self._encoder = encoder
        self._manager = encoder.sherpa.SpeakerEmbeddingManager(EMBEDDING_DIM)
        # One speaker per enrollment: each averages its own segments, and the
        # closest enrollment decides, so a second microphone never dilutes the first.
        self._names = [f"owner-{index}" for index in range(len(enrollments))]
        for name, embeddings in zip(self._names, enrollments, strict=True):
            if not self._manager.add(name, [vector.tolist() for vector in embeddings]):
                raise OwnerVoiceError("无法加载已录入的主人声纹")
        self._lock = threading.Lock()

    def score_pcm(self, pcm: bytes) -> float:
        samples = _pcm_samples(
            pcm, minimum_seconds=MIN_VERIFY_SECONDS, minimum_rms=_MIN_VERIFY_RMS
        )
        with self._lock:
            vector = self._encoder.extract(samples).tolist()
            score = max(
                float(self._manager.score(name, vector)) for name in self._names
            )
        if not math.isfinite(score):
            raise OwnerVoiceError("声纹相似度无效")
        return max(-1.0, min(1.0, score))


def _joined_utterances(heard: list[bytes]) -> NDArray[np.float32]:
    """Utterances one after another with a short pause, faded at every edge so
    no splice clicks."""
    pause = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.float32)
    parts: list[NDArray[np.float32]] = []
    for pcm in heard:
        if len(pcm) < 4:
            continue
        audio = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32)
        parts += [_faded(audio / 32768), pause]
    return np.concatenate(parts[:-1]) if parts else np.zeros(0, dtype=np.float32)


def _faded(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    ramp = min(len(audio) // 2, SAMPLE_RATE // 100)  # 10 ms
    if ramp:
        envelope = np.linspace(0, 1, ramp, dtype=np.float32)
        audio = audio.copy()
        audio[:ramp] *= envelope
        audio[-ramp:] *= envelope[::-1]
    return audio


def _audible(audio: NDArray[np.float32]) -> NDArray[np.float32]:
    """Bring speech to a normal listening level and keep a stray pop from
    blaring: typical speech frames to about -20 dBFS, peaks held under 0.9."""
    audio = _faded(np.nan_to_num(audio.astype(np.float32)))
    frames = audio[: len(audio) // 160 * 160].reshape(-1, 160)
    if not len(frames):
        return np.clip(audio, -0.9, 0.9)
    energy = np.sqrt(np.mean(frames * frames, axis=1))
    # A percentile, so a few loud pops do not turn the whole clip down.
    level = float(np.percentile(energy, 75))
    if level > 1e-4:
        audio = audio * min(10.0, 0.1 / level)
    return np.tanh(audio / 0.9) * 0.9


def _validate_enrollment(enrollment: dict[str, Any]) -> None:
    created_at = enrollment.get("createdAt")
    try:
        if not isinstance(created_at, str) or len(created_at) > 64:
            raise ValueError
        if datetime.fromisoformat(created_at).tzinfo is None:
            raise ValueError
    except ValueError as error:
        raise OwnerVoiceError("声纹录入时间无效，请重新录入") from error
    clip = enrollment.get("clip")
    if clip is not None and (
        not isinstance(clip, str) or not _CLIP_NAME.fullmatch(clip)
    ):
        raise OwnerVoiceError("声纹录音文件名无效，请重新录入")
    embeddings = enrollment.get("embeddings")
    if not isinstance(embeddings, list) or not 1 <= len(embeddings) <= 8:
        raise OwnerVoiceError("声纹样本数量无效，请重新录入")
    for embedding in embeddings:
        if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIM:
            raise OwnerVoiceError("声纹向量维度无效，请重新录入")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in embedding
        ):
            raise OwnerVoiceError("声纹向量数值无效，请重新录入")
        vector = _normalized(embedding)
        if not np.allclose(vector, embedding, atol=1e-5, rtol=1e-5):
            raise OwnerVoiceError("声纹向量未归一化，请重新录入")


class OwnerVoiceService:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.model_path = self.directory / MODEL_ID
        self.profile_path = self.directory / PROFILE_NAME

    def _model_ready(self) -> bool:
        try:
            if (
                self.model_path.is_symlink()
                or self.model_path.stat().st_size != MODEL_SIZE
            ):
                return False
            with self.model_path.open("rb") as source:
                return hashlib.file_digest(source, "sha256").hexdigest() == MODEL_SHA256
        except OSError:
            return False

    def _read_profile(self) -> dict[str, Any]:
        try:
            if self.profile_path.is_symlink():
                raise OwnerVoiceError("声纹文件不安全，请重新录入")
            file_status = self.profile_path.stat()
            if (
                not stat.S_ISREG(file_status.st_mode)
                or not private_file_is_private(self.profile_path)
                or file_status.st_size > _PROFILE_MAX_BYTES
            ):
                raise OwnerVoiceError("声纹文件权限或大小无效，请重新录入")
            with self.profile_path.open("rb") as source:
                raw = source.read(_PROFILE_MAX_BYTES + 1)
            if len(raw) > _PROFILE_MAX_BYTES:
                raise OwnerVoiceError("声纹文件过大，请重新录入")
            profile = json.loads(raw)
        except FileNotFoundError as error:
            raise OwnerVoiceError("尚未录入主人声纹") from error
        except (OSError, ValueError) as error:
            raise OwnerVoiceError("声纹文件无法读取，请重新录入") from error
        if not isinstance(profile, dict):
            raise OwnerVoiceError("声纹文件格式无效，请重新录入")
        if profile.get("version") == 1 and type(profile.get("version")) is int:
            # One enrollment in the first format; read it as the only one.
            profile = {
                **{key: value for key, value in profile.items() if key != "embeddings"},
                "version": PROFILE_VERSION,
                "enrollments": [
                    {
                        "createdAt": profile.get("createdAt"),
                        "embeddings": profile.get("embeddings"),
                    }
                ],
            }
        if (
            type(profile.get("version")) is not int
            or profile.get("version") != PROFILE_VERSION
            or profile.get("modelId") != MODEL_ID
            or profile.get("modelSha256") != MODEL_SHA256
            or profile.get("dimension") != EMBEDDING_DIM
            or profile.get("sampleRate") != SAMPLE_RATE
        ):
            raise OwnerVoiceError("声纹模型或格式已变化，请重新录入")
        _backend(profile.get("audioBackend"))
        enrollments = profile.get("enrollments")
        if (
            not isinstance(enrollments, list)
            or not 1 <= len(enrollments) <= MAX_ENROLLMENTS
        ):
            raise OwnerVoiceError("声纹录入次数无效，请重新录入")
        for enrollment in enrollments:
            if not isinstance(enrollment, dict):
                raise OwnerVoiceError("声纹文件格式无效，请重新录入")
            _validate_enrollment(enrollment)
        return profile

    def _write_profile(self, profile: dict[str, Any]) -> None:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".owner-profile-", dir=self.directory
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                private_file_permissions(Path(temporary), destination.fileno())
                json.dump(
                    profile,
                    destination,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.profile_path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def status(self, *, audio_backend: str | None = None) -> dict[str, object]:
        if audio_backend is not None:
            _backend(audio_backend)
        ready = self._model_ready()
        result: dict[str, object] = {
            "enrolled": False,
            "modelReady": ready,
            "modelId": MODEL_ID,
            "createdAt": None,
            "enrollmentCount": 0,
            "audioBackend": None,
            "detail": "尚未录入主人声纹",
        }
        try:
            profile = self._read_profile()
        except OwnerVoiceError as error:
            result["detail"] = str(error)
            return result
        result["createdAt"] = profile["enrollments"][-1]["createdAt"]
        result["enrollmentCount"] = len(profile["enrollments"])
        # When and how each voiceprint came in, so one can be removed on its own.
        result["enrollments"] = [
            {
                "createdAt": item["createdAt"],
                "source": "added" if item.get("source") == "added" else "recorded",
                "clip": item.get("clip"),
            }
            for item in profile["enrollments"]
        ]
        result["audioBackend"] = profile["audioBackend"]
        if audio_backend is not None and profile["audioBackend"] != audio_backend:
            result["detail"] = "音频采集后端已变化，请重新录入主人声纹"
            return result
        result["enrolled"] = ready
        result["detail"] = (
            "主人声纹已录入；相似度阈值尚未按本人样本校准"
            if ready
            else "声纹已保存，但模型缺失或校验失败，请准备模型"
        )
        return result

    def prepare(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, object]:
        if self._model_ready():
            return self.status()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        # A download stopped by a closed process leaves its part behind.
        for leftover in self.directory.glob(".owner-model-*"):
            leftover.unlink(missing_ok=True)
        # Mainland mirror first; every source must match the pinned digest.
        sources = speaker_embedding_urls(MODEL_ID, MODEL_URL)
        failure: OwnerVoiceError | None = None
        for source in sources:
            try:
                return self._prepare_from(source, on_progress, cancel_event)
            except OwnerVoiceError as error:
                failure = error
                if cancel_event is not None and cancel_event.is_set():
                    break
        assert failure is not None
        raise failure

    def _prepare_from(
        self,
        source: str,
        on_progress: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, object]:
        descriptor, temporary = tempfile.mkstemp(
            prefix=".owner-model-", dir=self.directory
        )
        try:
            request = Request(source, headers={"User-Agent": "fish-memory/0.1"})
            digest = hashlib.sha256()
            total = 0
            with os.fdopen(descriptor, "wb") as destination:
                with urlopen(request, timeout=30) as response:
                    # Small reads keep progress moving and cancel prompt.
                    while chunk := response.read(
                        min(256 * 1024, MODEL_SIZE + 1 - total)
                    ):
                        if cancel_event is not None and cancel_event.is_set():
                            raise OwnerVoiceError("下载已取消")
                        total += len(chunk)
                        if total > MODEL_SIZE:
                            raise OwnerVoiceError("下载的声纹模型超过固定大小上限")
                        digest.update(chunk)
                        destination.write(chunk)
                        if on_progress is not None:
                            on_progress(total, MODEL_SIZE)
                if total != MODEL_SIZE or digest.hexdigest() != MODEL_SHA256:
                    raise OwnerVoiceError("声纹模型大小或 SHA256 校验失败")
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.model_path)
        except OwnerVoiceError:
            raise
        except Exception as error:
            raise OwnerVoiceError(
                "准备本地主人声纹模型失败，请检查网络后重试"
            ) from error
        finally:
            Path(temporary).unlink(missing_ok=True)
        return self.status()

    def enroll(
        self,
        pcm: bytes,
        *,
        audio_backend: str,
        cancel_event: threading.Event | None = None,
        append: bool = False,
    ) -> dict[str, object]:
        """Save a voiceprint; `append` keeps earlier ones, up to MAX_ENROLLMENTS."""
        _backend(audio_backend)
        if not self._model_ready():
            raise OwnerVoiceError("声纹模型尚未准备或校验失败")
        samples = _speech_samples(pcm)
        embeddings = self._consistent_embeddings(samples, cancel_event)
        return self._save_enrollment(
            embeddings,
            audio_backend=audio_backend,
            append=append,
            source="recorded",
            # Listen to the recording as it was said, not the speech-only cuts.
            speech=np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768,
            cancel_event=cancel_event,
        )

    def add_speech(
        self,
        speech: bytes,
        *,
        audio_backend: str,
        heard: list[bytes] | None = None,
    ) -> dict[str, object]:
        """Add sentences TA did not recognize, which the owner confirmed as theirs.

        `speech` feeds the voiceprint; `heard` is each utterance whole, which is
        what the owner listens to later.
        """
        _backend(audio_backend)
        if not self._model_ready():
            raise OwnerVoiceError("声纹模型尚未准备或校验失败")
        samples = _pcm_samples(speech, minimum_seconds=MIN_ADDED_SPEECH_SECONDS)
        if len(samples) >= SAMPLE_RATE * MIN_EFFECTIVE_ENROLL_SECONDS:
            embeddings = self._consistent_embeddings(samples, None)
        else:
            embeddings = [_Encoder(self.model_path).extract(samples)]
        return self._save_enrollment(
            embeddings,
            audio_backend=audio_backend,
            append=True,
            source="added",
            speech=_joined_utterances(heard) if heard else samples,
        )

    def _consistent_embeddings(
        self, samples: NDArray[np.float32], cancel_event: threading.Event | None
    ) -> list[NDArray[np.float32]]:
        encoder = _Encoder(self.model_path)
        embeddings = []
        for part in np.array_split(samples, 3):
            _check_cancelled(cancel_event)
            embeddings.append(encoder.extract(part))
        consistency = min(
            float(np.dot(left, right))
            for index, left in enumerate(embeddings)
            for right in embeddings[index + 1 :]
        )
        if consistency < DEFAULT_THRESHOLD:
            raise OwnerVoiceError("录音各片段声纹不一致，请独自连续说话并重新录入")
        return embeddings

    def _save_enrollment(
        self,
        embeddings: list[NDArray[np.float32]],
        *,
        audio_backend: str,
        append: bool,
        source: str,
        speech: NDArray[np.float32],
        cancel_event: threading.Event | None = None,
    ) -> dict[str, object]:
        kept: list[dict[str, Any]] = []
        if append:
            try:
                previous = self._read_profile()
            except OwnerVoiceError:
                previous = None
            if previous is not None and previous["audioBackend"] == audio_backend:
                kept = list(previous["enrollments"])
        while len(kept) >= MAX_ENROLLMENTS:
            # Added sentences go first: a full recording is the better voiceprint.
            added = [i for i, item in enumerate(kept) if item.get("source") == "added"]
            kept.pop(added[0] if added else 0)
        profile = {
            "version": PROFILE_VERSION,
            "modelId": MODEL_ID,
            "modelSha256": MODEL_SHA256,
            "dimension": EMBEDDING_DIM,
            "sampleRate": SAMPLE_RATE,
            "audioBackend": audio_backend,
            "enrollments": [
                *kept,
                {
                    "createdAt": datetime.now(UTC).isoformat(),
                    "source": source,
                    "clip": self._write_clip(speech),
                    "embeddings": [embedding.tolist() for embedding in embeddings],
                },
            ],
        }
        _check_cancelled(cancel_event)
        self._write_profile(profile)
        self._remove_unused_clips(profile)
        return self.status(audio_backend=audio_backend)

    @property
    def clips_path(self) -> Path:
        return self.directory / CLIPS_DIRECTORY

    def _write_clip(self, speech: NDArray[np.float32]) -> str:
        """Save the speech a voiceprint came from, for the owner to listen to."""
        self.clips_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.wav"
        pcm = (_audible(speech[: SAMPLE_RATE * MAX_CLIP_SECONDS]) * 32767).astype("<i2")
        descriptor = os.open(
            self.clips_path / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "wb") as destination:
            private_file_permissions(self.clips_path / name, destination.fileno())
            with wave.open(destination, "wb") as clip:
                clip.setnchannels(1)
                clip.setsampwidth(2)
                clip.setframerate(SAMPLE_RATE)
                clip.writeframes(pcm.tobytes())
        return name

    def _remove_unused_clips(self, profile: dict[str, Any] | None) -> None:
        """Clips of voiceprints that were removed or made room go with them."""
        used = {item.get("clip") for item in (profile or {}).get("enrollments", [])}
        try:
            names = [
                path for path in self.clips_path.iterdir() if path.name not in used
            ]
        except FileNotFoundError:
            return
        for path in names:
            if _CLIP_NAME.fullmatch(path.name):
                path.unlink(missing_ok=True)

    def delete(self, created_at: str | None = None) -> dict[str, object]:
        """Remove every voiceprint, or only the one recorded at `created_at`."""
        if created_at is None:
            self.profile_path.unlink(missing_ok=True)
            self._remove_unused_clips(None)
            return self.status()
        profile = self._read_profile()
        remaining = [
            item for item in profile["enrollments"] if item["createdAt"] != created_at
        ]
        if len(remaining) == len(profile["enrollments"]):
            raise OwnerVoiceError("没有找到这段声纹，可能已经删掉了")
        if not remaining:
            self.profile_path.unlink(missing_ok=True)
            self._remove_unused_clips(None)
        else:
            self._write_profile({**profile, "enrollments": remaining})
            self._remove_unused_clips({"enrollments": remaining})
        return self.status()

    def load_verifier(
        self, *, audio_backend: str, threshold: float = DEFAULT_THRESHOLD
    ) -> OwnerVoiceVerifier:
        _backend(audio_backend)
        if (
            isinstance(threshold, bool)
            or not math.isfinite(threshold)
            or not 0 < threshold < 1
        ):
            raise OwnerVoiceError("声纹相似度阈值必须在 0 和 1 之间")
        if not self._model_ready():
            raise OwnerVoiceError("声纹模型尚未准备或校验失败")
        profile = self._read_profile()
        if profile["audioBackend"] != audio_backend:
            raise OwnerVoiceError("音频采集后端已变化，请重新录入主人声纹")
        encoder = _Encoder(self.model_path)
        return OwnerVoiceVerifier(
            encoder,
            [
                [_normalized(value) for value in enrollment["embeddings"]]
                for enrollment in profile["enrollments"]
            ],
            threshold=threshold,
        )


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise OwnerVoiceError("主人声纹录入已取消")
