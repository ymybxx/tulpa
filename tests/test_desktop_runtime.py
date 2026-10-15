from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from fish_memory.desktop_runtime import _print_audio_devices, build_parser


def test_windows_doctor_does_not_import_macos_livekit(monkeypatch, capsys):
    import fish_memory.desktop_runtime as runtime

    imported = []

    def unexpected_startup(*args, **kwargs):
        pytest.fail("offline doctor must not configure or start the application")

    def load_module(name):
        imported.append(name)
        return SimpleNamespace(create_app=unexpected_startup)

    monkeypatch.setattr(runtime, "_uses_livekit_vad", lambda: False)
    monkeypatch.setattr(runtime, "import_module", load_module)
    monkeypatch.setattr(runtime, "Settings", unexpected_startup)
    monkeypatch.setattr(runtime, "_run_api", unexpected_startup)
    monkeypatch.setattr(runtime, "_probe_embedded_database", unexpected_startup)
    monkeypatch.setattr(
        runtime, "_probe_local_vad", lambda: {"backend": "sherpa_silero"}
    )
    monkeypatch.setattr(runtime, "_probe_contact_sheet", lambda: {"panels": 9})
    runtime._run_doctor()
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["modules"] == imported
    assert imported[0] == "fish_memory.app"
    assert "livekit.plugins.silero" not in imported
    assert {"pg0", "onnxruntime", "sherpa_onnx", "aec_audio_processing"} <= set(
        imported
    )


@pytest.mark.parametrize(
    "error",
    [
        UnicodeDecodeError("gbk", b"\xff", 0, 1, "invalid resource encoding"),
        ModuleNotFoundError("missing application dependency"),
    ],
    ids=["bundled-resource-encoding", "missing-application-module"],
)
def test_doctor_rejects_application_import_failure_before_other_probes(
    monkeypatch, capsys, error
):
    import fish_memory.desktop_runtime as runtime

    def load_module(name):
        assert name == "fish_memory.app"
        raise error

    def unexpected_probe(*args, **kwargs):
        pytest.fail("application import failure must stop doctor immediately")

    monkeypatch.setattr(runtime, "_uses_livekit_vad", lambda: False)
    monkeypatch.setattr(runtime, "import_module", load_module)
    monkeypatch.setattr(runtime, "_probe_local_vad", unexpected_probe)
    monkeypatch.setattr(runtime, "_probe_contact_sheet", unexpected_probe)
    monkeypatch.setattr(runtime, "_probe_embedded_database", unexpected_probe)

    with pytest.raises(type(error)) as failure:
        runtime._run_doctor()
    assert failure.value is error
    assert capsys.readouterr().out == ""


def test_windows_doctor_requires_verified_bundled_model_without_download(
    monkeypatch, tmp_path
):
    import fish_memory.desktop_runtime as runtime

    monkeypatch.setattr(runtime, "_uses_livekit_vad", lambda: False)
    monkeypatch.setenv("FISH_MEMORY_BUNDLED_MODELS_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="missing or failed verification"):
        runtime._probe_local_vad()


def test_windows_doctor_exercises_sherpa_silence_and_reset(monkeypatch, tmp_path):
    import fish_memory.desktop_runtime as runtime

    calls = []

    class Detector:
        def __init__(self, config):
            assert config.model_dir == tmp_path / "vad"

        def is_speech(self, pcm):
            assert pcm == bytes(1024)
            calls.append("frame")
            return False

        def reset(self):
            calls.append("reset")

    monkeypatch.setattr(runtime, "_uses_livekit_vad", lambda: False)
    monkeypatch.setenv("FISH_MEMORY_BUNDLED_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("fish_memory.model_sources.file_matches", lambda *a: True)
    monkeypatch.setattr("fish_memory.companion.vad.SileroVad", Detector)
    assert runtime._probe_local_vad()["backend"] == "sherpa_silero"
    assert calls == [*(["frame"] * 16), "reset", *(["frame"] * 16), "reset"]


def test_contact_sheet_doctor_exercises_packaged_jpeg_and_font_path():
    from fish_memory.desktop_runtime import _probe_contact_sheet

    assert _probe_contact_sheet() == {"panels": 9, "width": 960, "height": 618}


def test_database_smoke_refuses_an_ordinary_developer_account(monkeypatch, tmp_path):
    import fish_memory.desktop_runtime as runtime

    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.delenv("FISH_MEMORY_WINDOWS_CI_DATABASE_SMOKE", raising=False)
    with pytest.raises(RuntimeError, match="disposable Windows CI"):
        asyncio.run(runtime._probe_embedded_database(tmp_path))


@pytest.mark.asyncio
@pytest.mark.parametrize("query_fails", [False, True])
async def test_database_smoke_isolated_query_always_closes_connection_and_pg0(
    monkeypatch, tmp_path, query_fails
):
    import fish_memory.desktop_runtime as runtime

    calls = []
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setenv("FISH_MEMORY_WINDOWS_CI_DATABASE_SMOKE", "1")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("RUNNER_ENVIRONMENT", "github-hosted")
    (tmp_path / ".fish-memory-package-smoke").touch()

    class Database:
        def __init__(self, *, name, data_dir, config):
            assert name.startswith("fish-memory-smoke-")
            assert data_dir == str(tmp_path / "数据库 space")

        def start(self):
            calls.append("start")
            return SimpleNamespace(
                running=True,
                uri="postgresql://synthetic",
                data_dir=str(tmp_path / "数据库 space"),
            )

        def stop(self):
            calls.append("stop")

        def info(self):
            return SimpleNamespace(running=False)

    class Connection:
        async def execute(self, query):
            assert query == "CREATE EXTENSION IF NOT EXISTS vector"
            calls.append("extension")

        async def fetchval(self, query):
            assert "::vector(3) <->" in query
            if query_fails:
                raise RuntimeError("synthetic query failure")
            return 1.0

        async def close(self, **kwargs):
            assert kwargs == {"timeout": 5}
            calls.append("connection-close")

    async def connect(uri, **kwargs):
        assert uri == "postgresql://synthetic"
        return Connection()

    monkeypatch.setattr("pg0.Pg0", Database)
    monkeypatch.setattr("asyncpg.connect", connect)
    if query_fails:
        with pytest.raises(RuntimeError, match="synthetic query"):
            await runtime._probe_embedded_database(tmp_path)
    else:
        assert await runtime._probe_embedded_database(tmp_path) == {
            "backend": "pg0",
            "vectorDistance": 1.0,
            "stopped": True,
        }
    assert calls == ["start", "extension", "connection-close", "stop"]


def test_desktop_runtime_parser_accepts_supported_roles() -> None:
    parser = build_parser()

    assert parser.parse_args(["api"]).role == "api"
    assert parser.parse_args(["companion"]).role == "companion"
    assert parser.parse_args(["environment-asr"]).role == "environment-asr"
    assert parser.parse_args(["audio-devices"]).role == "audio-devices"
    assert parser.parse_args(["doctor"]).role == "doctor"
    assert (
        parser.parse_args(
            ["owner-voice", "--action", "status", "--directory", "/tmp/owner-test"]
        ).role
        == "owner-voice"
    )


def test_audio_devices_prints_stable_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class SoundDevice:
        @staticmethod
        def query_devices() -> list[dict[str, object]]:
            return [
                {
                    "name": "MacBook Microphone",
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                }
            ]

    monkeypatch.setattr(
        "fish_memory.desktop_runtime.import_module",
        lambda _name: SoundDevice,
    )

    _print_audio_devices()

    assert json.loads(capsys.readouterr().out) == [
        {
            "id": "0",
            "name": "MacBook Microphone",
            "inputs": 1,
            "outputs": 0,
        }
    ]


def test_local_vad_doctor_uses_only_synthetic_silence_and_closes_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fish_memory.desktop_runtime import _probe_local_vad

    monkeypatch.setattr("fish_memory.desktop_runtime._uses_livekit_vad", lambda: True)

    calls = []

    class Detector:
        def __init__(self, config) -> None:
            calls.append("open")

        def is_speech(self, pcm: bytes) -> bool:
            assert pcm == bytes(320)
            calls.append("frame")
            return False

        def reset(self) -> None:
            calls.append("reset")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr("fish_memory.companion.livekit_vad.LiveKitVad", Detector)
    assert _probe_local_vad() == {
        "backend": "livekit_silero",
        "sampleRate": 16000,
        "frameMs": 10,
    }
    assert calls == ["open", *(["frame"] * 16), "reset", *(["frame"] * 16), "close"]


def test_local_vad_doctor_closes_worker_on_failed_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fish_memory.desktop_runtime import _probe_local_vad

    monkeypatch.setattr("fish_memory.desktop_runtime._uses_livekit_vad", lambda: True)

    calls = []

    class Detector:
        def __init__(self, config) -> None:
            pass

        def is_speech(self, pcm: bytes) -> bool:
            raise RuntimeError("inference failed")

        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr("fish_memory.companion.livekit_vad.LiveKitVad", Detector)
    with pytest.raises(RuntimeError, match="inference failed"):
        _probe_local_vad()
    assert calls == ["close"]


@pytest.mark.parametrize("action", ["status", "prepare", "enroll", "append", "delete"])
def test_owner_voice_cli_emits_json_protocol_without_llm_configuration(
    monkeypatch, capsys, tmp_path, action
):
    from fish_memory.desktop_runtime import _run_owner_voice

    calls = []
    monkeypatch.chdir(tmp_path)
    for name in tuple(os.environ):
        if name.startswith("FISH_MEMORY_LLM_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FISH_MEMORY_COMPANION_AUDIO_BACKEND", "livekit")
    monkeypatch.setenv("FISH_MEMORY_LIVEKIT_AUDIO_HELPER", "/unused/test-helper")
    records = action in {"enroll", "append"}
    if not records:
        monkeypatch.setattr(
            "fish_memory.desktop_runtime._owner_voice_capture_settings",
            lambda: pytest.fail("non-enrollment actions must not build Settings"),
        )

    class Service:
        def __init__(self, directory):
            assert directory == tmp_path

        def prepare(self, *, on_progress, cancel_event):
            assert not cancel_event.is_set()
            on_progress(28, 28)
            calls.append("prepare")

        def enroll(self, pcm, *, audio_backend, cancel_event, append):
            assert pcm == b"synthetic-in-memory-only"
            assert audio_backend == "livekit"
            assert not cancel_event.is_set()
            assert append is (action == "append")
            calls.append("enroll")

        def delete(self, created_at=None):
            assert created_at is None
            calls.append("delete")

        def status(self, *, audio_backend):
            assert audio_backend == "livekit"
            calls.append("status")
            return {
                "enrolled": False,
                "modelReady": True,
                "modelId": "test",
                "createdAt": None,
                "detail": "test",
                "audioBackend": "livekit",
            }

    async def capture(settings, *, duration_seconds, on_recording):
        assert duration_seconds == 15
        assert settings.companion_audio_backend == "livekit"
        assert settings.livekit_audio_helper == "/unused/test-helper"
        assert settings.llm_api_key.get_secret_value() == "owner-voice-local-unused"
        assert settings.llm_base_url == "http://127.0.0.1:1"
        assert calls == ["prepare"]
        on_recording()
        calls.append("capture")
        return b"synthetic-in-memory-only"

    monkeypatch.setattr("fish_memory.companion.owner_voice.OwnerVoiceService", Service)
    monkeypatch.setattr(
        "fish_memory.companion.owner_voice_enrollment.capture_owner_voice", capture
    )
    _run_owner_voice(
        build_parser().parse_args(
            ["owner-voice", "--action", action, "--directory", str(tmp_path)]
        )
    )
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["event"] == "result"
    assert events[-1]["status"]["audioBackend"] == "livekit"
    download = {"event": "download", "received": 28, "total": 28}
    if records:
        assert events[:-1] == [
            download,
            {"event": "recording", "durationSeconds": 15},
            {"event": "processing"},
        ]
        assert calls == ["prepare", "capture", "enroll", "status"]
    else:
        assert events[:-1] == ([download] if action == "prepare" else [])
        assert calls == ([action] if action != "status" else []) + ["status"]


def test_owner_voice_capture_settings_keep_existing_audio_schema_without_llm(
    monkeypatch, tmp_path
):
    from fish_memory.desktop_runtime import _owner_voice_capture_settings

    monkeypatch.chdir(tmp_path)
    for name in tuple(os.environ):
        if name.startswith("FISH_MEMORY_LLM_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("FISH_MEMORY_COMPANION_AUDIO_BACKEND", "sounddevice")
    monkeypatch.setenv("FISH_MEMORY_COMPANION_INPUT_DEVICE", "test-input")
    monkeypatch.setenv("FISH_MEMORY_COMPANION_AEC_DELAY_MS", "73")
    settings = _owner_voice_capture_settings()
    assert settings.companion_audio_backend == "sounddevice"
    assert settings.companion_input_device == "test-input"
    assert settings.companion_aec_delay_ms == 73
    assert settings.llm_model == "owner-voice-local-unused"


def test_owner_voice_cli_hides_untrusted_error_details(monkeypatch, capsys, tmp_path):
    from fish_memory.desktop_runtime import _run_owner_voice

    async def fail(args):
        raise RuntimeError("must-not-leak-pcm-or-embedding")

    monkeypatch.setattr("fish_memory.desktop_runtime._owner_voice_action", fail)
    with pytest.raises(SystemExit) as error:
        _run_owner_voice(
            build_parser().parse_args(
                ["owner-voice", "--action", "status", "--directory", str(tmp_path)]
            )
        )
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert "must-not-leak" not in captured.out + captured.err
    assert json.loads(captured.out)["event"] == "error"


def test_owner_voice_cli_cancellation_has_nonzero_json_result(
    monkeypatch, capsys, tmp_path
):
    import asyncio

    from fish_memory.desktop_runtime import _run_owner_voice

    async def cancel(args):
        raise asyncio.CancelledError

    monkeypatch.setattr("fish_memory.desktop_runtime._owner_voice_action", cancel)
    with pytest.raises(SystemExit) as error:
        _run_owner_voice(
            build_parser().parse_args(
                ["owner-voice", "--action", "enroll", "--directory", str(tmp_path)]
            )
        )
    assert error.value.code == 130
    assert json.loads(capsys.readouterr().out)["code"] == "cancelled"


@pytest.mark.asyncio
async def test_owner_voice_cli_sigterm_cancels_capture_and_restores_handlers(
    monkeypatch, tmp_path
):
    import asyncio
    import signal

    from fish_memory.desktop_runtime import _owner_voice_action

    monkeypatch.setattr(
        "fish_memory.runtime_platform.sys",
        SimpleNamespace(platform="darwin", stdin=None),
    )

    monkeypatch.setattr(
        "fish_memory.desktop_runtime._owner_voice_capture_settings",
        lambda: SimpleNamespace(companion_audio_backend="livekit"),
    )
    monkeypatch.setenv("FISH_MEMORY_COMPANION_AUDIO_BACKEND", "livekit")
    callbacks = {}
    restored = []
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(
        loop,
        "add_signal_handler",
        lambda number, callback: callbacks.__setitem__(number, callback),
    )
    monkeypatch.setattr(loop, "remove_signal_handler", lambda number: True)
    monkeypatch.setattr(
        "fish_memory.runtime_platform.signal.signal",
        lambda number, previous: restored.append(number),
    )
    started = asyncio.Event()
    closed = asyncio.Event()

    class Service:
        def __init__(self, directory):
            pass

        def prepare(self, **kwargs):
            pass

        def enroll(self, *args, **kwargs):
            pytest.fail("cancelled capture must not enroll")

    async def capture(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr("fish_memory.companion.owner_voice.OwnerVoiceService", Service)
    monkeypatch.setattr(
        "fish_memory.companion.owner_voice_enrollment.capture_owner_voice", capture
    )
    args = build_parser().parse_args(
        ["owner-voice", "--action", "enroll", "--directory", str(tmp_path)]
    )
    task = asyncio.create_task(_owner_voice_action(args))
    await started.wait()
    callbacks[signal.SIGTERM]()
    callbacks[signal.SIGTERM]()  # repeated termination must not cancel cleanup twice
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()
    assert set(restored) == {signal.SIGTERM, signal.SIGINT}


@pytest.mark.parametrize(
    ("platform", "configured", "expected"),
    [
        ("win32", None, "1"),
        ("win32", "6", "6"),
        ("darwin", None, None),
        ("linux", "8", "8"),
    ],
    ids=["windows-default", "windows-explicit", "macos-unset", "linux-explicit"],
)
def test_desktop_blas_default_precedes_role_imports(
    monkeypatch, platform, configured, expected
):
    import fish_memory.desktop_runtime as runtime

    # Register restoration even when the caller originally has no BLAS setting.
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", configured or "")
    if configured is None:
        monkeypatch.delenv("OPENBLAS_NUM_THREADS")
    monkeypatch.setattr(runtime, "sys", SimpleNamespace(platform=platform))
    observed = []

    def observe(stage):
        observed.append((stage, os.environ.get("OPENBLAS_NUM_THREADS")))

    def parse_args():
        observe("parse")
        return SimpleNamespace(role="api", log_level="INFO")

    def run_api(log_level):
        assert log_level == "INFO"
        # The role imports NumPy/model dependencies only after this boundary.
        observe("role-import")

    monkeypatch.setattr(runtime, "_configure_protocol_stdio", lambda: observe("stdio"))
    monkeypatch.setattr(
        runtime, "build_parser", lambda: SimpleNamespace(parse_args=parse_args)
    )
    monkeypatch.setattr(runtime, "_run_api", run_api)
    runtime.main()
    assert observed == [
        (stage, expected) for stage in ("stdio", "parse", "role-import")
    ]
