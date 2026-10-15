"""Internal entry point bundled inside the desktop application."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import sys
import threading
from importlib import import_module
from pathlib import Path
from typing import Any, NoReturn

from fish_memory.config import Settings
from fish_memory.runtime_platform import shutdown_handlers, with_parent_shutdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fish Memory desktop runtime")
    parser.add_argument(
        "role",
        choices=(
            "api",
            "companion",
            "environment-asr",
            "audio-devices",
            "doctor",
            "owner-voice",
            "facial-test",
        ),
        help="runtime component to start",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    parser.add_argument(
        "--action", choices=("status", "prepare", "enroll", "append", "delete")
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--config-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=int, default=15)
    # With --action delete: only the voiceprint recorded at this time.
    parser.add_argument("--created-at")
    parser.add_argument("--database-smoke", action="store_true")
    return parser


def main() -> None:
    _configure_desktop_numeric_threads()
    _configure_protocol_stdio()
    args = build_parser().parse_args()
    if args.role == "facial-test":
        _run_facial_test(args)
        return
    if args.role == "owner-voice":
        _run_owner_voice(args)
        return
    if args.role == "audio-devices":
        _print_audio_devices()
        return
    if args.role == "doctor":
        if args.database_smoke and args.directory is None:
            raise SystemExit("database smoke requires an isolated --directory")
        _run_doctor(database_root=args.directory if args.database_smoke else None)
        return
    if args.role == "api":
        _run_api(args.log_level)
        return
    if args.role == "environment-asr":
        _run_environment_asr(args.log_level)
        return
    _run_companion(args.log_level)


def _configure_desktop_numeric_threads() -> None:
    if sys.platform == "win32":
        # NumPy's OpenBLAS otherwise starts one worker per logical CPU, each
        # committing a large scratch buffer before any model work. Desktop
        # inference uses ONNX Runtime's own thread limit; set this before role
        # imports load NumPy and preserve an explicitly configured BLAS limit.
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _configure_protocol_stdio() -> None:
    # Electron decodes child pipes as UTF-8. Frozen Python does not necessarily
    # honor the developer interpreter's UTF-8 mode on a Chinese Windows host.
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            configure = getattr(stream, "reconfigure", None)
            if callable(configure):
                configure(encoding="utf-8", errors="backslashreplace")


def _run_facial_test(args: argparse.Namespace) -> None:
    if args.config_directory is None:
        raise SystemExit("facial-test requires --config-directory")
    try:
        from fish_memory.facial_test.service import serve_configured

        _quiet_desktop_transport_logs()
        asyncio.run(
            with_parent_shutdown(serve_configured(args.config_directory, args.output))
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        return
    except Exception as error:
        print(
            json.dumps(
                {"type": "error", "stage": "startup", "message": type(error).__name__}
            ),
            flush=True,
        )
        raise SystemExit(1) from None


def _owner_voice_event(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def _run_owner_voice(args: argparse.Namespace) -> None:
    from fish_memory.companion.owner_voice import OwnerVoiceError

    try:
        if args.action is None or args.directory is None:
            raise OwnerVoiceError("主人声纹操作需要 action 和 directory")
        asyncio.run(_owner_voice_action(args))
    except (asyncio.CancelledError, KeyboardInterrupt):
        _owner_voice_event(
            {"event": "error", "error": "主人声纹操作已取消", "code": "cancelled"}
        )
        raise SystemExit(130) from None
    except Exception as error:
        _owner_voice_event(
            {
                "event": "error",
                "error": str(error)
                if isinstance(error, OwnerVoiceError)
                else "本地主人声纹操作失败",
                "code": type(error).__name__,
            }
        )
        raise SystemExit(1) from None


async def _owner_voice_action(args: argparse.Namespace) -> None:
    from fish_memory.companion.owner_voice import OwnerVoiceError, OwnerVoiceService
    from fish_memory.companion.owner_voice_enrollment import capture_owner_voice

    audio_backend = os.getenv(
        "FISH_MEMORY_COMPANION_AUDIO_BACKEND",
        Settings.model_fields["companion_audio_backend"].default,
    )
    if audio_backend not in {"livekit", "sounddevice"}:
        raise OwnerVoiceError("不支持的主人声纹音频后端")
    service = OwnerVoiceService(args.directory)
    task = asyncio.current_task()
    cancel_event = threading.Event()

    def cancel() -> None:
        if not cancel_event.is_set():
            cancel_event.set()
            if task is not None:
                task.cancel()

    with shutdown_handlers(cancel):
        if args.action in {"prepare", "enroll", "append"}:
            if args.action != "prepare" and not 6 <= args.duration <= 30:
                raise OwnerVoiceError("主人声纹录音时长必须在 6 至 30 秒之间")
            reported = [0]

            def progress(received: int, total: int) -> None:
                # About every megabyte, and once at the end.
                if received - reported[0] >= 1024 * 1024 or received == total:
                    reported[0] = received
                    _owner_voice_event(
                        {"event": "download", "received": received, "total": total}
                    )

            await asyncio.to_thread(
                service.prepare, on_progress=progress, cancel_event=cancel_event
            )
        if args.action in {"enroll", "append"}:
            settings = _owner_voice_capture_settings()
            pcm = await capture_owner_voice(
                settings,
                duration_seconds=args.duration,
                on_recording=lambda: _owner_voice_event(
                    {
                        "event": "recording",
                        "durationSeconds": args.duration,
                    }
                ),
            )
            _owner_voice_event({"event": "processing"})
            try:
                await asyncio.to_thread(
                    service.enroll,
                    pcm,
                    audio_backend=audio_backend,
                    cancel_event=cancel_event,
                    append=args.action == "append",
                )
            finally:
                del pcm
        elif args.action == "delete":
            await asyncio.to_thread(service.delete, args.created_at)
        status = await asyncio.to_thread(service.status, audio_backend=audio_backend)
        _owner_voice_event({"event": "result", "status": status})


def _owner_voice_capture_settings() -> Settings:
    # Enrollment uses only the existing audio-backend settings. Satisfy the
    # shared schema without requiring, reading, or using a real LLM credential;
    # this role never constructs a conversation client or sends an Ask.
    values: dict[str, Any] = {
        "_env_file": None,
        "FISH_MEMORY_LLM_PROVIDER": "openai",
        "FISH_MEMORY_LLM_API_KEY": "owner-voice-local-unused",
        "FISH_MEMORY_LLM_MODEL": "owner-voice-local-unused",
        "FISH_MEMORY_LLM_BASE_URL": "http://127.0.0.1:1",
    }
    return Settings(**values)


def _desktop_settings() -> Settings:
    """Everything comes from the desktop app's environment, never a stray `.env`.

    Settings reads `.env` from the working directory for the command-line
    server. The desktop runtime starts in the repository during development,
    where that file would silently override what the app configured.
    """
    # `_env_file` is a BaseSettings.__init__ argument; ty only sees the fields.
    return Settings(_env_file=None)  # ty: ignore[unknown-argument]


def _run_api(log_level: str) -> NoReturn:
    import uvicorn

    from fish_memory.app import create_app
    from fish_memory.main import (
        configure_application_logging,
        read_server_host,
        read_server_port,
    )

    configure_application_logging()
    _quiet_desktop_transport_logs()
    settings = _desktop_settings()
    application = create_app(settings)
    options: dict[str, Any] = dict(
        host=read_server_host(os.getenv("FISH_MEMORY_HOST")),
        port=read_server_port(os.getenv("FISH_MEMORY_PORT")),
        workers=1,
        log_level=log_level.casefold(),
        access_log=False,
    )
    if sys.platform == "win32" and os.getenv("FISH_MEMORY_PARENT_CONTROL") == "stdin":
        server = uvicorn.Server(uvicorn.Config(application, **options))

        async def serve() -> None:
            # Uvicorn owns OS signal handling; this only watches Electron's pipe.
            with shutdown_handlers(
                lambda: setattr(server, "should_exit", True), signals=False
            ):
                await server.serve()

        asyncio.run(serve())
    else:
        uvicorn.run(application, **options)
    raise SystemExit(0)


def _run_companion(log_level: str) -> None:
    from fish_memory.companion.runner import CompanionRunner

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _quiet_desktop_transport_logs()
    settings = _desktop_settings()

    async def run() -> None:
        runner = CompanionRunner(settings)
        try:
            await runner.run()
        finally:
            await runner.close()

    try:
        # Turning voice off stops this process with SIGTERM. Handled, it closes
        # the session first, so a sentence cut off mid-way is reported as
        # stopped instead of leaving TA's bubble saying it is still speaking.
        asyncio.run(with_parent_shutdown(run(), signals=True))
    except (KeyboardInterrupt, asyncio.CancelledError):
        return


def _quiet_desktop_transport_logs() -> None:
    """Keep high-frequency localhost traffic out of the persistent app log."""
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _run_environment_asr(log_level: str) -> None:
    from fish_memory.companion.environment_asr import run_environment_asr

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _quiet_desktop_transport_logs()
    asyncio.run(run_environment_asr(_desktop_settings()))


def _print_audio_devices() -> None:
    try:
        sounddevice = import_module("sounddevice")
        devices = sounddevice.query_devices()
    except Exception as error:
        raise SystemExit(f"Unable to query audio devices: {error}") from error
    payload = []
    for index, raw in enumerate(devices):
        device = dict(raw)
        payload.append(
            {
                "id": str(index),
                "name": str(device.get("name", index)),
                "inputs": int(device.get("max_input_channels", 0)),
                "outputs": int(device.get("max_output_channels", 0)),
            }
        )
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def _run_doctor(*, database_root: Path | None = None) -> None:
    modules = (
        # Check the API's complete import graph, including bundled resources.
        # Importing the factory does not instantiate Settings, create the app,
        # or enter its lifespan (database, clients, workers and devices).
        "fish_memory.app",
        "aec_audio_processing",
        "asyncpg",
        "onnxruntime",
        "pg0",
        "PIL.Image",
        "PIL.ImageFont",
        "sentencepiece",
        "sherpa_onnx",
        "sounddevice",
        "tokenizers",
        "webrtcvad",
    )
    if _uses_livekit_vad():
        modules += ("livekit.plugins.silero",)
    loaded = []
    for name in modules:
        import_module(name)
        loaded.append(name)
    vad = _probe_local_vad()
    contact_sheet = _probe_contact_sheet()
    report: dict[str, object] = {
        "ok": True,
        "modules": loaded,
        "vad": vad,
        "contactSheet": contact_sheet,
    }
    if database_root is not None:
        report["database"] = asyncio.run(_probe_embedded_database(database_root))
    print(
        json.dumps(
            report,
            separators=(",", ":"),
        )
    )


async def _probe_embedded_database(root: Path) -> dict[str, object]:
    """Package gate restricted to a disposable hosted Windows CI account."""
    from uuid import uuid4

    import asyncpg
    from pg0 import Pg0

    root = await asyncio.to_thread(root.resolve)
    if (
        sys.platform != "win32"
        or os.getenv("FISH_MEMORY_WINDOWS_CI_DATABASE_SMOKE") != "1"
        or os.getenv("GITHUB_ACTIONS") != "true"
        or os.getenv("RUNNER_ENVIRONMENT") != "github-hosted"
    ):
        raise RuntimeError(
            "database smoke requires an explicitly enabled disposable Windows CI "
            "account"
        )
    if not (root / ".fish-memory-package-smoke").is_file():
        raise RuntimeError("database smoke temporary-directory marker is missing")
    data = root / "数据库 space"
    database = Pg0(
        name=f"fish-memory-smoke-{uuid4().hex}",
        data_dir=str(data),
        config={"shared_buffers": "32MB", "maintenance_work_mem": "32MB", "jit": "off"},
    )
    connection = None
    info = None
    distance = None
    try:
        try:
            info = await asyncio.to_thread(database.start)
        except Exception:
            # Failure-only A/B probe in this standalone hosted-CI doctor. Keep
            # the original error; this never retries startup or changes the API.
            try:
                from fish_memory.windows_postgres_diagnostics import (
                    diagnose_postgres_loader,
                )

                diagnostics = await asyncio.to_thread(
                    diagnose_postgres_loader, Path(os.environ["TULPA_PG0_HOME"])
                )
            except Exception as diagnostic_error:
                diagnostics = {"diagnosticError": type(diagnostic_error).__name__}
            print(
                json.dumps({"windowsPostgresLoader": diagnostics}),
                file=sys.stderr,
                flush=True,
            )
            raise
        if not info.running or not info.uri or not info.data_dir:
            raise RuntimeError("packaged pg0 did not start")
        if await asyncio.to_thread(Path(info.data_dir).resolve) != data:
            raise RuntimeError("packaged pg0 escaped the isolated data directory")
        connection = await asyncpg.connect(info.uri, timeout=10, command_timeout=10)
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        distance = await connection.fetchval(
            "SELECT '[1,2,3]'::vector(3) <-> '[1,2,4]'::vector(3)"
        )
        if distance != 1.0:
            raise RuntimeError("packaged pgvector distance self-test failed")
    finally:
        try:
            if connection is not None:
                await connection.close(timeout=5)
        finally:
            await asyncio.to_thread(database.stop)
    status = await asyncio.to_thread(database.info)
    if status.running or (data / "postmaster.pid").exists():
        raise RuntimeError("packaged PostgreSQL did not finish shutting down")
    return {"backend": "pg0", "vectorDistance": distance, "stopped": True}


def _probe_contact_sheet() -> dict[str, object]:
    """Exercise packaged JPEG/font resources without any desktop capture."""
    import base64
    from io import BytesIO
    from uuid import UUID

    from PIL import Image

    from fish_memory.observations.environment_history_contact_sheet import (
        build_contact_sheet,
    )
    from fish_memory.observations.environment_history_models import (
        EnvironmentHistoryWindow,
    )

    frames = []
    for index in range(9):
        output = BytesIO()
        with Image.new("RGB", (320, 180), (index * 25, 32, 64)) as image:
            image.save(output, format="JPEG")
        jpeg = output.getvalue()
        frames.append(
            {
                "frameId": str(UUID(int=index + 1)),
                "revision": index + 1,
                "observedAt": 1.0 + index,
                "expiresAt": 4.0 + index,
                "displayID": 0,
                "pixelWidth": 320,
                "pixelHeight": 180,
                "byteLength": len(jpeg),
                "mimeType": "image/jpeg",
                "jpegBase64": base64.b64encode(jpeg).decode("ascii"),
            }
        )
    window = EnvironmentHistoryWindow.model_validate(
        {
            "captureSessionId": str(UUID(int=1)),
            "windowIndex": 0,
            "startedAt": 1.0,
            "endedAt": 11.0,
            "frames": frames,
            "audioStatus": "unavailable",
            "audioCapturedSamples": 0,
        }
    )
    sheet = build_contact_sheet(window)
    with Image.open(BytesIO(base64.b64decode(sheet.jpeg_base64))) as decoded:
        decoded.load()
        if len(sheet.panels) != 9 or decoded.size != (960, 618):
            raise RuntimeError("Packaged contact-sheet self-test failed")
    return {"panels": 9, "width": sheet.width, "height": sheet.height}


def _probe_local_vad() -> dict[str, object]:
    """Exercise bundled model/resources/worker without opening audio devices."""
    if not _uses_livekit_vad():
        return _probe_sherpa_vad()
    from fish_memory.companion.livekit_vad import LiveKitVad
    from fish_memory.companion.vad import SileroVadConfig

    # The official provider uses its package resource, never this legacy path.
    detector = LiveKitVad(SileroVadConfig(model_dir=Path(".")))
    try:
        before = [detector.is_speech(bytes(320)) for _ in range(16)]
        detector.reset()
        after = [detector.is_speech(bytes(320)) for _ in range(16)]
        if any(before) or any(after):
            raise RuntimeError("Bundled LiveKit VAD rejected digital silence self-test")
    finally:
        detector.close()
    return {"backend": "livekit_silero", "sampleRate": 16000, "frameMs": 10}


def _uses_livekit_vad() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in {
        "arm64",
        "aarch64",
    }


def _probe_sherpa_vad() -> dict[str, object]:
    """Exercise the shipped Windows VAD, without downloads or user state."""
    from fish_memory.companion.vad import SileroVad, SileroVadConfig
    from fish_memory.model_catalog import SILERO_VAD_FILE
    from fish_memory.model_sources import bundled_models_dir, file_matches

    bundled = bundled_models_dir()
    if bundled is None:
        raise RuntimeError("doctor requires FISH_MEMORY_BUNDLED_MODELS_DIR")
    model = bundled / str(SILERO_VAD_FILE.bundled)
    if not file_matches(model, SILERO_VAD_FILE.sha256, SILERO_VAD_FILE.size):
        raise RuntimeError("Bundled Silero VAD is missing or failed verification")
    detector = SileroVad(SileroVadConfig(model_dir=model.parent))
    for _round in range(2):
        if any(detector.is_speech(bytes(1024)) for _ in range(16)):
            raise RuntimeError("Bundled Silero VAD rejected digital silence self-test")
        detector.reset()
    return {"backend": "sherpa_silero", "sampleRate": 16000, "frameMs": 32}


if __name__ == "__main__":
    main()
