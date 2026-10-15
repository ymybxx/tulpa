"""JSONL subprocess and batch runner, independent of Fish API/ASR/perception.

Start: .venv/bin/python -m scripts.facial_demo.service --output output/facial-demo
Batch: append --batch (only the eight synthetic cases in cases.json are sent).
All stdout lines are protocol JSON; credentials are never logged.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import sys
import time
import wave
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .core import (
    CASES,
    active_channels,
    parameter_list,
    questions,
    reasoning_prompt,
    split_segments,
    standard_parameters,
)
from .providers import ProviderFailure, Providers, load_configuration
from .runtime_speech import load_runtime_speech

Emit = Callable[[dict[str, Any]], None]


class AudioCache:
    """Bounded process-local PCM cache; only fully completed synthesis is cached."""

    def __init__(self, maximum_bytes: int = 32 * 1024 * 1024):
        self.maximum_bytes = maximum_bytes
        self.entries: OrderedDict[str, bytes] = OrderedDict()
        self.size = 0

    def get(self, text: str) -> bytes | None:
        value = self.entries.get(text)
        if value is not None:
            self.entries.move_to_end(text)
        return value

    def put(self, text: str, pcm: bytes) -> None:
        if len(pcm) > self.maximum_bytes:
            return
        previous = self.entries.pop(text, b"")
        self.size -= len(previous)
        self.entries[text] = pcm
        self.size += len(pcm)
        while self.size > self.maximum_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.size -= len(evicted)


def save_wav(path: Path, pcm: bytes, sample_rate: int) -> None:
    if len(pcm) % 2:
        raise ValueError("TTS返回非完整16位PCM采样")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


class DemoRunner:
    def __init__(self, providers: Providers, output: Path | None = None):
        self.providers = providers
        self.output = output.resolve() if output else None
        self.cache = AudioCache()

    async def run(self, request: dict[str, Any], emit: Emit) -> None:
        request_id = request.get("id")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 120:
            raise ValueError("run需要1到120字符的字符串id")
        mode = request.get("mode", "compare")
        if mode not in {"jev", "reasoning", "compare"}:
            raise ValueError("mode必须为jev、reasoning或compare")
        segments = split_segments(request.get("text"))
        character = request.get("character", "当前宠物")
        if (
            not isinstance(character, str)
            or not character.strip()
            or len(character) > 200
        ):
            raise ValueError("character需要1到200字符的模型名称")
        channels = active_channels(request.get("parameters"))
        modes = ("jev", "reasoning") if mode == "compare" else (mode,)
        started = time.perf_counter()
        output = None
        if self.output:
            safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", request_id)[:70]
            output = self.output / f"{safe_id}-{time.time_ns()}"
            output.mkdir(parents=True, exist_ok=False)
            (output / "rubric.json").write_text(
                json.dumps(
                    {
                        "activeChannels": parameter_list(channels),
                        "jevQuestions": questions(channels),
                        "reasoningSystemPrompt": reasoning_prompt(channels),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            (output / "request.json").write_text(
                json.dumps(request, ensure_ascii=False, indent=2)
            )
        journal = (output / "events.jsonl").open("w") if output else None
        errors = 0
        pcm_by_segment: dict[int, bytes] = {}

        def send(event: dict[str, Any]) -> None:
            event = {
                "id": request_id,
                "runElapsedMs": round((time.perf_counter() - started) * 1000, 2),
                **event,
            }
            if journal:
                stored = dict(event)
                if "pcm" in stored:
                    stored["pcmBytes"] = len(base64.b64decode(stored.pop("pcm")))
                journal.write(json.dumps(stored, ensure_ascii=False) + "\n")
                journal.flush()
            emit(event)

        def failure(error: Exception, stage: str, index: int, elapsed: float) -> None:
            nonlocal errors
            errors += 1
            send(
                {
                    "type": "error",
                    "stage": stage,
                    "segmentIndex": index,
                    "message": str(error)
                    if isinstance(error, ProviderFailure)
                    else type(error).__name__,
                    "raw": error.raw if isinstance(error, ProviderFailure) else None,
                    "elapsedMs": round(elapsed * 1000, 2),
                }
            )

        async def predict(selected: str, index: int, state: dict[str, Any]) -> None:
            began = time.perf_counter()
            try:
                prediction = await self.providers.face(selected, state, channels)
                send(
                    {
                        "type": "face",
                        "segmentIndex": index,
                        "text": segments[index],
                        "state": state,
                        **prediction,
                    }
                )
            except Exception as error:
                failure(error, selected, index, time.perf_counter() - began)

        audio_lock = asyncio.Lock()

        async def synthesize(index: int) -> None:
            async with audio_lock:
                began = time.perf_counter()
                pcm = bytearray()
                cached = self.cache.get(segments[index])
                cache_hit = cached is not None
                first_audio_ms = None

                def chunk(data: bytes) -> None:
                    nonlocal first_audio_ms
                    if first_audio_ms is None:
                        first_audio_ms = round((time.perf_counter() - began) * 1000, 2)
                    pcm.extend(data)
                    send(
                        {
                            "type": "audio",
                            "segmentIndex": index,
                            "pcm": base64.b64encode(data).decode("ascii"),
                            "sampleRate": self.providers.configuration.sample_rate,
                            "format": "s16le",
                            "channels": 1,
                            "cacheHit": cache_hit,
                            "elapsedMs": round((time.perf_counter() - began) * 1000, 2),
                        }
                    )

                try:
                    if cached is not None:
                        chunk(cached)
                    else:
                        async for data in self.providers.audio(segments[index]):
                            chunk(data)
                    if not pcm or len(pcm) % 2:
                        raise ProviderFailure("TTS没有返回完整16位PCM音频")
                    pcm_by_segment[index] = bytes(pcm)
                    self.cache.put(segments[index], bytes(pcm))
                    wav_path = output / f"segment-{index:02d}.wav" if output else None
                    if wav_path:
                        save_wav(
                            wav_path,
                            bytes(pcm),
                            self.providers.configuration.sample_rate,
                        )
                    send(
                        {
                            "type": "audio_end",
                            "segmentIndex": index,
                            "sampleRate": self.providers.configuration.sample_rate,
                            "durationMs": round(
                                len(pcm)
                                / 2
                                / self.providers.configuration.sample_rate
                                * 1000,
                                2,
                            ),
                            "elapsedMs": round((time.perf_counter() - began) * 1000, 2),
                            "firstAudioMs": first_audio_ms,
                            "cacheHit": cache_hit,
                            "connectionState": "cache" if cache_hit else "cold",
                            "wavPath": str(wav_path) if wav_path else None,
                            "sha256": hashlib.sha256(pcm).hexdigest(),
                        }
                    )
                except Exception as error:
                    failure(error, "tts", index, time.perf_counter() - began)

        async def segment(index: int) -> None:
            # Model state is expression text only; renderer values stay local.
            state = {
                "previousSegments": segments[:index],
                "currentSegment": segments[index],
            }
            send(
                {
                    "type": "segment",
                    "segmentIndex": index,
                    "text": segments[index],
                    "state": state,
                }
            )
            async with asyncio.TaskGroup() as group:
                for selected in modes:
                    group.create_task(predict(selected, index, state))

        try:
            send(
                {
                    "type": "started",
                    "mode": mode,
                    "segments": segments,
                    "activeChannels": parameter_list(channels),
                    "parameterSource": request.get(
                        "parameterSource", "explicit_model_parameters"
                    ),
                    "configuration": self.providers.configuration.public(),
                    "artifactDirectory": str(output) if output else None,
                }
            )
            # Face inference has a two-clause bound. Audio has its own sequential
            # pipeline, so even two slow face requests cannot stall later speech.
            limit = asyncio.Semaphore(2)

            async def limited(index: int) -> None:
                async with limit:
                    await segment(index)

            async def speech_pipeline() -> None:
                for index in range(len(segments)):
                    await synthesize(index)

            async with asyncio.TaskGroup() as group:
                group.create_task(speech_pipeline())
                for index in range(len(segments)):
                    group.create_task(limited(index))
            if output and len(pcm_by_segment) == len(segments):
                save_wav(
                    output / "audio.wav",
                    b"".join(pcm_by_segment[i] for i in range(len(segments))),
                    self.providers.configuration.sample_rate,
                )
            send(
                {
                    "type": "done",
                    "status": "partial_error" if errors else "completed",
                    "errors": errors,
                }
            )
        except asyncio.CancelledError:
            send({"type": "done", "status": "stopped", "errors": errors})
            raise
        finally:
            if journal:
                journal.close()


class ConfiguredRunner:
    """One isolated provider lifecycle per run, using current desktop settings."""

    def __init__(
        self,
        directory: Path,
        output: Path | None = None,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
    ):
        self.directory = directory
        self.output = output
        self.runner: DemoRunner | None = None
        self.cache = AudioCache()
        self.configuration = None
        self.api_url = api_url
        self.api_key = api_key

    async def run(self, request: dict[str, Any], emit: Emit) -> None:
        if request.get("mode", "jev") != "jev":
            raise ValueError("模型界面表情测试仅支持JEV")
        speech = None
        if self.api_url is not None or self.api_key is not None:
            if not self.api_url or not self.api_key:
                raise ValueError("表情测试连接的本地语音服务尚未就绪")
            speech = await load_runtime_speech(self.api_url, self.api_key)
        configuration = load_configuration(
            self.directory, require_reasoning=False, runtime_speech=speech
        )
        if configuration != self.configuration:
            # Never reuse a previous voice's PCM after model or voice settings change.
            self.cache = AudioCache()
            self.configuration = configuration
        # The look editor's test shows the face as TA would make it, varied.
        providers = Providers(configuration, vary=True)
        runner = DemoRunner(providers, self.output)
        runner.cache = self.cache
        self.runner = runner
        try:
            await runner.run({**request, "mode": "jev"}, emit)
        finally:
            self.runner = None
            await providers.close()


class JsonlService:
    def __init__(self, runner: DemoRunner | ConfiguredRunner, emit: Emit):
        self.runner = runner
        self.emit = emit
        self.task: asyncio.Task[None] | None = None
        self.request_id: str | None = None
        self.terminal_sent = False

    async def stop(self) -> None:
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            # Cancellation can arrive before execute() starts and before the
            # runner has installed its cancellation handler.
            if not self.terminal_sent:
                self.emit({"id": self.request_id, "type": "done", "status": "stopped"})
                self.terminal_sent = True
        self.task = None
        self.request_id = None

    async def dispatch(self, request: dict[str, Any]) -> None:
        request_id = request.get("id")
        operation = request.get("operation")
        if operation == "run":
            await self.stop()
            self.request_id = request_id
            self.terminal_sent = False

            def relay(event: dict[str, Any]) -> None:
                if event.get("type") == "done":
                    self.terminal_sent = True
                self.emit(event)

            async def execute() -> None:
                try:
                    await self.runner.run(request, relay)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self.emit(
                        {
                            "id": request_id,
                            "type": "error",
                            "stage": "run",
                            "message": str(error)
                            if isinstance(error, ValueError)
                            else type(error).__name__,
                        }
                    )
                    relay({"id": request_id, "type": "done", "status": "failed"})

            self.task = asyncio.create_task(execute())
        elif operation == "stop":
            if request_id == self.request_id:
                await self.stop()
            else:
                self.emit({"id": request_id, "type": "done", "status": "not_running"})
        elif operation == "face_state":
            # Older standalone clients may still send feedback. Never forward it.
            return
        else:
            raise ValueError("未知operation")


async def serve_runner(
    runner: DemoRunner | ConfiguredRunner, ready: dict[str, Any]
) -> None:
    def emit(event: dict[str, Any]) -> None:
        print(json.dumps(event, ensure_ascii=False), flush=True)

    service = JsonlService(runner, emit)
    emit({"type": "ready", **ready})
    transport = None
    if sys.platform == "win32":
        # A child's ordinary stdin is not an overlapped Windows pipe, so the
        # Proactor loop cannot attach connect_read_pipe to it.
        readline = _read_windows_stdin_line
    else:
        reader = asyncio.StreamReader(limit=1024 * 1024)
        protocol = asyncio.StreamReaderProtocol(reader)
        transport, _ = await asyncio.get_running_loop().connect_read_pipe(
            lambda: protocol, sys.stdin
        )
        readline = reader.readline
    try:
        while line := await readline():
            request = None
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("请求必须是JSON对象")
                await service.dispatch(request)
            except (ValueError, TypeError) as error:
                emit(
                    {
                        "id": request.get("id") if isinstance(request, dict) else None,
                        "type": "error",
                        "stage": "protocol",
                        "message": str(error),
                    }
                )
    finally:
        await service.stop()
        if transport is not None:
            transport.close()


async def _read_windows_stdin_line() -> bytes:
    from anyio.to_thread import run_sync

    # AnyIO's worker is abandoned on cancellation, so a blocked stdin read
    # cannot prevent asynchronous provider cleanup or process exit. Read one
    # bounded line at a time instead of accumulating an unbounded queue.
    limit = 1024 * 1024
    line = await run_sync(sys.stdin.buffer.readline, limit + 1, abandon_on_cancel=True)
    if len(line) > limit:
        raise ValueError("facial-test request exceeds the JSONL size limit")
    return line


async def serve(providers: Providers, output: Path | None) -> None:
    await serve_runner(
        DemoRunner(providers, output),
        {"configuration": providers.configuration.public(), "cases": list(CASES)},
    )


async def serve_configured(directory: Path, output: Path | None = None) -> None:
    """No models, microphones or server start until an explicit JSONL run request."""
    await serve_runner(
        ConfiguredRunner(
            directory,
            output,
            api_url=os.environ.get("FISH_MEMORY_FACIAL_TEST_API_URL"),
            api_key=os.environ.get("FISH_MEMORY_FACIAL_TEST_API_KEY"),
        ),
        {"mode": "jev"},
    )


async def batch(
    providers: Providers,
    output: Path,
    *,
    mode: str,
    repeat: int,
    parameters: list[dict[str, Any]],
    parameter_source: str,
) -> None:
    runner = DemoRunner(providers, output)
    channels = active_channels(parameters)
    summary = []
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=True)
    (output / "rubric.json").write_text(
        json.dumps(
            {
                "activeChannels": parameter_list(channels),
                "parameterSource": parameter_source,
                "jevQuestions": questions(channels),
                "reasoningSystemPrompt": reasoning_prompt(channels),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for attempt in range(repeat):
        for case in CASES:
            events = []

            def record(event: dict[str, Any], events: list = events) -> None:
                if event["type"] != "audio":
                    events.append(event)

            await runner.run(
                {
                    "id": f"{case['id']}-{attempt + 1}",
                    "operation": "run",
                    "text": case["text"],
                    "mode": mode,
                    "parameters": parameters,
                    "parameterSource": parameter_source,
                },
                record,
            )
            summary.append({"case": case, "attempt": attempt + 1, "events": events})
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "attempt": attempt + 1,
                        "status": events[-1].get("status"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            (output / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2)
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument(
        "--mode", choices=("jev", "reasoning", "compare"), default="compare"
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--tts-model")
    parser.add_argument("--tts-voice")
    parameter_options = parser.add_mutually_exclusive_group()
    parameter_options.add_argument(
        "--parameters",
        type=Path,
        help="batch使用的模型参数JSON数组，项为id/minimum/maximum/defaultValue",
    )
    parameter_options.add_argument(
        "--standard-parameters",
        action="store_true",
        help="仅后端batch：显式使用标准全部JEV参数，不代表模型实际支持",
    )
    args = parser.parse_args()
    if args.batch and not args.output:
        parser.error("--batch需要--output")
    if args.batch and not (args.parameters or args.standard_parameters):
        parser.error("--batch需要--parameters文件或显式--standard-parameters")
    if not args.batch and (args.parameters or args.standard_parameters):
        parser.error("参数清单flags仅用于--batch；JSONL模式每次run必须携带parameters")
    if not 1 <= args.repeat <= 3:
        parser.error("--repeat必须在1到3之间")

    async def run() -> None:
        configuration = load_configuration(
            tts_model=args.tts_model, tts_voice=args.tts_voice
        )
        providers = Providers(configuration)
        try:
            if args.batch:
                parameters = (
                    json.loads(args.parameters.read_text())
                    if args.parameters
                    else standard_parameters()
                )
                await batch(
                    providers,
                    args.output,
                    mode=args.mode,
                    repeat=args.repeat,
                    parameters=parameters,
                    parameter_source=(
                        "explicit_model_parameters"
                        if args.parameters
                        else "standard_only_model_support_unverified"
                    ),
                )
            else:
                await serve(providers, args.output)
        finally:
            await providers.close()

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    except Exception as error:
        # No traceback, request headers or configuration secret values on stdout.
        print(
            json.dumps(
                {"type": "error", "stage": "startup", "message": type(error).__name__}
            ),
            flush=True,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
