"""Offline coverage for the model page's isolated, JEV-only runtime role."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from fish_memory import desktop_runtime
from fish_memory.facial_test import core, providers, service
from fish_memory.facial_test.runtime_speech import RuntimeSpeech

PARAMETERS = [{"id": "ParamAngleX", "minimum": -30, "maximum": 30, "defaultValue": 0}]


def write_configuration(
    directory: Path, *, jev_model="fixture-jev", speech_key="secret"
):
    (directory / "config.json").write_text(
        json.dumps(
            {
                "modelAssignments": {"speech": "s", "reasoning": "broken-unused"},
                "models": {
                    "voice": [{"id": "s", "provider": "dashscope"}],
                    "text": [{"id": "broken-unused"}],
                },
                "experimentalCompanion": {"model": jev_model},
            }
        )
    )
    (directory / "secrets.json").write_text(
        json.dumps(
            {"values": {"modelApiKeys": {"s": speech_key}, "jevApiKey": "jev-secret"}}
        )
    )


def test_jev_only_configuration_ignores_absent_or_incomplete_reasoning(tmp_path):
    write_configuration(tmp_path)
    config = providers.load_configuration(tmp_path, require_reasoning=False)
    assert config.jev_model == "fixture-jev"
    assert config.tts_voice == "longanxiaoxin"
    assert config.reasoning_url == config.reasoning_model == config.reasoning_key == ""
    assert "reasoningModel" not in config.public()
    assert "secret" not in repr(config)
    assert "secret" not in json.dumps(config.public())
    assert "悠小喵" not in core.STATE_FIELD_INSTRUCTIONS
    assert "previousSegments" in core.STATE_FIELD_INSTRUCTIONS
    assert "currentSegment" in core.STATE_FIELD_INSTRUCTIONS


class FakeProviders:
    instances = []

    def __init__(self, configuration, *, vary=False):
        self.configuration = configuration
        self.vary = vary
        self.face_calls = []
        self.audio_calls = []
        self.closed = False
        self.instances.append(self)

    async def face(self, mode, state, channels):
        self.face_calls.append((mode, state))
        return {"mode": mode, "targets": core.neutral_face(channels)}

    async def audio(self, text):
        self.audio_calls.append(text)
        yield b"\x00\x00" * 100

    async def close(self):
        self.closed = True


async def test_configured_run_reloads_settings_and_uses_only_selected_model_text(
    monkeypatch, tmp_path
):
    write_configuration(tmp_path)
    monkeypatch.setattr(service, "Providers", FakeProviders)
    FakeProviders.instances = []
    runner = service.ConfiguredRunner(tmp_path)
    request = {
        "id": "first",
        "operation": "run",
        "character": "另一只模型",
        "text": "你好。再见。",
        "parameters": PARAMETERS,
        "history": ["NEVER INCLUDE CHAT HISTORY"],
        "context": "NEVER INCLUDE BACKGROUND",
    }
    first = []
    await runner.run(request, first.append)
    initial = FakeProviders.instances[0]
    # The look editor test varies unsure steps as TA does when speaking.
    assert initial.vary is True
    assert initial.closed and len(initial.face_calls) == 2
    assert {mode for mode, _ in initial.face_calls} == {"jev"}
    for _, state in initial.face_calls:
        assert "character" not in state
        assert "NEVER INCLUDE" not in json.dumps(state)
        assert list(state) == ["previousSegments", "currentSegment"]
    assert first[0]["type"] == "started" and first[-1]["status"] == "completed"
    assert "reasoningModel" not in first[0]["configuration"]
    write_configuration(tmp_path, jev_model="updated-jev", speech_key="updated-key")
    await runner.run({**request, "id": "second"}, lambda _: None)
    second = FakeProviders.instances[1]
    assert second.closed
    assert second.configuration.jev_model == "updated-jev"
    assert second.configuration.speech_key == "updated-key"
    assert second.audio_calls == ["你好。", "再见。"]  # No stale voice PCM cache.
    with pytest.raises(ValueError, match="仅支持JEV"):
        await runner.run({**request, "mode": "compare"}, lambda _: None)
    assert len(FakeProviders.instances) == 2


@pytest.mark.parametrize("legacy_provider", ["auto", "local", None])
async def test_desktop_test_uses_current_character_voice_not_legacy_global_service(
    monkeypatch, tmp_path, legacy_provider
):
    write_configuration(tmp_path)
    path = tmp_path / "config.json"
    stored = json.loads(path.read_text())
    stored["models"]["voice"] = (
        [{"id": "s", "provider": legacy_provider}] if legacy_provider else []
    )
    path.write_text(json.dumps(stored))
    secret_path = tmp_path / "secrets.json"
    secrets = json.loads(secret_path.read_text())
    secrets["values"]["modelApiKeys"] = {}
    secret_path.write_text(json.dumps(secrets))
    original = (path.read_bytes(), secret_path.read_bytes())
    selected = RuntimeSpeech(
        api_url="http://127.0.0.1:54321",
        api_key="runtime-secret",
        provider="dashscope",
        model="qwen-audio-3.0-tts-flash",
        voice="longanxiaoxin",
        sample_rate=24000,
    )
    seen = []

    async def load_voice(api_url, api_key):
        seen.append((api_url, api_key))
        return selected

    monkeypatch.setattr(service, "load_runtime_speech", load_voice, raising=False)
    monkeypatch.setattr(service, "Providers", FakeProviders)
    FakeProviders.instances = []
    runner = service.ConfiguredRunner(
        tmp_path, api_url=selected.api_url, api_key=selected.api_key
    )
    request = {"id": "cloud", "text": "你好。", "parameters": PARAMETERS}
    events = []
    await runner.run(request, events.append)
    cloud = FakeProviders.instances[0].configuration
    assert cloud.tts_model == "qwen-audio-3.0-tts-flash"
    assert cloud.tts_voice == "longanxiaoxin"
    assert cloud.sample_rate == 24000
    assert cloud.runtime_speech is selected
    assert events[-1]["status"] == "completed"
    assert "runtime-secret" not in json.dumps(events)
    assert "runtime-secret" not in repr(cloud)

    selected = RuntimeSpeech(
        api_url=selected.api_url,
        api_key=selected.api_key,
        provider="local",
        model="matcha-icefall-zh-en",
        voice="default",
        sample_rate=16000,
    )
    await runner.run({**request, "id": "local"}, lambda _: None)
    local = FakeProviders.instances[1].configuration
    assert local.runtime_speech is selected
    assert local.tts_model == "matcha-icefall-zh-en"
    assert local.tts_voice == "default"
    assert local.sample_rate == 16000
    assert FakeProviders.instances[1].audio_calls == ["你好。"]
    assert seen == [(selected.api_url, selected.api_key)] * 2
    assert (path.read_bytes(), secret_path.read_bytes()) == original


async def test_desktop_audio_uses_shared_speech_service(monkeypatch, tmp_path):
    write_configuration(tmp_path)
    speech = RuntimeSpeech(
        api_url="http://127.0.0.1:54321",
        api_key="runtime-secret",
        provider="local",
        model="matcha-icefall-zh-en",
        voice="default",
        sample_rate=16000,
    )
    seen = []

    async def audio(selected, text):
        seen.append((selected, text))
        yield b"\x00\x00"

    monkeypatch.setattr(providers, "runtime_audio", audio)
    monkeypatch.setattr(
        providers,
        "DashScopeTtsSession",
        lambda **_: pytest.fail("desktop test bypassed the shared voice service"),
    )
    configuration = providers.load_configuration(
        tmp_path, require_reasoning=False, runtime_speech=speech
    )
    adapter = providers.Providers(configuration)
    try:
        assert [chunk async for chunk in adapter.audio("你好")] == [b"\x00\x00"]
        assert seen == [(speech, "你好")]
    finally:
        await adapter.close()


async def test_configured_stop_cancels_audio_and_closes_providers(
    monkeypatch, tmp_path
):
    write_configuration(tmp_path)
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingProviders(FakeProviders):
        async def audio(self, text):
            started.set()
            try:
                await asyncio.Event().wait()
                yield b"\x00\x00"
            finally:
                cancelled.set()

    monkeypatch.setattr(service, "Providers", BlockingProviders)
    BlockingProviders.instances = []
    events = []
    jsonl = service.JsonlService(service.ConfiguredRunner(tmp_path), events.append)
    await jsonl.dispatch(
        {"id": "stop", "operation": "run", "text": "测试。", "parameters": PARAMETERS}
    )
    await started.wait()
    await jsonl.dispatch({"id": "stop", "operation": "stop"})
    assert cancelled.is_set()
    assert BlockingProviders.instances[0].closed
    assert events[-1]["status"] == "stopped"


def test_runtime_role_delegates_without_starting_any_other_component(
    monkeypatch, tmp_path
):
    seen = []

    async def isolated(directory, output):
        seen.append((directory, output))

    monkeypatch.setattr(service, "serve_configured", isolated)
    monkeypatch.setattr(
        sys,
        "argv",
        ["runtime", "facial-test", "--config-directory", str(tmp_path)],
    )
    for name in ("_run_api", "_run_companion", "_run_environment_asr"):
        monkeypatch.setattr(
            desktop_runtime,
            name,
            lambda *_: pytest.fail("unrelated runtime started"),
        )
    desktop_runtime.main()
    assert seen == [(tmp_path, None)]


def test_runtime_startup_errors_have_safe_json(monkeypatch, tmp_path, capsys):
    async def fail(*_):
        raise RuntimeError("untrusted secret detail")

    monkeypatch.setattr(service, "serve_configured", fail)
    args = desktop_runtime.build_parser().parse_args(
        ["facial-test", "--config-directory", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as error:
        desktop_runtime._run_facial_test(args)
    assert error.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result == {"type": "error", "stage": "startup", "message": "RuntimeError"}


def test_packaged_core_reads_local_resources_without_checkout(monkeypatch, tmp_path):
    # Reproduce the spec's destination with no repository ancestors available.
    folder = tmp_path / "frozen" / "fish_memory" / "facial_test"
    folder.mkdir(parents=True)
    shutil.copy(Path(core.__file__), folder / "core.py")
    shutil.copy(core.CATALOG_PATH, folder / "facial-demo-parameters.json")
    shutil.copy(Path(core.__file__).with_name("cases.json"), folder / "cases.json")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    name = "isolated_frozen_facial_core"
    spec = importlib.util.spec_from_file_location(name, folder / "core.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    assert module.CATALOG_PATH == folder / "facial-demo-parameters.json"
    assert len(module.catalog()) == 29
    assert len(module.standard_parameters()) == 18
    assert len(module.CASES) == 8


def test_runtime_ready_and_eof_require_no_credentials(tmp_path):
    # A pipe matches Electron stdin and exercises EOF shutdown; no run/API occurs.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fish_memory.desktop_runtime",
            "facial-test",
            "--config-directory",
            str(tmp_path),
        ],
        input="",
        text=True,
        capture_output=True,
        timeout=10,
        check=True,
    )
    assert json.loads(result.stdout) == {"type": "ready", "mode": "jev"}
    assert result.stderr == ""
