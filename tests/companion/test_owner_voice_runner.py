from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from fish_memory.companion import owner_voice
from fish_memory.companion.owner_voice_gate import OwnerVoiceGate
from fish_memory.companion.runner import CompanionRunner
from fish_memory.config import Settings


def runner_for(tmp_path: Path, *, enabled: bool) -> CompanionRunner:
    runner = object.__new__(CompanionRunner)
    runner.settings = cast(
        Settings,
        SimpleNamespace(
            owner_voice_enabled=enabled,
            owner_voice_dir=str(tmp_path),
            owner_voice_threshold=0.5,
            companion_audio_backend="livekit",
        ),
    )
    runner._owner_voice_verifier = None
    return runner


@pytest.mark.asyncio
async def test_disabled_owner_voice_does_not_load_or_download_models(
    monkeypatch, tmp_path
):
    def unexpected(*args, **kwargs):
        pytest.fail("Disabled owner recognition must not open its model/profile")

    monkeypatch.setattr(owner_voice, "OwnerVoiceService", unexpected)
    runner = runner_for(tmp_path, enabled=False)
    await runner._load_owner_voice()
    assert runner._create_owner_voice_gate() is None


@pytest.mark.asyncio
async def test_enabled_owner_voice_reuses_one_local_verifier_and_fresh_gates(
    monkeypatch, tmp_path
):
    calls = []
    verifier = SimpleNamespace(score_pcm=lambda pcm: 0.9)

    class Service:
        def __init__(self, directory):
            assert directory == tmp_path

        def load_verifier(self, **kwargs):
            calls.append(kwargs)
            return verifier

    monkeypatch.setattr(owner_voice, "OwnerVoiceService", Service)
    runner = runner_for(tmp_path, enabled=True)
    await runner._load_owner_voice()
    first = runner._create_owner_voice_gate()
    second = runner._create_owner_voice_gate()
    assert isinstance(first, OwnerVoiceGate)
    assert isinstance(second, OwnerVoiceGate)
    assert first is not second
    assert calls == [{"audio_backend": "livekit", "threshold": 0.5}]
    assert runner._owner_voice_verifier is verifier


@pytest.mark.asyncio
async def test_missing_or_invalid_profile_never_silently_opens_voice_input(
    monkeypatch, tmp_path
):
    class Service:
        def __init__(self, directory):
            pass

        def load_verifier(self, **kwargs):
            raise RuntimeError("invalid profile")

    monkeypatch.setattr(owner_voice, "OwnerVoiceService", Service)
    runner = runner_for(tmp_path, enabled=True)
    await runner._load_owner_voice()
    assert runner._create_owner_voice_gate() is not None
    assert runner._owner_voice_verifier is not None
    with pytest.raises(RuntimeError, match="input remains blocked"):
        runner._owner_voice_verifier.score_pcm(bytes(32000))


def test_uninitialized_enabled_runner_is_also_fail_closed(tmp_path):
    runner = runner_for(tmp_path, enabled=True)
    assert runner._create_owner_voice_gate() is not None


def test_owner_voice_defaults_off_and_has_independent_config():
    assert Settings.model_fields["owner_voice_enabled"].default is False
    assert Settings.model_fields["owner_voice_threshold"].default == 0.5
    assert (
        Settings.model_fields["owner_voice_dir"].validation_alias
        == "FISH_MEMORY_OWNER_VOICE_DIR"
    )
