import asyncio
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from fish_memory.app import create_app
from fish_memory.config import Settings
from fish_memory.llm.runtime import ModelRuntime
from fish_memory.runtime_configuration import (
    RuntimeConfiguration,
    RuntimeConfigurationService,
)
from fish_memory.tts.runtime import CustomVoiceRuntime


class Client:
    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.closed = False

    async def complete_json(self):
        return self.model

    async def close(self):
        self.closed = True


def settings():
    return Settings.model_construct(
        desktop_mode=True,
        database_url="postgresql://test",
        llm_provider="openai",
        llm_base_url="https://example.invalid/v1",
        llm_model="old",
        llm_api_key=SecretStr("model-canary"),
        api_key=SecretStr("service-test-key"),
        companion_voice_draft_enabled=False,
    )


def payload():
    endpoint = {
        "baseUrl": "https://example.invalid/v1",
        "model": "new",
        "apiKey": "new-key-canary",
    }
    return {
        "models": {"chat": endpoint, "reasoning": endpoint},
        "speech": {"provider": "local", "api_key": None},
        "voice_draft_enabled": True,
    }


@pytest.mark.asyncio
async def test_authenticated_update_keeps_pool_and_changes_models_and_capabilities():
    config = settings()
    app = create_app(config)
    pool = object()
    models = ModelRuntime(
        config.resolved_llm_routes(), text_factory=Client, environment_factory=Client
    )
    custom = CustomVoiceRuntime(config, factory=lambda **_: None)
    app.state.settings, app.state.pool = config, pool
    app.state.runtime_configuration = RuntimeConfigurationService(
        config, models, custom
    )
    chat, reasoning = models.chat, models.reasoning
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.put("/internal/runtime/configuration", json=payload())
        assert response.status_code == 401
        assert await models.chat.complete_json() == "old"
        headers = {"Authorization": "Bearer service-test-key"}
        response = await client.put(
            "/internal/runtime/configuration", json=payload(), headers=headers
        )
        assert response.status_code == 200
        assert "canary" not in response.text
        assert await models.chat.complete_json() == "new"
        assert await models.reasoning.complete_json() == "new"
        assert (models.chat, models.reasoning) == (chat, reasoning)
        assert app.state.pool is pool
        assert (await client.get("/internal/capabilities", headers=headers)).json()[
            "voiceDrafts"
        ] is True
        broken = payload()
        broken["models"]["reasoning"] = {"apiKey": "must-not-echo"}
        response = await client.put(
            "/internal/runtime/configuration", json=broken, headers=headers
        )
        assert response.status_code == 400
        assert "must-not-echo" not in response.text
        assert await models.chat.complete_json() == "new"
    await models.close()
    await custom.close()


@pytest.mark.asyncio
async def test_facial_wait_mode_hot_applies_without_experimental_routing():
    config = settings()
    assert config.speech_facial_wait_for_ready is False
    assert config.experimental_companion.enabled is False
    app = create_app(config)
    models = ModelRuntime(
        config.resolved_llm_routes(), text_factory=Client, environment_factory=Client
    )
    custom = CustomVoiceRuntime(config, factory=lambda **_: None)
    app.state.settings = config
    app.state.runtime_configuration = RuntimeConfigurationService(
        config, models, custom
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            headers = {"Authorization": "Bearer service-test-key"}
            for enabled in (True, False):
                update = {**payload(), "speech_facial_wait_for_ready": enabled}
                response = await client.put(
                    "/internal/runtime/configuration", json=update, headers=headers
                )
                assert response.status_code == 200
                assert config.speech_facial_wait_for_ready is enabled
                assert config.experimental_companion.enabled is False
                # Older callers omit the optional field; unrelated updates retain it.
                response = await client.put(
                    "/internal/runtime/configuration", json=payload(), headers=headers
                )
                assert response.status_code == 200
                assert config.speech_facial_wait_for_ready is enabled
            response = await client.put(
                "/internal/runtime/configuration",
                json={**payload(), "speech_facial_wait_for_ready": "true"},
                headers=headers,
            )
            assert response.status_code == 400
            assert config.speech_facial_wait_for_ready is False
    finally:
        await models.close()
        await custom.close()


@pytest.mark.asyncio
async def test_web_search_switch_hot_applies_to_replies_and_speaking_up():
    config = settings()
    models = ModelRuntime(
        config.resolved_llm_routes(), text_factory=Client, environment_factory=Client
    )
    custom = CustomVoiceRuntime(config, factory=lambda **_: None)
    experiment = SimpleNamespace(search_enabled=True)
    service = RuntimeConfigurationService(config, models, custom, experiment=experiment)
    try:
        await service.apply(
            RuntimeConfiguration.model_validate(
                {**payload(), "web_search_enabled": False}
            )
        )
        assert config.answer_enable_search is False
        assert experiment.search_enabled is False
        # Callers that leave it out keep what was set.
        await service.apply(RuntimeConfiguration.model_validate(payload()))
        assert experiment.search_enabled is False
        await service.apply(
            RuntimeConfiguration.model_validate(
                {**payload(), "web_search_enabled": True}
            )
        )
        assert config.answer_enable_search is True
        assert experiment.search_enabled is True
    finally:
        await models.close()
        await custom.close()


@pytest.mark.asyncio
async def test_log_retention_hot_applies_without_rebuilding_runtime():
    config = settings()
    assert config.llm_log_retention_days == 7
    app = create_app(config)
    models = ModelRuntime(
        config.resolved_llm_routes(), text_factory=Client, environment_factory=Client
    )
    custom = CustomVoiceRuntime(config, factory=lambda **_: None)
    app.state.settings = config
    app.state.runtime_configuration = RuntimeConfigurationService(
        config, models, custom
    )
    pool = app.state.pool = object()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            headers = {"Authorization": "Bearer service-test-key"}
            for days in (1, 30, 7, 365):
                update = {**payload(), "llm_log_retention_days": days}
                response = await client.put(
                    "/internal/runtime/configuration", json=update, headers=headers
                )
                assert response.status_code == 200
                assert config.llm_log_retention_days == days
                assert app.state.pool is pool
                # Older callers and unrelated updates preserve the chosen period.
                response = await client.put(
                    "/internal/runtime/configuration", json=payload(), headers=headers
                )
                assert response.status_code == 200
                assert config.llm_log_retention_days == days
            for invalid in (0, -1, 366, 1.5, True, "7"):
                response = await client.put(
                    "/internal/runtime/configuration",
                    json={**payload(), "llm_log_retention_days": invalid},
                    headers=headers,
                )
                assert response.status_code == 400
                assert config.llm_log_retention_days == 365
    finally:
        await models.close()
        await custom.close()


@pytest.mark.asyncio
async def test_tts_key_update_does_not_close_an_in_progress_voice_upload():
    config = settings()
    config.tts_api_key = SecretStr("old-key")
    clients = []

    class VoiceClient:
        def __init__(self, **kwargs):
            self.key = kwargs["api_key"]
            self.entered, self.finish = asyncio.Event(), asyncio.Event()
            self.closed = 0
            clients.append(self)

        async def create_voice(self, audio, *, target_model):
            self.entered.set()
            await self.finish.wait()
            assert not self.closed
            return self.key

        async def list_voices(self):
            return [self.key]

        async def close(self):
            self.closed += 1

    runtime = CustomVoiceRuntime(config, factory=VoiceClient)
    upload = asyncio.create_task(runtime.create_voice(b"fixture", target_model="test"))
    await asyncio.sleep(0)
    old = clients[0]
    await old.entered.wait()
    config.tts_api_key = SecretStr("new-key")
    runtime.invalidate()
    assert await runtime.list_voices() == ["new-key"]
    assert old.closed == 0
    old.finish.set()
    assert await upload == "old-key"
    await asyncio.sleep(0)
    assert old.closed == 1
    config.tts_api_key = None
    runtime.invalidate()
    with pytest.raises(RuntimeError, match="not configured"):
        await runtime.list_voices()
    await runtime.close()
    assert [client.closed for client in clients] == [1, 1]


@pytest.mark.asyncio
async def test_disabled_agent_key_removal_cleans_the_persisted_copy_once():
    from fish_memory.runtime_configuration import RuntimeConfiguration

    config = settings()
    config.agent_tasks_enabled = False
    config.agent_api_key = SecretStr("old-agent-key")
    models = ModelRuntime(
        config.resolved_llm_routes(), text_factory=Client, environment_factory=Client
    )
    custom = CustomVoiceRuntime(config, factory=lambda **_: None)
    cleared = []
    service = RuntimeConfigurationService(
        config,
        models,
        custom,
        forget_agent_credentials=lambda: cleared.append(True),
    )
    body = RuntimeConfiguration.model_validate(
        {**payload(), "forget_agent_credentials": True}
    )
    await service.apply(body)
    await service.apply(body)
    assert cleared == [True]
    assert config.agent_api_key is None
    assert await models.chat.complete_json() == "new"
    await models.close()
    await custom.close()
