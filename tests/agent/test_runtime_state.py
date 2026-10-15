import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import SecretStr

from fish_memory.agent.configuration import configure_global_agent
from fish_memory.agent.hermes import HermesConnectorFactory
from fish_memory.agent.runtime_state import AgentRuntimeState, EnabledAgentProcessor
from fish_memory.config import Settings


@pytest.mark.asyncio
async def test_disabled_worker_waits_without_polling_and_resumes_same_instance():
    state = AgentRuntimeState(enabled=False, default_target_id=None)
    processor = SimpleNamespace(process_once=AsyncMock(return_value=True))
    worker = EnabledAgentProcessor(processor, state)
    pending = asyncio.create_task(worker.process_once())
    await asyncio.sleep(0)
    processor.process_once.assert_not_called()
    state.configure(enabled=True, default_target_id=uuid4())
    assert await asyncio.wait_for(pending, 1) is True
    state.configure(enabled=False, default_target_id=None)
    pending = asyncio.create_task(worker.process_once())
    await asyncio.sleep(0)
    assert processor.process_once.call_count == 1
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending


@pytest.mark.asyncio
async def test_global_configuration_updates_local_target_without_external_calls():
    state = AgentRuntimeState(enabled=False, default_target_id=None)
    settings = Settings.model_construct(agent_tasks_enabled=False)
    target_id = uuid4()
    targets = SimpleNamespace(
        upsert_target=AsyncMock(
            return_value=SimpleNamespace(target=SimpleNamespace(id=target_id))
        )
    )
    credentials = SimpleNamespace(store_global=Mock(), clear_global_credentials=Mock())
    factory = HermesConnectorFactory()
    result = await configure_global_agent(
        settings,
        state,
        targets,
        factory,
        credentials,
        enabled=True,
        url="http://127.0.0.1:8642/p/test",
        api_key=SecretStr("test-key"),
    )
    assert result.configured and state.enabled and settings.agent_tasks_enabled
    assert state.default_target_id == target_id
    credentials.store_global.assert_called_once_with(result.credential_ref, "test-key")
    assert targets.upsert_target.call_args.args[:4] == (
        "hermes",
        "http://127.0.0.1:8642",
        "fish-memory:global",
        "/p/test",
    )
    await configure_global_agent(
        settings,
        state,
        targets,
        factory,
        credentials,
        enabled=False,
        url="http://127.0.0.1:8642/p/test",
        api_key=SecretStr("test-key"),
    )
    assert not state.enabled and state.default_target_id == target_id
    credentials.clear_global_credentials.assert_not_called()
    revision = state.revision
    targets.upsert_target.side_effect = RuntimeError("Database unavailable")
    with pytest.raises(RuntimeError):
        await configure_global_agent(
            settings,
            state,
            targets,
            factory,
            credentials,
            enabled=True,
            url="http://127.0.0.1:8643",
            api_key=SecretStr("new-key"),
        )
    assert not state.enabled and state.revision == revision
    assert settings.agent_url == "http://127.0.0.1:8642/p/test"
    assert credentials.store_global.call_count == 2
