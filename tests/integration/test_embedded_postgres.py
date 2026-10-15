import asyncio
import os
import shutil
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest
from pg0 import Pg0
from pydantic import SecretStr

import fish_memory.app as app_module
from fish_memory.config import Settings
from fish_memory.embeddings.client import EmbeddingSpace


@pytest.mark.pg0
@pytest.mark.asyncio
async def test_app_starts_pg0_and_initializes_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.getenv("FISH_MEMORY_RUN_PG0_TESTS") != "1":
        pytest.skip("FISH_MEMORY_RUN_PG0_TESTS is not enabled")

    instance_name = f"fish-memory-test-{uuid4().hex}"
    connection: asyncpg.Connection | None = None

    class NoopLlm:
        def __init__(self, **kwargs: str) -> None:
            del kwargs

        async def close(self) -> None:
            return None

    class NoopEmbedding:
        space = EmbeddingSpace(
            provider="local",
            model="intfloat/multilingual-e5-small",
            dimensions=384,
        )

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(app_module, "BailianChatClient", NoopLlm)
    monkeypatch.setattr(
        app_module,
        "create_embedding_client",
        lambda settings: NoopEmbedding(),
    )
    settings = Settings.model_construct(
        database_url=f"pg0://{instance_name}",
        system_prompt="system",
        llm_provider="openai",
        llm_api_key=SecretStr("test-key"),
        llm_model="qwen-plus",
        llm_base_url="https://example.invalid/v1",
    )
    app = app_module.create_app(settings)
    try:
        async with app.router.lifespan_context(app):
            info = await asyncio.to_thread(Pg0(name=instance_name).info)
            assert info.running
            assert info.uri is not None
            connection = await asyncpg.connect(info.uri)
            try:
                table = await connection.fetchval(
                    "SELECT to_regclass('public.messages')"
                )
                assert table == "messages"
            finally:
                await connection.close()
                connection = None

        info = await asyncio.to_thread(Pg0(name=instance_name).info)
        assert not info.running
    finally:
        if connection is not None:
            await connection.close()
        await asyncio.to_thread(Pg0(name=instance_name).drop)
        instance_dir = Path.home() / ".pg0" / "instances" / instance_name
        await asyncio.to_thread(shutil.rmtree, instance_dir, True)
