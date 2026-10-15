import json
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID, uuid4

import asyncpg

from fish_memory.embeddings.client import EmbeddingSpace
from fish_memory.soul import SoulProfile
from fish_memory.tts.voice_config import FishVoiceConfig


class FishStore:
    """Owns Fish lifecycle configuration that is not part of chat messages."""

    _ASSISTANT_BINDING_LOCK_ID = 1_572_077_402

    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        embedding_space: EmbeddingSpace,
    ) -> None:
        self._pool = pool
        self._embedding_space = embedding_space

    async def create(
        self,
        *,
        role: str | None = None,
        setting: str | None = None,
        name: str | None = None,
        background: str | None = None,
    ) -> UUID:
        fish_id = uuid4()
        profile = SoulProfile.from_values(role, setting, name, background)
        row = await self._pool.fetchrow(
            """
            INSERT INTO fishes (
                id,
                role,
                setting,
                name,
                background,
                embedding_provider,
                embedding_model,
                embedding_dimensions
            )
            VALUES ($1, $2, $3, $4, $8, $5, $6, $7)
            RETURNING id
            """,
            fish_id,
            profile.role,
            profile.setting,
            profile.name or None,
            self._embedding_space.provider,
            self._embedding_space.model,
            self._embedding_space.dimensions,
            profile.background or None,
        )
        if row is None:
            raise RuntimeError("database did not return the created fish")
        return row["id"]  # type: ignore[return-value]

    async def get_profile(self, fish_id: UUID) -> SoulProfile | None:
        row = await self._pool.fetchrow(
            """
            SELECT role, setting, name, background
            FROM fishes
            WHERE id = $1
              AND deleted_at IS NULL
            """,
            fish_id,
        )
        if row is None:
            return None
        return SoulProfile.from_values(
            _string_or_none(row["role"]),
            _string_or_none(row["setting"]),
            _string_or_none(row["name"]),
            _string_or_none(row["background"]),
        )

    async def exists(self, fish_id: UUID) -> bool:
        row = await self._pool.fetchrow(
            """
            SELECT 1
            FROM fishes
            WHERE id = $1
              AND deleted_at IS NULL
            """,
            fish_id,
        )
        return row is not None

    async def update_profile(self, fish_id: UUID, profile: SoulProfile) -> bool:
        row = await self._pool.fetchrow(
            """
            UPDATE fishes
            SET role = $2,
                setting = $3,
                name = $4,
                background = $5
            WHERE id = $1
              AND deleted_at IS NULL
            RETURNING id
            """,
            fish_id,
            profile.role,
            profile.setting,
            profile.name or None,
            profile.background or None,
        )
        return row is not None

    async def get_wake_words(self, fish_id: UUID) -> tuple[str, ...] | None:
        """The Fish's own wake words; None when it never chose any."""
        row = await self._pool.fetchrow(
            """
            SELECT wake_words
            FROM fishes
            WHERE id = $1
              AND deleted_at IS NULL
            """,
            fish_id,
        )
        if row is None or row["wake_words"] is None:
            return None
        value = row["wake_words"]
        if isinstance(value, str):
            value = json.loads(value)
        return tuple(str(word) for word in value)

    async def update_wake_words(self, fish_id: UUID, words: Sequence[str]) -> bool:
        row = await self._pool.fetchrow(
            """
            UPDATE fishes
            SET wake_words = $2::jsonb
            WHERE id = $1
              AND deleted_at IS NULL
            RETURNING id
            """,
            fish_id,
            json.dumps(list(words), ensure_ascii=False),
        )
        return row is not None

    async def get_created_at(self, fish_id: UUID) -> datetime | None:
        row = await self._pool.fetchrow(
            "SELECT created_at FROM fishes WHERE id = $1 AND deleted_at IS NULL",
            fish_id,
        )
        return None if row is None else row["created_at"]

    async def get_voice_config(self, fish_id: UUID) -> FishVoiceConfig | None:
        row = await self._pool.fetchrow(
            """
            SELECT voice_config
            FROM fishes
            WHERE id = $1
              AND deleted_at IS NULL
            """,
            fish_id,
        )
        if row is None or row["voice_config"] is None:
            return None
        value = row["voice_config"]
        if isinstance(value, str):
            value = json.loads(value)
        return FishVoiceConfig.model_validate(value)

    async def update_voice_config(
        self,
        fish_id: UUID,
        config: FishVoiceConfig | None,
    ) -> bool:
        payload = None if config is None else config.model_dump_json()
        row = await self._pool.fetchrow(
            """
            UPDATE fishes
            SET voice_config = $2::jsonb
            WHERE id = $1
              AND deleted_at IS NULL
            RETURNING id
            """,
            fish_id,
            payload,
        )
        return row is not None

    async def count_voice_references(self, voice_id: str) -> int:
        value = await self._pool.fetchval(
            """
            SELECT count(*)
            FROM fishes
            WHERE deleted_at IS NULL
              AND voice_config ->> 'voice' = $1
            """,
            voice_id,
        )
        return int(value or 0)

    async def clear_voice_references(self, voice_id: str) -> None:
        await self._pool.execute(
            """
            UPDATE fishes
            SET voice_config = NULL
            WHERE deleted_at IS NULL
              AND voice_config ->> 'voice' = $1
            """,
            voice_id,
        )

    async def get_assistant(self) -> UUID | None:
        value = await self._pool.fetchval(
            """
            SELECT id
            FROM fishes
            WHERE is_assistant
              AND deleted_at IS NULL
            LIMIT 1
            """
        )
        return value if isinstance(value, UUID) else None

    async def is_assistant(self, fish_id: UUID) -> bool:
        value = await self._pool.fetchval(
            """
            SELECT is_assistant
            FROM fishes
            WHERE id = $1
              AND deleted_at IS NULL
            """,
            fish_id,
        )
        return value is True

    async def bind_assistant(self, fish_id: UUID) -> bool:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    "SELECT pg_advisory_xact_lock($1)",
                    self._ASSISTANT_BINDING_LOCK_ID,
                )
                target = await connection.fetchrow(
                    """
                    SELECT id
                    FROM fishes
                    WHERE id = $1
                      AND deleted_at IS NULL
                    FOR UPDATE
                    """,
                    fish_id,
                )
                if target is None:
                    return False
                await connection.execute(
                    """
                    UPDATE fishes
                    SET is_assistant = FALSE
                    WHERE is_assistant
                      AND deleted_at IS NULL
                    """
                )
                await connection.execute(
                    """
                    UPDATE fishes
                    SET is_assistant = TRUE
                    WHERE id = $1
                      AND deleted_at IS NULL
                    """,
                    fish_id,
                )
        return True

    async def unbind_assistant(self) -> bool:
        row = await self._pool.fetchrow(
            """
            UPDATE fishes
            SET is_assistant = FALSE
            WHERE is_assistant
              AND deleted_at IS NULL
            RETURNING id
            """
        )
        return row is not None


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None
