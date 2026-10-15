"""Pictures and text files the user sends with a chat message.

An upload is kept with no message until the message carrying it is saved,
then belongs to that message. The message text already names each one
("[发来图片：a.png]"), so History, Jev and memory see them by name; only the
reply to that message receives the pictures and file text themselves.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

import asyncpg

MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_TEXT_CHARACTERS = 100_000
MAX_ATTACHMENTS = 8
IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif"})


class AttachmentError(ValueError):
    """An upload or a message's attachments cannot be accepted."""


@dataclass(frozen=True, slots=True)
class MessageAttachment:
    id: UUID
    kind: Literal["image", "text"]
    name: str
    mime_type: str
    data: bytes | None
    text: str | None
    width: int | None
    height: int | None

    @property
    def data_url(self) -> str:
        assert self.data is not None
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


class MessageAttachmentStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def add_image(
        self,
        fish_id: UUID,
        name: str,
        mime_type: str,
        data: bytes,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> UUID:
        if mime_type not in IMAGE_TYPES:
            raise AttachmentError("只支持 JPEG、PNG、WebP、GIF 图片")
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise AttachmentError("图片不能超过 6MB")
        return await self._insert(
            fish_id, "image", name, mime_type, data=data, width=width, height=height
        )

    async def add_text(
        self, fish_id: UUID, name: str, mime_type: str, text: str
    ) -> UUID:
        if not text.strip():
            raise AttachmentError("文件是空的")
        if len(text) > MAX_TEXT_CHARACTERS:
            raise AttachmentError("文件不能超过 10 万字")
        return await self._insert(fish_id, "text", name, mime_type, text=text)

    async def _insert(
        self,
        fish_id: UUID,
        kind: str,
        name: str,
        mime_type: str,
        *,
        data: bytes | None = None,
        text: str | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> UUID:
        attachment_id = uuid4()
        await self._pool.execute(
            """
            INSERT INTO message_attachments (
                id, fish_id, kind, name, mime_type, data, text_content, width, height
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
            attachment_id,
            fish_id,
            kind,
            (name.strip() or ("图片" if kind == "image" else "文件"))[:200],
            mime_type,
            data,
            text,
            width,
            height,
        )
        return attachment_id

    async def attach(
        self, fish_id: UUID, message_id: UUID, attachment_ids: Sequence[UUID]
    ) -> None:
        """Gives uploads still waiting for a message to this one, all or none."""
        if not attachment_ids:
            return
        if len(attachment_ids) > MAX_ATTACHMENTS:
            raise AttachmentError("一条消息最多带 8 个附件")
        updated = await self._pool.fetch(
            """
            UPDATE message_attachments
            SET message_id = $2
            WHERE fish_id = $1 AND id = ANY($3::uuid[]) AND message_id IS NULL
            RETURNING id
            """,
            fish_id,
            message_id,
            list(attachment_ids),
        )
        if len(updated) != len(set(attachment_ids)):
            raise AttachmentError("附件已失效，请重新添加")

    async def waiting(self, fish_id: UUID, attachment_ids: Sequence[UUID]) -> bool:
        """Whether every upload exists and is still free to go with a message."""
        if not attachment_ids:
            return True
        count = await self._pool.fetchval(
            """
            SELECT count(*) FROM message_attachments
            WHERE fish_id = $1 AND id = ANY($2::uuid[]) AND message_id IS NULL
            """,
            fish_id,
            list(attachment_ids),
        )
        return count == len(set(attachment_ids))

    async def for_message(self, message_id: UUID) -> list[MessageAttachment]:
        rows = await self._pool.fetch(
            """
            SELECT id, kind, name, mime_type, data, text_content, width, height
            FROM message_attachments
            WHERE message_id = $1
            ORDER BY created_at, id
            """,
            message_id,
        )
        return [_attachment(row) for row in rows]

    async def image(
        self, fish_id: UUID, attachment_id: UUID
    ) -> MessageAttachment | None:
        row = await self._pool.fetchrow(
            """
            SELECT id, kind, name, mime_type, data, text_content, width, height
            FROM message_attachments
            WHERE fish_id = $1 AND id = $2 AND kind = 'image'
            """,
            fish_id,
            attachment_id,
        )
        return None if row is None else _attachment(row)

    async def names_for_messages(
        self, message_ids: Sequence[UUID]
    ) -> dict[UUID, list[dict[str, str]]]:
        """What each message carried, for showing it in the chat history."""
        if not message_ids:
            return {}
        rows = await self._pool.fetch(
            """
            SELECT id, message_id, kind, name
            FROM message_attachments
            WHERE message_id = ANY($1::uuid[])
            ORDER BY created_at, id
            """,
            list(message_ids),
        )
        result: dict[UUID, list[dict[str, str]]] = {}
        for row in rows:
            result.setdefault(row["message_id"], []).append(
                {"id": str(row["id"]), "kind": row["kind"], "name": row["name"]}
            )
        return result


def _attachment(row: asyncpg.Record) -> MessageAttachment:
    return MessageAttachment(
        id=row["id"],
        kind=row["kind"],
        name=row["name"],
        mime_type=row["mime_type"],
        data=row["data"],
        text=row["text_content"],
        width=row["width"],
        height=row["height"],
    )
