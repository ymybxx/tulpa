from typing import cast
from uuid import uuid4

import pytest

from fish_memory.db.message_attachments import AttachmentError, MessageAttachmentStore
from fish_memory.db.messages import MessagesStore
from fish_memory.db.migrations import MigrationPool, run_migrations


@pytest.mark.asyncio
async def test_uploads_wait_for_one_message_then_belong_to_it(postgres_pool):
    await run_migrations(cast(MigrationPool, postgres_pool))
    fish = uuid4()
    messages = MessagesStore(postgres_pool)
    attachments = MessageAttachmentStore(postgres_pool)
    picture = await attachments.add_image(fish, "a.png", "image/png", b"\x89PNG")
    text = await attachments.add_text(fish, "b.txt", "text/plain", "口令：长颈鹿")
    assert await attachments.waiting(fish, [picture, text])
    assert not await attachments.waiting(uuid4(), [picture])

    message = await messages.insert_user_message(
        fish, "看看 [发来图片：a.png] [发来文件：b.txt]"
    )
    await attachments.attach(fish, message.id, [picture, text])

    carried = await attachments.for_message(message.id)
    assert [(item.kind, item.name) for item in carried] == [
        ("image", "a.png"),
        ("text", "b.txt"),
    ]
    assert carried[0].data_url == "data:image/png;base64,iVBORw=="
    assert carried[1].text == "口令：长颈鹿"
    assert (await attachments.names_for_messages([message.id]))[message.id][1] == {
        "id": str(text),
        "kind": "text",
        "name": "b.txt",
    }
    assert (await attachments.image(fish, picture)).data == b"\x89PNG"
    with pytest.raises(AttachmentError):
        await attachments.attach(fish, message.id, [picture])
    with pytest.raises(AttachmentError):
        await attachments.add_image(fish, "a.svg", "image/svg+xml", b"<svg/>")
