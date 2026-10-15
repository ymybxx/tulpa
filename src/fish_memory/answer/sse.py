import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from functools import partial
from typing import TypeAlias
from uuid import UUID

from fish_memory.answer.runtime import FishRuntime
from fish_memory.models import Message, MessageStatus, visible_message_content

EventData: TypeAlias = dict[str, object]
TERMINAL_STATUSES = {
    MessageStatus.COMPLETED,
    MessageStatus.INTERRUPTED,
    MessageStatus.FAILED,
}


@dataclass(frozen=True, slots=True)
class AnswerEvent:
    event: str
    event_id: str
    data: EventData


async def iter_answer_events(
    runtime: FishRuntime,
    *,
    after_sequence: int | None = None,
) -> AsyncGenerator[AnswerEvent, None]:
    observed_user_message_id: UUID | None = None
    observed_message_id: UUID | None = None
    stream_cursor = 0
    speech_cursor = 0
    observed_flush_offset = 0
    observed_metadata_revision = -1
    observed_terminal_status: MessageStatus | None = None

    while True:
        close_after_event = False
        async with runtime.condition:
            await runtime.condition.wait_for(
                partial(
                    _has_change,
                    runtime,
                    observed_user_message_id,
                    observed_message_id,
                    stream_cursor,
                    speech_cursor,
                    observed_flush_offset,
                    observed_metadata_revision,
                    observed_terminal_status,
                    after_sequence,
                )
            )
            user_message = runtime.latest_user_message
            if _is_unseen_user_message(
                user_message,
                observed_user_message_id,
                after_sequence,
            ):
                assert user_message is not None
                observed_user_message_id = user_message.id
                event = AnswerEvent(
                    event="answer.user",
                    event_id=f"{user_message.id}:user",
                    data={
                        "userMessageId": str(user_message.id),
                        "sequence": user_message.sequence,
                        "role": user_message.role.value,
                        "content": visible_message_content(user_message),
                        "status": user_message.status.value,
                    },
                )
            elif (message_id := runtime.answer_message_id) is None:
                continue
            elif message_id != observed_message_id:
                observed_message_id = message_id
                stream_cursor = 0
                speech_cursor = 0
                observed_flush_offset = runtime.flush_offset
                observed_metadata_revision = runtime.pet_metadata_revision
                observed_terminal_status = None
                content_end = len(runtime.content)
                event = AnswerEvent(
                    event="answer.snapshot",
                    event_id=f"{message_id}:{content_end}",
                    data={
                        "messageId": str(message_id),
                        "sequence": runtime.answer_sequence,
                        "status": (
                            runtime.status.value if runtime.status is not None else None
                        ),
                        "content": runtime.content,
                        "playbackOwner": runtime.playback_owner,
                        "consumedOffset": runtime.consumed_offset,
                        "end": runtime.end_conversation,
                        "emotion": runtime.pet_emotion.value,
                        "startOffset": 0,
                        "endOffset": content_end,
                        "flushOffset": runtime.flush_offset,
                        **_speech_data(runtime),
                    },
                )
                stream_cursor = content_end
                speech_cursor = len(runtime.speech_content)
            elif runtime.pet_metadata_revision != observed_metadata_revision:
                observed_metadata_revision = runtime.pet_metadata_revision
                event = AnswerEvent(
                    event="answer.metadata",
                    event_id=(f"{message_id}:metadata:{observed_metadata_revision}"),
                    data={
                        "messageId": str(message_id),
                        "emotion": runtime.pet_emotion.value,
                        "revision": observed_metadata_revision,
                    },
                )
            elif len(runtime.content) > stream_cursor or (
                runtime.speech_tags and len(runtime.speech_content) > speech_cursor
            ):
                content_end = len(runtime.content)
                event = AnswerEvent(
                    event="answer.delta",
                    event_id=f"{message_id}:{content_end}",
                    data={
                        "messageId": str(message_id),
                        "content": runtime.content[stream_cursor:content_end],
                        "startOffset": stream_cursor,
                        "endOffset": content_end,
                        **_speech_data(runtime),
                    },
                )
                stream_cursor = content_end
                speech_cursor = len(runtime.speech_content)
            elif runtime.flush_offset > observed_flush_offset:
                observed_flush_offset = runtime.flush_offset
                event = AnswerEvent(
                    event="answer.segment",
                    event_id=f"{message_id}:flush:{observed_flush_offset}",
                    data={
                        "messageId": str(message_id),
                        "flushOffset": observed_flush_offset,
                        "reason": "lane_answer_complete",
                    },
                )
            else:
                status = runtime.status
                if status not in TERMINAL_STATUSES:
                    continue
                observed_terminal_status = status
                event = AnswerEvent(
                    event=f"answer.{status.value}",
                    event_id=f"{message_id}:{len(runtime.content)}",
                    data={
                        "messageId": str(message_id),
                        "sequence": runtime.answer_sequence,
                        "content": runtime.content,
                        "playbackOwner": runtime.playback_owner,
                        "consumedOffset": runtime.consumed_offset,
                        "contentLength": len(runtime.content),
                        "end": runtime.end_conversation,
                        "emotion": runtime.pet_emotion.value,
                        "flushOffset": runtime.flush_offset,
                        "failure": runtime.failure,
                        **_speech_data(runtime),
                    },
                )
                close_after_event = status in {
                    MessageStatus.COMPLETED,
                    MessageStatus.FAILED,
                }

        yield event
        if close_after_event:
            return


def _has_change(
    runtime: FishRuntime,
    observed_user_message_id: UUID | None,
    observed_message_id: UUID | None,
    stream_cursor: int,
    speech_cursor: int,
    observed_flush_offset: int,
    observed_metadata_revision: int,
    observed_terminal_status: MessageStatus | None,
    after_sequence: int | None,
) -> bool:
    if _is_unseen_user_message(
        runtime.latest_user_message,
        observed_user_message_id,
        after_sequence,
    ):
        return True
    if runtime.answer_message_id is None:
        return False
    if after_sequence is not None and (
        runtime.answer_sequence is None or runtime.answer_sequence <= after_sequence
    ):
        return False
    if runtime.answer_message_id != observed_message_id:
        return True
    if runtime.pet_metadata_revision != observed_metadata_revision:
        return True
    if len(runtime.content) > stream_cursor:
        return True
    if runtime.speech_tags and len(runtime.speech_content) > speech_cursor:
        return True
    if runtime.flush_offset > observed_flush_offset:
        return True
    return (
        runtime.status in TERMINAL_STATUSES
        and runtime.status != observed_terminal_status
    )


def _speech_data(runtime: FishRuntime) -> dict[str, object]:
    return (
        {"speechContent": runtime.speech_content, "speechTags": True}
        if runtime.speech_tags
        else {}
    )


def _is_unseen_user_message(
    message: Message | None,
    observed_message_id: UUID | None,
    after_sequence: int | None,
) -> bool:
    return (
        message is not None
        and message.id != observed_message_id
        and (after_sequence is None or message.sequence > after_sequence)
    )


def format_sse_event(event: AnswerEvent) -> str:
    data = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.event_id}\nevent: {event.event}\ndata: {data}\n\n"
