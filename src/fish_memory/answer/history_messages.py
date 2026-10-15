"""Render one History snapshot without duplicating dialogue in its native-role view."""

from fish_memory.answer.context import ChatMessage
from fish_memory.history.models import HistoryChunk, format_time

RECENT_DIALOGUE_LIMIT = 16


def history_role_messages(chunks: tuple[HistoryChunk, ...]) -> tuple[ChatMessage, ...]:
    ordered = [
        (
            chunk,
            sorted(
                chunk.units,
                key=lambda unit: (unit.started_at, unit.sequence, str(unit.source_id)),
            ),
        )
        for chunk in chunks
    ]
    recent = {
        unit.source_id
        for unit in [u for _, units in ordered for u in units if u.label in {"u", "a"}][
            -RECENT_DIALOGUE_LIMIT:
        ]
    }
    messages: list[ChatMessage] = []
    for chunk, units in ordered:
        if not chunk.summary and not units:
            continue
        heading = f"{format_time(chunk.started_at)} - {format_time(chunk.ended_at)}"
        reference = [heading]
        if chunk.summary:
            reference.append(f"summary: {chunk.summary}")

        for unit in units:
            if unit.source_id not in recent:
                reference.append(unit.render())
                continue
            _flush_reference(messages, reference)
            messages.append(
                {
                    "role": "assistant" if unit.label == "a" else "user",
                    "content": unit.content.strip(),
                }
            )
        _flush_reference(messages, reference)
    return tuple(messages)


def _flush_reference(messages: list[ChatMessage], reference: list[str]) -> None:
    if reference:
        messages.append(
            {
                "role": "user",
                "content": "History:\n" + "\n".join(reference),
                "context_reference": True,
            }
        )
        reference.clear()
