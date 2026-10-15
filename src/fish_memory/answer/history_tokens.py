from collections.abc import Sequence
from itertools import groupby

HISTORY_MESSAGE_OVERHEAD_TOKENS = 12


def estimate_text_tokens(text: str) -> int:
    """Return a conservative, local estimate without a model tokenizer."""
    estimated = 0
    for is_ascii, characters in groupby(text, key=str.isascii):
        length = sum(1 for _ in characters)
        if is_ascii:
            estimated += (length + 2) // 3
        else:
            estimated += length
    return estimated


def estimate_history_message_tokens(content: str) -> int:
    return HISTORY_MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens(content)


def retain_recent_message_window(
    messages: Sequence[tuple[int, int]],
    *,
    retain_limit: int,
) -> tuple[int, int]:
    """Return the earliest sequence and total for a whole-message suffix."""
    if not messages:
        return 0, 0

    total = 0
    start_sequence = messages[-1][0]
    for sequence, token_count in reversed(messages):
        if total and total + token_count > retain_limit:
            break
        total += token_count
        start_sequence = sequence
    return start_sequence, total
