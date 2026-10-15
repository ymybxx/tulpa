from fish_memory.answer.history_tokens import (
    HISTORY_MESSAGE_OVERHEAD_TOKENS,
    estimate_history_message_tokens,
    estimate_text_tokens,
    retain_recent_message_window,
)


def test_estimates_non_ascii_characters_conservatively() -> None:
    assert estimate_text_tokens("我每天打篮球") == 6


def test_groups_ascii_text_into_conservative_three_character_units() -> None:
    assert estimate_text_tokens("abcdef") == 2
    assert estimate_text_tokens("abcdefg") == 3


def test_adds_fixed_message_envelope_to_content_estimate() -> None:
    assert (
        estimate_history_message_tokens("你好") == HISTORY_MESSAGE_OVERHEAD_TOKENS + 2
    )


def test_retains_whole_messages_from_the_newest_until_the_target() -> None:
    start_sequence, total = retain_recent_message_window(
        [(1, 13), (2, 13), (3, 13), (4, 13)],
        retain_limit=20,
    )

    assert start_sequence == 4
    assert total == 13
