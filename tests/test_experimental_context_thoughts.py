from fish_memory.experimental.context import (
    new_since_last_thought,
    recorded_thoughts,
    with_recent_thoughts,
)

THOUGHT = "event: 宠物的内心想法（尚未表达）："


def history(*lines):
    return {
        "role": "user",
        "content": "History:\n" + "\n".join(lines),
        "context_reference": True,
    }


def test_only_latest_thoughts_stay_in_place_across_blocks_and_nested_history():
    messages = [
        {"role": "system", "content": "persona"},
        history(THOUGHT + "一", "as: 画面A"),
        {"role": "user", "content": "你好"},
        {
            "role": "user",
            "content": "",
            "history_messages": [history(THOUGHT + "二", THOUGHT + "三")],
        },
        history(
            "2026-09-23 12:30:00-05:00 - 2026-09-23 12:40:00-05:00", THOUGHT + "四"
        ),
    ]
    assert recorded_thoughts(messages) == ["一", "二", "三", "四"]
    kept = with_recent_thoughts(messages, 2)
    assert kept[1]["content"] == "History:\nas: 画面A"
    assert kept[2] == messages[2]
    assert kept[3]["history_messages"][0]["content"] == "History:\nevent: 内心想法：三"
    assert kept[4]["content"].endswith("\nevent: 内心想法：四")
    assert recorded_thoughts(messages) == ["一", "二", "三", "四"]  # not mutated


def test_thinking_can_keep_the_stored_label():
    kept = with_recent_thoughts(
        [history(THOUGHT + "一", THOUGHT + "二")], 1, relabel=False
    )
    assert kept[0]["content"] == "History:\n" + THOUGHT + "二"


def test_something_new_is_anything_but_thoughts_after_the_latest_thought():
    assert new_since_last_thought([history("as: 画面A")])
    assert not new_since_last_thought([history("as: 画面A", THOUGHT + "一")])
    assert new_since_last_thought([history(THOUGHT + "一", "as: 画面B")])
    assert new_since_last_thought(
        [history(THOUGHT + "一"), {"role": "user", "content": "你在想什么"}]
    )
    assert not new_since_last_thought(
        [
            history(THOUGHT + "一"),
            history("2026-09-23 12:30:00-05:00 - 2026-09-23 12:40:00-05:00"),
        ]
    )
