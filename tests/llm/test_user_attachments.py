from fish_memory.answer.context import ChatMessage, user_attachment_parts
from fish_memory.llm.ark_search import _input_messages
from fish_memory.llm.client import _diagnostic_messages, _provider_messages

PICTURE = "data:image/png;base64,AAAA"
MESSAGE: ChatMessage = {
    "role": "user",
    "content": "帮我看看这个报错 [发来图片：error.png] [发来文件：log.txt]",
    "user_attachments": (
        {"kind": "image", "name": "error.png", "data_url": PICTURE},
        {"kind": "text", "name": "log.txt", "text": "Traceback: boom"},
    ),
}


def test_parts_follow_the_order_sent_with_the_agreed_wording():
    assert user_attachment_parts(MESSAGE) == [
        ("text", "用户发来的图片：error.png"),
        ("image", PICTURE),
        ("text", "用户发来的文件《log.txt》：\n```\nTraceback: boom\n```"),
    ]
    assert user_attachment_parts({"role": "user", "content": "只有字"}) == []


def test_the_model_receives_pictures_and_file_text_after_the_words():
    content = _provider_messages([MESSAGE])[0]["content"]
    assert content == [
        {"type": "text", "text": MESSAGE["content"]},
        {"type": "text", "text": "用户发来的图片：error.png"},
        {"type": "image_url", "image_url": {"url": PICTURE}},
        {
            "type": "text",
            "text": "用户发来的文件《log.txt》：\n```\nTraceback: boom\n```",
        },
    ]


def test_the_call_log_keeps_file_text_but_not_picture_bytes():
    logged = _diagnostic_messages([MESSAGE])[0]["content"]
    assert isinstance(logged, str)
    assert "用户发来的文件《log.txt》" in logged
    assert "[图片]" in logged
    assert PICTURE not in logged


def test_ark_search_receives_the_same_parts():
    content = _input_messages([MESSAGE])[0]["content"]
    assert content[:4] == [
        {"type": "input_text", "text": MESSAGE["content"]},
        {"type": "input_text", "text": "用户发来的图片：error.png"},
        {"type": "input_image", "image_url": PICTURE},
        {
            "type": "input_text",
            "text": "用户发来的文件《log.txt》：\n```\nTraceback: boom\n```",
        },
    ]


def test_the_experimental_turn_carries_what_the_user_sent():
    from datetime import UTC, datetime

    from fish_memory.experimental.context import expression_context

    history: ChatMessage = {
        "role": "user",
        "content": "History:\nu: 帮我看看这个报错 [发来图片：error.png]",
        "context_reference": True,
        "user_attachments": MESSAGE["user_attachments"],
    }
    turn = expression_context([history], datetime(2026, 9, 27, tzinfo=UTC), "用户输入")
    assert "user_attachments" not in turn[0]
    assert turn[-1]["user_attachments"] == MESSAGE["user_attachments"]
    assert (
        "user_attachments"
        not in expression_context(
            [{"role": "user", "content": "你好"}],
            datetime(2026, 9, 27, tzinfo=UTC),
            "定时唤醒",
        )[-1]
    )


def test_a_file_with_fences_of_its_own_gets_a_longer_one():
    fenced: ChatMessage = {
        "role": "user",
        "content": "看代码",
        "user_attachments": (
            {"kind": "text", "name": "a.md", "text": "```py\nx\n```"},
        ),
    }
    assert user_attachment_parts(fenced) == [
        ("text", "用户发来的文件《a.md》：\n````\n```py\nx\n```\n````")
    ]
