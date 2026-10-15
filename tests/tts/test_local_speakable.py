from fish_memory.tts.local import speakable_text


def test_local_voice_does_not_read_emoji_names() -> None:
    assert speakable_text("你好呀😊～今天☀️真好✨！") == "你好呀～今天真好！"
    assert speakable_text("👨‍👩‍👧 一家人 ⭐️") == " 一家人 "
    assert (
        speakable_text("没有表情，也不动标点：（）……") == "没有表情，也不动标点：（）……"
    )
