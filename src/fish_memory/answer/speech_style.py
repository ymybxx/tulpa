"""One dialogue-format contract for every user-facing speech producer."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fish_memory.answer.context import ChatMessage

EMOTION_FIELD_RULES = (
    "m 表示本次发言的情绪，用于选择宠物表情；只能是 neutral、angry、happy、shy、sad。"
)

SPOKEN_TEXT_RULES = (
    "禁止为台词附加宠物自身动作、神态、内心活动、说话语气或声线的旁白说明，"
    "无论写在括号、星号中，还是单独成句。"
)

SPOKEN_TEXT_TURN_REMINDER = SPOKEN_TEXT_RULES

# The model sees Chinese instructions but must emit the provider's English
# labels verbatim, so every label carries its meaning and when it fits.
CONTROL_TAG_GUIDE = {
    "empathetic": "温柔体贴：安慰、关心、温柔地说话",
    "excited": "兴奋激动：开心、期待、得意",
    "amazed": "惊叹：惊喜、佩服、没想到",
    "curious": "好奇：追问、感兴趣",
    "mischievously": "调皮使坏：捉弄、坏笑、打小算盘",
    "sarcastic": "反讽：阴阳怪气、说反话吐槽",
    "scornful": "不屑：嫌弃、傲娇地瞧不上",
    "reluctantly": "不情愿：勉强答应、嘴硬服软",
    "serious": "严肃认真：讲正事、郑重表态",
    "sad": "难过：失落、委屈",
    "crying": "哭腔：真的哭出来或快哭了",
    "trembling": "声音发颤：害怕、紧张、激动到发抖",
    "panicked": "慌张：着急、出事了",
    "angry": "生气：恼火、发脾气",
    "shouting": "大声喊：隔空喊话、提高嗓门",
    "deep and loud shouting": "低沉洪亮的吼声：气势十足地宣布、夸张喊口号",
    "whispers": "耳语：压低声音说悄悄话、说秘密",
    "asmr": "贴耳轻柔的气声：哄睡、深夜低声陪伴",
    "like dracula": "吸血鬼般阴森拖腔：讲鬼故事、故意装神秘吓人",
    "bored": "无聊：提不起劲、敷衍",
    "tired": "疲惫：困了、累了",
    "very slowly": "很慢地说：一字一顿、强调",
    "very fast": "很快地说：连珠炮、急着说完",
}
EFFECT_TAG_GUIDE = {
    "giggles": "咯咯轻笑",
    "laughing": "放声大笑",
    "sighing": "叹气",
    "gasp": "倒吸一口气",
    "snorts": "哼、嗤鼻",
    "clears throat": "清嗓子",
    "cough": "咳嗽",
}


def _tag_lines(guide: dict[str, str]) -> str:
    return "\n".join(f"[{tag}] {meaning}" for tag, meaning in guide.items())


def speech_tags_enabled(messages: Sequence["ChatMessage"]) -> bool:
    return any(message.get("speech_tags", False) for message in messages)


def with_speech_tag_rules(
    messages: list["ChatMessage"], *, enabled: bool | None = None
) -> list["ChatMessage"]:
    """Apply the voice protocol only to speech producers, never decision inputs."""
    if not (speech_tags_enabled(messages) if enabled is None else enabled):
        return messages
    heading = "# 语音表达标签"
    rules = (
        f"{heading}\n"
        "语音标签控制 TTS 的声音，不是旁白，不受上面旁白禁令限制；"
        "只写在 a 或 append 的台词里，不在其他字段使用或另加说明。\n"
        "- 只能原样使用下表的英文标签，不翻译、不改写、不自造；"
        "没有贴切的就不加。多数台词不需要标签，标签不能代替台词正文。\n"
        "- 按这句话此刻的情绪选择，不因角色设定或前几轮的用法固定沿用同一个标签。\n"
        "- 语气标签写在句首，一直生效到下一个语气标签或本次发言结束；"
        "语气转变时再换。\n"
        "- 声音标签写在实际发出这个声音的位置，只发声一次。\n"
        "- 用户要求用某种语气说话时，用措辞和表中贴切的标签表达，"
        "不要把语气名写成标签或台词；例如温柔用 [empathetic]，"
        "害羞、撒娇靠措辞，真笑出声再加 [giggles]。\n"
        f"语气标签：\n{_tag_lines(CONTROL_TAG_GUIDE)}\n"
        f"声音标签：\n{_tag_lines(EFFECT_TAG_GUIDE)}"
    )
    result = [message.copy() for message in messages]
    for message in result:
        if message["role"] == "system":
            if heading not in message["content"]:
                message["content"] += "\n\n" + rules
            message["speech_tags"] = True
            break
    return result
