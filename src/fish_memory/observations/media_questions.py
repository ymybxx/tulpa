"""Local scope selection for requests about previously played content."""

from __future__ import annotations

import re


def requests_media_recap(question: str) -> bool:
    """Select additional stored evidence, without interpreting the story itself."""
    original = question.casefold()
    text = re.sub(r"\s+", "", original)
    if "设置页面" in text or "设置界面" in text:
        return False
    subject = any(
        word in text
        for word in (
            "视频",
            "短片",
            "影片",
            "剧情",
            "故事",
            "音频",
            "播客",
            "刚才那段",
            "刚才播",
            "刚才听",
        )
    )
    subject = subject or bool(
        re.search(
            r"\b(?:videos?|clips?|movies?|stor(?:y|ies)|audio|podcasts?)\b", original
        )
    )
    recap = any(
        word in text
        for word in (
            "总结",
            "讲了",
            "讲的",
            "说的",
            "说什么",
            "讲什么",
            "发生",
            "道理",
            "完整",
            "整个",
            "整段",
            "从头",
            "内容",
            "梳理",
        )
    )
    recap = recap or bool(
        re.search(r"\b(?:summari[sz]\w*|recap|whole|happen\w*|about)\b", original)
    )
    if "为什么" in text and any(
        word in text for word in ("剧情", "故事", "结局", "最后")
    ):
        recap = True
    return subject and recap
