from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID

from fish_memory.clock import format_prompt_timestamp, system_local_time
from fish_memory.observations.models import Observation, ObservationKind

_DEFAULT_WINDOW = timedelta(hours=1)
_NOW_WINDOW = timedelta(minutes=15)
_MAX_WINDOW = timedelta(days=30)
_DEFAULT_LIMIT = 16

_CHINESE_OBSERVATION_PATTERNS = (
    re.compile(
        r"(?:我|你.{0,4}我).{0,12}"
        r"(?:在做什么|在干什么|在干嘛|做了什么|干了什么|干了啥|"
        r"用了什么|打开了什么|看了什么|在忙什么|"
        r"在哪个(?:应用|软件|窗口|网页|网站|游戏)|"
        r"用的是(?:哪个|什么)(?:应用|软件|浏览器))"
    ),
    re.compile(
        r"(?:我|我的).{0,12}"
        r"(?:电脑|屏幕|桌面|窗口|应用|软件|浏览器|网页|网站|游戏)"
        r".{0,12}(?:上有什么|上是什么|显示什么|在哪|打开了|运行着|"
        r"用过|用了|看过|看着|操作过|浏览过|玩过|是哪个|有哪些)"
    ),
    re.compile(
        r"我.{0,8}(?:现在|当前|此刻|刚才|刚刚|今天|昨天|最近).{0,12}"
        r"(?:电脑|屏幕|桌面|窗口|应用|软件|浏览器|网页|网站|游戏)"
        r".{0,8}(?:做|干|用|看|打开|操作|浏览|玩)"
    ),
    re.compile(
        r"你.{0,8}(?:看到|看见|识别|知道|了解).{0,8}(?:我|我的)"
        r".{0,8}(?:电脑|屏幕|桌面|窗口|应用|软件|浏览器|网页|网站|游戏)"
    ),
    re.compile(
        r"(?:我|我的).{0,12}"
        r"(?:表情|脸|姿势|情绪|微笑|笑|疲倦|困|注意力|当前状态)"
        r".{0,8}(?:怎么样|是什么|如何|吗|有没有|是不是|看起来)"
    ),
    re.compile(
        r"(?:我|我的).{0,8}(?:现在|当前|此刻|刚才).{0,8}"
        r"(?:什么|怎样|怎么样)?(?:表情|脸色|姿势|情绪|状态|注意力)"
    ),
    re.compile(r"我.{0,8}(?:看起来|显得).{0,8}(?:怎么样|累|困|开心|难过|专注)"),
)
_ENGLISH_OBSERVATION_PATTERNS = (
    re.compile(
        r"\bwhat\s+(?:am|was)\s+i\s+"
        r"(?:doing|working\s+on|looking\s+at|using|playing)\b"
    ),
    re.compile(
        r"\b(?:what|which)\s+(?:app|application|window|website|site|"
        r"screen|game)s?\b.{0,48}\b(?:i|my|me)\b"
    ),
    re.compile(
        r"\b(?:what(?:'s| is)|show|tell|can\s+you\s+see|do\s+you\s+see)"
        r".{0,48}\bmy\s+(?:current\s+)?(?:screen|computer|desktop|app|"
        r"application|window|website|browser|game|camera|face|expression)\b"
    ),
    re.compile(
        r"\bcan\s+you\s+(?:see|tell).{0,32}\bwhat\s+i"
        r"(?:'m|\s+am|\s+was)\s+(?:doing|using|looking\s+at|playing)\b"
    ),
    re.compile(
        r"\b(?:do\s+i\s+look|am\s+i)\s+"
        r"(?:happy|sad|angry|tired|sleepy|focused|distracted|smiling)\b"
    ),
)

_CHINESE_RECENT_RE = re.compile(
    r"(?:最近|近|过去)\s*([0-9一二两三四五六七八九十]+)\s*"
    r"(?:个)?(小时|天|日)"
)
_ENGLISH_RECENT_RE = re.compile(
    r"\b(?:last|past)\s+"
    r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(hours?|days?)\b"
)

_ENGLISH_COUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


class ObservationQueryService(Protocol):
    async def query(
        self,
        fish_id: UUID,
        *,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
        app_bundle_id: str | None = None,
        observation_type: str | None = None,
        kind: ObservationKind | None = None,
        limit: int = 100,
    ) -> tuple[Observation, ...]: ...


class ObservationContextLoader:
    """Load a small, question-gated view of the independent Observation domain."""

    def __init__(
        self,
        service: ObservationQueryService,
        *,
        now: Callable[[], datetime],
        limit: int = _DEFAULT_LIMIT,
    ) -> None:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        self._service = service
        self._now = now
        self._limit = limit

    async def load(self, fish_id: UUID, question: str) -> str:
        now = self._now()
        window = _query_window(question, now)
        if window is None:
            return ""
        start_at, end_at = window
        observations = await self._service.query(
            fish_id,
            start_at=start_at,
            end_at=end_at,
            limit=self._limit,
        )
        return render_observation_context(observations)


def _query_window(
    question: str,
    now: datetime,
) -> tuple[datetime, datetime] | None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    normalized = " ".join(question.casefold().split())
    if not normalized or not _looks_like_observation_question(normalized):
        return None

    local_midnight = system_local_time(now).replace(
        tzinfo=None, hour=0, minute=0, second=0, microsecond=0
    )
    today_start = local_midnight.astimezone()
    if "昨天" in normalized or re.search(r"\byesterday\b", normalized):
        return (local_midnight - timedelta(days=1)).astimezone(), today_start
    if "今天" in normalized or re.search(r"\btoday\b", normalized):
        return today_start, now

    recent = _recent_duration(normalized)
    if recent is not None:
        return now - min(recent, _MAX_WINDOW), now

    if re.search(r"(?:刚才|刚刚|方才|不久前|最近)", normalized) or re.search(
        r"\b(?:just\s+now|recently|a\s+moment\s+ago)\b",
        normalized,
    ):
        return now - _DEFAULT_WINDOW, now
    if re.search(r"(?:现在|当前|此刻|正在)", normalized) or re.search(
        r"\b(?:now|currently|at\s+the\s+moment|right\s+now)\b",
        normalized,
    ):
        return now - _NOW_WINDOW, now
    return now - _DEFAULT_WINDOW, now


def _looks_like_observation_question(normalized: str) -> bool:
    return any(
        pattern.search(normalized)
        for pattern in (
            *_CHINESE_OBSERVATION_PATTERNS,
            *_ENGLISH_OBSERVATION_PATTERNS,
        )
    )


def _recent_duration(normalized: str) -> timedelta | None:
    chinese_match = _CHINESE_RECENT_RE.search(normalized)
    if chinese_match is not None:
        count = _parse_chinese_count(chinese_match.group(1))
        if count is not None:
            unit = chinese_match.group(2)
            return timedelta(hours=count) if unit == "小时" else timedelta(days=count)

    english_match = _ENGLISH_RECENT_RE.search(normalized)
    if english_match is not None:
        raw_count = english_match.group(1)
        count = int(raw_count) if raw_count.isdigit() else _ENGLISH_COUNTS[raw_count]
        unit = english_match.group(2)
        return (
            timedelta(hours=count) if unit.startswith("hour") else timedelta(days=count)
        )

    if re.search(r"(?:最近|过去)\s*(?:一|1)\s*(?:个)?小时", normalized):
        return timedelta(hours=1)
    if re.search(r"(?:最近|过去)\s*(?:一|1)\s*天", normalized):
        return timedelta(days=1)
    if re.search(r"\b(?:last|past)\s+hour\b", normalized):
        return timedelta(hours=1)
    if re.search(r"\b(?:last|past)\s+day\b", normalized):
        return timedelta(days=1)
    return None


def _parse_chinese_count(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    if value == "十":
        return 10
    if "十" in value:
        tens, _, ones = value.partition("十")
        tens_value = 1 if not tens else _CHINESE_DIGITS.get(tens)
        ones_value = 0 if not ones else _CHINESE_DIGITS.get(ones)
        if tens_value is None or ones_value is None:
            return None
        return tens_value * 10 + ones_value
    return _CHINESE_DIGITS.get(value)


def render_observation_context(observations: Sequence[Observation]) -> str:
    if not observations:
        return ""
    prefix = "# 你留意到的事情\n"
    lines: list[str] = []
    for observation in sorted(observations, key=lambda item: item.started_at):
        app = " ".join(observation.app_name.split()) if observation.app_name else ""
        title = _format_observation_time(observation)
        if app:
            title += f"｜{app}"
        lines.append(f"\n{title}\n{observation.summary.strip()}")
    return prefix + "\n".join(lines)


def _format_observation_time(observation: Observation) -> str:
    started_at = observation.started_at.astimezone(UTC)
    ended_at = observation.ended_at.astimezone(UTC)
    start = format_prompt_timestamp(started_at, sep=" ")
    end = format_prompt_timestamp(ended_at, sep=" ")
    return start if start == end else f"{start} ~ {end}"
