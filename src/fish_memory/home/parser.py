from __future__ import annotations

import re

from fish_memory.home.catalog import HomeEntityCatalog, normalize_home_alias
from fish_memory.home.models import (
    HomeAction,
    HomeEntity,
    HomeOperation,
    HomeRouteDecision,
)

_BRIGHTNESS_RE = re.compile(r"(?:亮度|调亮|调暗).*?(\d{1,3})\s*%?")
_TEMPERATURE_RE = re.compile(r"(?:温度|调到|设为).*?(\d{1,2}(?:\.\d+)?)\s*度")
_POSITION_RE = re.compile(r"(?:位置|开到).*?(\d{1,3})\s*%?")
_REJECT_MARKERS = (
    "没有让",
    "不要",
    "如果",
    "假如",
    "是否可以",
    "他说",
    "她说",
    "提到",
)
_NEGATED_COMMAND_RE = re.compile(
    r"别\s*(?:打开|开启|关闭|关掉|调整|调亮|调暗|播放|暂停|启动|执行|开|关)"
)
_QUOTE_MARKERS = ('"', "'", "“", "”", "‘", "’")
_OPEN_TARGET_MARKERS = ("打开", "开启", "升起", "open", "turnon")
_CLOSE_TARGET_MARKERS = ("关闭", "关掉", "降下", "close", "turnoff")


class LocalHomeCommandParser:
    def infer_action(self, text: str, entity: HomeEntity) -> HomeAction | None:
        content = text.strip()
        if not _is_direct_command(content):
            return None
        operation, value = _parse_operation(content, entity)
        if operation is None or operation not in entity.operations:
            return None
        if _has_conflicting_target_state_intent(content, operation):
            return None
        return HomeAction(entity.entity_id, operation, value)

    def parse(
        self,
        text: str,
        catalog: HomeEntityCatalog,
    ) -> HomeRouteDecision | None:
        content = text.strip()
        if not _is_direct_command(content):
            return None

        candidates = catalog.search(content, limit=10)
        if not candidates:
            return None
        best_score = candidates[0].score
        best = [candidate for candidate in candidates if candidate.score == best_score]
        actions: list[HomeAction] = []
        for candidate in best:
            if action := self.infer_action(content, candidate.entity):
                actions.append(action)
        if len(actions) != 1:
            return None
        return HomeRouteDecision(
            commands=(actions[0],),
            source="local",
        )


def _is_direct_command(content: str) -> bool:
    return bool(
        content
        and not any(item in content for item in _REJECT_MARKERS)
        and not _NEGATED_COMMAND_RE.search(content)
        and not any(item in content for item in _QUOTE_MARKERS)
    )


def _has_conflicting_target_state_intent(
    text: str,
    operation: HomeOperation,
) -> bool:
    if operation not in {
        HomeOperation.TURN_ON,
        HomeOperation.TURN_OFF,
        HomeOperation.OPEN,
        HomeOperation.CLOSE,
    }:
        return False
    normalized = normalize_home_alias(text)
    has_open = any(marker in normalized for marker in _OPEN_TARGET_MARKERS)
    has_close = any(marker in normalized for marker in _CLOSE_TARGET_MARKERS)
    return has_open and has_close


def _parse_operation(
    text: str,
    entity: HomeEntity,
) -> tuple[HomeOperation | None, int | float | None]:
    normalized = normalize_home_alias(text)
    if any(marker in normalized for marker in ("开着吗", "关着吗", "状态", "多少")):
        return HomeOperation.QUERY, None
    if match := _BRIGHTNESS_RE.search(text):
        return HomeOperation.SET_BRIGHTNESS, int(match.group(1))
    brighter = any(marker in text for marker in ("调亮", "调高"))
    dimmer = any(marker in text for marker in ("调暗", "调低"))
    if brighter or dimmer:
        brightness = entity.attributes.get("brightness")
        if isinstance(brightness, int) and not isinstance(brightness, bool):
            current = round(brightness / 255 * 100)
            delta = 10 if brighter else -10
            return HomeOperation.SET_BRIGHTNESS, min(100, max(0, current + delta))
    if match := _TEMPERATURE_RE.search(text):
        return HomeOperation.SET_TEMPERATURE, float(match.group(1))
    if match := _POSITION_RE.search(text):
        return HomeOperation.SET_POSITION, int(match.group(1))
    if entity.domain == "cover":
        if any(marker in normalized for marker in ("打开", "开启", "升起", "open")):
            return HomeOperation.OPEN, None
        if any(marker in normalized for marker in ("关闭", "关掉", "降下", "close")):
            return HomeOperation.CLOSE, None
    rules = (
        (("打开", "开启", "开灯", "turnon"), HomeOperation.TURN_ON),
        (("关闭", "关掉", "关灯", "turnoff"), HomeOperation.TURN_OFF),
        (("暂停", "pause"), HomeOperation.PAUSE),
        (("播放", "play"), HomeOperation.PLAY),
        (("解锁", "unlock"), HomeOperation.UNLOCK),
        (("上锁", "锁上", "lock"), HomeOperation.LOCK),
        (("回充", "回到基站", "returntobase"), HomeOperation.RETURN_TO_BASE),
        (("启动", "开始清扫", "start"), HomeOperation.START),
        (("停止", "stop"), HomeOperation.STOP),
        (("执行", "激活", "activate"), HomeOperation.ACTIVATE),
    )
    for markers, operation in rules:
        if any(marker in normalized for marker in markers):
            return operation, None
    return None, None
