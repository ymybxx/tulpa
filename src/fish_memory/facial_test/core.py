"""Shared input, semantic controls and synthetic cases for both demo providers."""

from __future__ import annotations

import json
import math
import random
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeGuard


@dataclass(frozen=True)
class Channel:
    description: str
    anchors: tuple[float, ...]
    levels: tuple[str, ...]
    neutral: float = 0
    label: str = ""
    group: str = ""
    default_value: float = 0
    mode: str = "score"
    false_description: str = ""
    true_description: str = ""


LEGACY_CHANNELS = {
    "eyeOpen": Channel(
        "双眼睁开程度；不控制眨眼，闭眼笑与笑眼形状是独立维度。",
        (0, 0.5, 1, 1.5),
        (
            "双眼完全闭合",
            "眼睑低垂或眯眼半睁",
            "自然平视正常睁眼",
            "惊讶或警觉地明显睁大双眼",
        ),
        1,
    ),
    "eyeSmile": Channel(
        "双眼的笑眼弧形程度；不是高兴概率，不等于眼睛闭合程度。",
        (0, 0.5, 1),
        (
            "没有笑眼，眼睑保持平直或认真",
            "有温和的笑眼弧形",
            "笑意明显，眼睛弯成月牙形",
        ),
    ),
    "browHeight": Channel(
        "双眉整体高度，压低为负，抬高为正；眉形倾斜另由browForm控制。",
        (-1, -0.5, 0, 0.5, 1),
        (
            "眉毛明显下压",
            "眉毛略微压低",
            "眉毛自然放松高度",
            "眉毛轻轻抬起",
            "眉毛明显挑高",
        ),
    ),
    "browForm": Channel(
        "双眉形状；负值表示内端下压的怒眉，正值表示内端上抬的八字委屈眉。",
        (-1, -0.5, 0, 0.5, 1),
        (
            "明显的怒眉，眉头内端向下压",
            "略带锋利的怒眉，眉头内端稍向下压",
            "自然平直眉形",
            "眉头内端略抬起，带担心委屈",
            "眉头内端明显抬起的八字眉，忧伤委屈",
        ),
    ),
    "mouthForm": Channel(
        "闭口基底嘴形；负值嘴角下垂，正值嘴角上扬；朗读开口由音频口型单独控制。",
        (-1, -0.5, 0, 0.5, 1),
        (
            "嘴角明显下垂",
            "嘴角略向下抿",
            "嘴角平直自然",
            "嘴角轻轻上扬",
            "嘴角明显上扬露出笑意",
        ),
    ),
    "blush": Channel(
        "脸颊实际可见的红晕程度；不是害羞概率，不能仅凭愤怒或高兴就自动脸红。",
        (0, 0.25, 0.5, 0.75, 1),
        (
            "脸颊没有红晕",
            "脸颊隐约泛红",
            "脸颊看得出淡淡红晕",
            "脸颊红晕明显",
            "脸颊通红，红晕强烈",
        ),
    ),
}

LEGACY_NEUTRAL = {key: channel.neutral for key, channel in LEGACY_CHANNELS.items()}
# The renderer owns the single checked-in catalog. PyInstaller copies that same
# file next to this module; a frozen runtime never reaches back into the checkout.
CATALOG_PATH = Path(__file__).with_name("facial-demo-parameters.json")
if not CATALOG_PATH.is_file() and not getattr(sys, "frozen", False):
    CATALOG_PATH = (
        Path(__file__).resolve().parents[3]
        / "desktop/pet/src/shared/facial-demo-parameters.json"
    )


@lru_cache(maxsize=1)
def catalog() -> dict[str, dict[str, Any]]:
    """The same checked-in standard definitions are consumed by the renderer."""
    entries = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return {entry["id"]: entry for entry in entries}


def finite_number(value: Any) -> TypeGuard[float]:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def prompt_text(value: Any, limit: int) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= limit


def _level_at(value: float, entry: dict[str, Any], boundary: str) -> str:
    anchors, levels = entry["anchors"], entry["levels"]
    if value in anchors:
        return levels[anchors.index(value)]
    # A clipped native boundary must not retain an impossible extreme label.
    for low in range(len(anchors) - 1):
        if anchors[low] < value < anchors[low + 1]:
            fraction = (value - anchors[low]) / (anchors[low + 1] - anchors[low])
            return (
                f"介于「{levels[low]}」与「{levels[low + 1]}」之间，"
                f"从「{levels[low]}」向「{levels[low + 1]}」过渡"
                f"约{fraction * 100:.3f}%的姿态（本模型可用{boundary}）"
            )
    raise ValueError(f"标准通道锚点范围无效：{entry['id']}")


def active_channels(parameters: Any) -> dict[str, Channel]:
    """Resolve supported standard and user-added native parameters."""
    if not isinstance(parameters, list) or not parameters:
        raise ValueError("run需要非空parameters清单，不能猜测模型支持的标准参数")
    definitions = catalog()
    result = {}
    for parameter in parameters:
        if (
            not isinstance(parameter, dict)
            or not {"id", "minimum", "maximum", "defaultValue"} <= set(parameter)
            or set(parameter)
            - {
                "id",
                "minimum",
                "maximum",
                "defaultValue",
                "description",
                "mode",
                "levels",
                "anchors",
                "falseDescription",
                "trueDescription",
            }
        ):
            raise ValueError("parameters每项需要id/minimum/maximum/defaultValue")
        key = parameter["id"]
        if not prompt_text(key, 100) or "\n" in key or "\r" in key:
            raise ValueError("parameters包含无效参数ID")
        if key in result:
            raise ValueError(f"parameters重复参数：{key}")
        entry: dict[str, Any] | None = definitions.get(key)
        if entry is not None and entry["owner"] != "jev":
            raise ValueError(f"参数由{entry['owner']}控制，不能交给模型：{key}")
        low, high, default = (
            parameter["minimum"],
            parameter["maximum"],
            parameter["defaultValue"],
        )
        if (
            not all(finite_number(value) for value in (low, high, default))
            or low >= high
            or not low <= default <= high
        ):
            raise ValueError(f"参数范围或默认值无效：{key}")
        description = parameter.get("description")
        if description is not None and not prompt_text(description, 500):
            raise ValueError(f"参数描述无效：{key}")
        mode = parameter.get("mode", "score")
        if mode not in ("score", "binary"):
            raise ValueError(f"参数判断模式无效：{key}")
        if entry is not None:
            low, high = max(low, entry["min"]), min(high, entry["max"])
            if low >= high:
                raise ValueError(f"参数与标准规范没有可控范围交集：{key}")
        if mode == "binary":
            if "levels" in parameter or "anchors" in parameter:
                raise ValueError(f"选择式参数不能包含评分梯度：{key}")
            false_text = parameter.get(
                "falseDescription", entry["levels"][0] if entry else None
            )
            true_text = parameter.get(
                "trueDescription", entry["levels"][-1] if entry else None
            )
            if (
                not prompt_text(false_text, 100)
                or not prompt_text(true_text, 100)
                or false_text.strip() == true_text.strip()
                or (entry is None and not prompt_text(description, 500))
            ):
                raise ValueError(f"选择式参数需要不同的是/否描述：{key}")
            result[key] = Channel(
                description=(
                    entry["description"]
                    if entry is not None and not description
                    else description.strip()
                ),
                anchors=(low, high),
                levels=(false_text.strip(), true_text.strip()),
                neutral=max(low, min(high, entry["neutral"] if entry else default)),
                label=entry["label"] if entry else key,
                group=entry["group"] if entry else "custom",
                default_value=max(low, min(high, default)),
                mode="binary",
                false_description=false_text.strip(),
                true_description=true_text.strip(),
            )
            continue
        if "falseDescription" in parameter or "trueDescription" in parameter:
            raise ValueError(f"渐进式参数不能包含选择式描述：{key}")
        if entry is None:
            levels = parameter.get("levels")
            if (
                not prompt_text(description, 500)
                or not isinstance(levels, list)
                or not 2 <= len(levels) <= 10
                or not all(prompt_text(level, 100) for level in levels)
                or len({level.strip() for level in levels}) != len(levels)
            ):
                raise ValueError(f"自定义参数需要描述和2至10级姿态：{key}")
            anchors = parameter.get("anchors")
            if anchors is None:
                anchors = _spaced_anchors(low, high, len(levels))
            if not _valid_anchors(anchors, len(levels), low, high):
                raise ValueError(f"自定义参数评分锚点无效：{key}")
            result[key] = Channel(
                description=description.strip(),
                anchors=tuple(anchors),
                levels=tuple(level.strip() for level in levels),
                neutral=default,
                label=key,
                group="custom",
                default_value=default,
            )
            continue
        levels = parameter.get("levels", entry["levels"])
        if (
            not isinstance(levels, list)
            or not 2 <= len(levels) <= 10
            or not all(prompt_text(level, 100) for level in levels)
            or len({level.strip() for level in levels}) != len(levels)
        ):
            raise ValueError(f"标准参数评分描述无效：{key}")
        anchors = parameter.get("anchors")
        if anchors is None:
            anchors = (
                entry["anchors"]
                if len(levels) == len(entry["levels"])
                else _spaced_anchors(low, high, len(levels))
            )
        if not _valid_anchors(anchors, len(levels), entry["min"], entry["max"]):
            raise ValueError(f"标准参数评分锚点无效：{key}")
        entry = {
            **entry,
            "anchors": anchors,
            "levels": [level.strip() for level in levels],
        }
        anchors = tuple(
            sorted({max(low, min(high, value)) for value in entry["anchors"]})
        )
        if len(anchors) < 2:
            raise ValueError(f"参数缺少两个不同的有效锚点：{key}")
        result[key] = Channel(
            description=description.strip() if description else entry["description"],
            anchors=anchors,
            levels=tuple(
                _level_at(value, entry, "下限" if value == low else "上限")
                for value in anchors
            ),
            neutral=max(low, min(high, entry["neutral"])),
            label=entry["label"],
            group=entry["group"],
            default_value=max(low, min(high, default)),
        )
    return result


def _valid_anchors(value: Any, count: int, low: float, high: float) -> bool:
    return (
        isinstance(value, list)
        and len(value) == count
        and all(
            finite_number(item)
            and low <= item <= high
            and (index == 0 or item > value[index - 1])
            for index, item in enumerate(value)
        )
    )


def _spaced_anchors(low: float, high: float, count: int) -> list[float]:
    return [
        low
        if index == 0
        else high
        if index == count - 1
        else low + (high - low) * index / (count - 1)
        for index in range(count)
    ]


def standard_parameters() -> list[dict[str, Any]]:
    """Explicit backend-only batch schema; this is not a model capability claim."""
    return [
        {
            "id": key,
            "minimum": entry["min"],
            "maximum": entry["max"],
            "defaultValue": entry["neutral"],
        }
        for key, entry in catalog().items()
        if entry["owner"] == "jev"
    ]


def parameter_list(channels: dict[str, Channel]) -> list[dict[str, Any]]:
    return [
        {
            "id": key,
            "minimum": channel.anchors[0],
            "maximum": channel.anchors[-1],
            "defaultValue": channel.default_value,
        }
        for key, channel in channels.items()
    ]


def neutral_face(channels: dict[str, Channel]) -> dict[str, float]:
    return {key: channel.neutral for key, channel in channels.items()}


STATE_FIELD_INSTRUCTIONS = (
    "state.previousSegments是此前朗读的文本，仅供理解前文；"
    "state.currentSegment是本次朗读的文本，用于判断本题目标姿态。"
)


def questions(channels: dict[str, Channel]) -> dict[str, Any]:
    # The question ID is not visible to JEV; each instruction names its channel.
    return {
        key: {
            "type": "noul" if channel.mode == "binary" else "score",
            "instructions": (
                f"{STATE_FIELD_INSTRUCTIONS} 本题只判断{key}"
                f"{'是否应取最大值' if channel.mode == 'binary' else ''}："
                f"{channel.description}"
            ),
            "criteria": (
                {"false": channel.false_description, "true": channel.true_description}
                if channel.mode == "binary"
                else list(channel.levels)
            ),
        }
        for key, channel in channels.items()
    }


def validate_face(value: Any, channels: dict[str, Channel]) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(channels):
        raise ValueError("表情必须包含且仅包含本次启用的标准参数ID")
    result = {}
    for key, channel in channels.items():
        number = value[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or not channel.anchors[0] <= number <= channel.anchors[-1]
        ):
            raise ValueError(f"无效表情通道：{key}")
        result[key] = float(number)
    return result


# A step the model gives this much of its belief is taken as it said.
CONFIDENT_LEVEL = 0.9


def varied_level(
    answer: dict[str, Any], channel: Channel, rng: random.Random
) -> int | None:
    """A step near the model's own pick, drawn from where it was unsure.

    Only the step it favours and its neighbour on the same side of the usual
    step can be drawn, weighted by the model's probabilities: a frowning mouth
    may frown more or less, never smile. A confident answer, or one without
    probabilities, gives None and is used as scored.
    """
    raw = answer.get("probabilities")
    if not isinstance(raw, dict):
        return None
    probabilities: dict[int, float] = {}
    for key, value in raw.items():
        if (
            isinstance(key, str)
            and key.isdigit()
            and int(key) < len(channel.anchors)
            and finite_number(value)
            and value >= 0
        ):
            probabilities[int(key)] = float(value)
    if not probabilities:
        return None
    top = max(probabilities, key=probabilities.__getitem__)
    if probabilities[top] >= CONFIDENT_LEVEL:
        return None
    usual = min(
        range(len(channel.anchors)),
        key=lambda index: abs(channel.anchors[index] - channel.neutral),
    )
    side = (top > usual) - (top < usual)
    if side == 0:
        # Favouring the usual step: lean only the way the model leaned overall.
        score = answer.get("score")
        lean = score - usual if finite_number(score) else 0
        side = (lean > 0) - (lean < 0)
    candidates = [
        level
        for level in (top - 1, top, top + 1)
        if 0 <= level < len(channel.anchors)
        and (level - usual) * side >= 0
        and (side != 0 or level == usual)
    ]
    weights = [probabilities.get(level, 0.0) for level in candidates]
    if sum(weights) <= 0:
        return None
    return rng.choices(candidates, weights=weights)[0]


def decode_jev(
    raw: dict[str, Any],
    channels: dict[str, Channel],
    rng: random.Random | None = None,
) -> dict[str, float]:
    """The face the model answered; with `rng`, unsure steps vary (varied_level)."""
    answers = raw.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("JEV未返回answers")
    result = {}
    for key, channel in channels.items():
        answer = answers.get(key)
        if channel.mode == "binary":
            probability = answer.get("noul") if isinstance(answer, dict) else None
            if (
                not isinstance(answer, dict)
                or answer.get("type") != "noul"
                or not finite_number(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError(f"JEV返回无效noul：{key}")
            result[key] = (
                channel.anchors[-1] if probability > 0.5 else channel.anchors[0]
            )
            continue
        score = answer.get("score") if isinstance(answer, dict) else None
        if (
            not isinstance(answer, dict)
            or answer.get("type") != "score"
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= len(channel.anchors) - 1
        ):
            raise ValueError(f"JEV返回无效score：{key}")
        level = varied_level(answer, channel, rng) if rng is not None else None
        if level is not None:
            result[key] = channel.anchors[level]
            continue
        low = math.floor(score)
        high = min(low + 1, len(channel.anchors) - 1)
        target = channel.anchors[low] + (
            channel.anchors[high] - channel.anchors[low]
        ) * (score - low)
        result[key] = max(channel.anchors[0], min(channel.anchors[-1], target))
    return validate_face(result, channels)


def reasoning_prompt(channels: dict[str, Channel]) -> str:
    schema = {
        key: {
            "description": channel.description,
            "neutral": channel.neutral,
            "minimum": channel.anchors[0],
            "maximum": channel.anchors[-1],
            "anchors": dict(
                zip(map(str, channel.anchors), channel.levels, strict=True)
            ),
        }
        for key, channel in channels.items()
    }
    return (
        STATE_FIELD_INSTRUCTIONS
        + " 一次联合决定全部已启用参数，只输出JSON对象，键严格为下列参数ID，"
        "值为范围内连续数值，不要解释，不要emotion字段。通道定义："
        + json.dumps(schema, ensure_ascii=False)
    )


def split_segments(text: object) -> list[str]:
    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
        raise ValueError("朗读文字必须为1到2000字符")
    segments = [
        part.strip()
        for part in re.findall(r"[^，,。！？!?；;\n]+[，,。！？!?；;\n]*", text)
    ]
    segments = [part for part in segments if part]
    if not segments or len(segments) > 24:
        raise ValueError("朗读材料需要1到24个分句")
    return segments


CASES = tuple(
    json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))
)
