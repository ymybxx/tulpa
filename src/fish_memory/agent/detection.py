"""Cheap, local gates for delegated Agent task routing."""

from __future__ import annotations

import re

_CHINESE_TASK_REQUEST_MARKERS = (
    "帮我",
    "替我",
    "给我",
    "任务",
    "找",
    "查",
    "看",
    "搜",
    "检索",
    "继续",
    "接着",
    "然后",
    "顺便",
    "补充",
    "做",
    "写",
    "生成",
    "测试",
    "验证",
    "重试",
    "再试",
    "重做",
    "重新执行",
    "调查",
    "调研",
    "研究",
    "整理",
    "总结",
    "分析",
    "排查",
    "实现",
    "修复",
    "执行",
    "运行",
    "处理",
    "完成",
    "计划",
    "安排",
    "提醒",
    "监控",
    "监测",
    "跟踪",
    "追踪",
    "创建",
    "新建",
    "更新",
    "配置",
    "部署",
    "下载",
    "安装",
    "改成",
    "改为",
    "修改",
    "纠正",
)

_CHINESE_AGAIN_TASK_RE = re.compile(
    r"(?:再|重新)(?:帮|替|给|找|查|看|搜|做|试|跑|执行|处理|研究|整理|更新|修改|安排|提醒|监控)"
)
_CHINESE_DISPOSAL_REQUEST_RE = re.compile(
    r"(?:^|[，。！？\s])(?:(?:请|麻烦|帮我|替我)\s*)?(?:把|将)\s*\S+"
)

# English action words are intentionally matched only at the start of an
# imperative or after an explicit delegation phrase.  This keeps broad verbs
# such as ``find`` and ``run`` from opening T for statements like "I find..."
# and "The fun run...".  ASCII lookarounds also avoid matching inside words.
_ENGLISH_TASK_ACTION = (
    r"(?:find|check|inspect|look\s+up|search(?:\s+for)?|research|investigate|"
    r"analy[sz]e|summari[sz]e|review|run|execute|perform|create|make|build|"
    r"generate|write|test|verify|validate|schedule|plan|arrange|"
    r"remind|monitor|watch|track|continue|resume|update|retry|rerun|"
    r"try\s+again|implement|fix|repair|debug|work\s+on|"
    r"configure|set\s+up|deploy|download|install|prepare\s+a\s+report|"
    r"change\s+the\s+running)"
)
_ENGLISH_TASK_REQUEST_RE = re.compile(
    rf"(?:"
    rf"^(?:(?:please|kindly)\s+|(?:let['’]s|go\s+ahead\s+and)\s+)?"
    rf"|(?<![a-z0-9_])(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    rf"|(?<![a-z0-9_])(?:help|assist)\s+me(?:\s+to)?\s+"
    rf"|(?<![a-z0-9_])i\s+(?:want|need|would\s+like)\s+you\s+to\s+"
    rf"){_ENGLISH_TASK_ACTION}(?![a-z0-9_])",
    re.IGNORECASE,
)
_ENGLISH_TASK_NOUN_RE = re.compile(
    r"(?<![a-z0-9_])task(?![a-z0-9_])",
    re.IGNORECASE,
)
_ENGLISH_DO_TASK_RE = re.compile(
    r"(?:"
    r"^\s*(?:please\s+)?do(?![a-z0-9_])\s+"
    r"(?!(?:you|i|we|they|he|she|these|those)\b)"
    r"|(?<![a-z0-9_])(?:can|could|would|will)\s+you\s+"
    r"(?:please\s+)?do(?![a-z0-9_])"
    r"|(?<![a-z0-9_])(?:help|assist)\s+me(?:\s+to)?\s+"
    r"do(?![a-z0-9_])"
    r"|(?<![a-z0-9_])i\s+(?:want|need|would\s+like)\s+you\s+to\s+"
    r"do(?![a-z0-9_])"
    r")",
    re.IGNORECASE,
)

_TASK_CORRECTION_MARKERS = (
    "改成",
    "改为",
    "修改为",
    "纠正",
    "换成",
    "不要再",
    "change the running",
    "change it to",
    "correct the running",
    "instead",
)

_TASK_RETRY_MARKERS = (
    "重试",
    "再试",
    "重新跑",
    "重新执行",
    "retry",
    "try again",
    "rerun",
)


def looks_like_agent_task_request(content: str) -> bool:
    """Return whether starting the T lane is plausibly useful."""

    normalized = content.strip().casefold()
    return (
        any(marker in normalized for marker in _CHINESE_TASK_REQUEST_MARKERS)
        or _CHINESE_AGAIN_TASK_RE.search(normalized) is not None
        or _CHINESE_DISPOSAL_REQUEST_RE.search(normalized) is not None
        or _ENGLISH_TASK_REQUEST_RE.search(normalized) is not None
        or _ENGLISH_TASK_NOUN_RE.search(normalized) is not None
        or _ENGLISH_DO_TASK_RE.search(normalized) is not None
    )


def looks_like_agent_task_correction(content: str) -> bool:
    """Recognize an explicit correction of work that is already running."""

    lowered = content.casefold()
    return any(marker in lowered for marker in _TASK_CORRECTION_MARKERS)


def looks_like_agent_task_retry(content: str) -> bool:
    """Recognize an explicit request to retry an uncertain run."""

    lowered = content.casefold()
    return any(marker in lowered for marker in _TASK_RETRY_MARKERS)
