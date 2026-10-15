from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import cast

from fish_memory.answer.context import ChatMessage
from fish_memory.home.models import HomeAction, HomeCandidate, HomeOperation
from fish_memory.llm.client import JsonCompletionClient
from fish_memory.performance.tracker import MessagePerformanceTracker

_SYSTEM_PROMPT = """你只负责从输入 entities 中选择当前请求明确指向的设备端点。
candidateIndexes 只能引用输入实体的 candidateIndex，可以选择一个或多个。
operation 和 value 已由上游确定，不能修改；不要返回它们。
无法确定目标时也不得猜测，调用方会把空选择视为歧义。
只返回 JSON：{"candidateIndexes":[0]}"""


class HomeDeviceSelector:
    def __init__(
        self,
        llm: JsonCompletionClient,
        *,
        candidate_limit: int = 80,
    ) -> None:
        if candidate_limit < 1:
            raise ValueError("Home selector candidate limit must be positive")
        self._llm = llm
        self._candidate_limit = candidate_limit

    async def select(
        self,
        text: str,
        candidates: Sequence[HomeCandidate],
        *,
        operation: HomeOperation,
        value: str | int | float | bool | None,
        history: Sequence[ChatMessage] = (),
        tracker: MessagePerformanceTracker | None = None,
    ) -> tuple[HomeAction, ...]:
        if not candidates:
            raise ValueError("Home selector requires at least one candidate")
        if len(candidates) > self._candidate_limit:
            raise ValueError("Home selector candidate count exceeds its limit")
        payload = {
            "history": list(history),
            "content": text,
            "operation": operation.value,
            "value": value,
            "entities": [
                {
                    "candidateIndex": index,
                    "name": candidate.entity.aliases[0],
                    "deviceName": candidate.entity.device_name,
                    "area": candidate.entity.area_name,
                    "domain": candidate.entity.domain,
                    "state": candidate.entity.state,
                    "allowedOperations": sorted(
                        item.value for item in candidate.entity.operations
                    ),
                }
                for index, candidate in enumerate(candidates)
            ],
        }
        messages: list[ChatMessage] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        started_ns = tracker.now_ns() if tracker is not None else time.perf_counter_ns()
        request_payload = self._llm.json_request_payload(messages)
        completion = None
        error: str | None = None
        try:
            completion = await self._llm.complete_json(messages)
            indexes = _parse_indexes(completion.content, len(candidates))
            actions: list[HomeAction] = []
            for index in indexes:
                entity = candidates[index].entity
                if operation not in entity.operations:
                    raise ValueError(
                        f"selected candidate does not support {operation.value}"
                    )
                actions.append(HomeAction(entity.entity_id, operation, value))
            return tuple(actions)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            raise
        finally:
            if tracker is not None:
                tracker.record(
                    "llm.home_selector",
                    started_ns=started_ns,
                    finished_ns=tracker.now_ns(),
                    input_payload=request_payload,
                    output_payload=(
                        None
                        if completion is None
                        else {
                            "content": completion.content,
                            "input_tokens": completion.input_tokens,
                            "output_tokens": completion.output_tokens,
                        }
                    ),
                    error=error,
                )


def _parse_indexes(content: str, candidate_count: int) -> tuple[int, ...]:
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"candidateIndexes"}:
        raise ValueError("Home selector response must contain only candidateIndexes")
    raw_indexes = value["candidateIndexes"]
    if not isinstance(raw_indexes, list):
        raise ValueError("candidateIndexes must be an array")
    if any(
        isinstance(index, bool) or not isinstance(index, int) for index in raw_indexes
    ):
        raise ValueError("candidateIndex must be an integer")
    indexes = tuple(cast(int, index) for index in raw_indexes)
    if len(set(indexes)) != len(indexes):
        raise ValueError("candidateIndexes must not contain duplicates")
    if any(index < 0 or index >= candidate_count for index in indexes):
        raise ValueError("candidateIndex is outside the candidate array")
    return indexes
