from __future__ import annotations

import json
import time
from collections.abc import Sequence
from typing import cast

from fish_memory.answer.context import ChatMessage
from fish_memory.home.brightness import resolve_brightness_value
from fish_memory.home.catalog import (
    group_candidates_by_device,
    limit_candidates_by_device,
)
from fish_memory.home.models import (
    HomeAction,
    HomeCandidate,
    HomeOperation,
    HomeRouteDecision,
    HomeSetCommand,
    HomeSetTarget,
)
from fish_memory.home.prompt import HOME_ROUTER_SYSTEM_PROMPT
from fish_memory.home.targeting import validate_set_target
from fish_memory.llm.client import JsonCompletionClient
from fish_memory.performance.tracker import MessagePerformanceTracker

_MUTUALLY_EXCLUSIVE_OPERATIONS = (
    frozenset({HomeOperation.TURN_ON, HomeOperation.TURN_OFF}),
    frozenset({HomeOperation.OPEN, HomeOperation.CLOSE}),
    frozenset({HomeOperation.LOCK, HomeOperation.UNLOCK}),
    frozenset({HomeOperation.PLAY, HomeOperation.PAUSE}),
    frozenset({HomeOperation.START, HomeOperation.RETURN_TO_BASE}),
)


class HomeRouter:
    def __init__(
        self,
        llm: JsonCompletionClient,
        *,
        device_limit: int = 40,
        command_limit: int = 40,
    ) -> None:
        self._llm = llm
        self._device_limit = device_limit
        self._command_limit = command_limit

    @property
    def device_limit(self) -> int:
        return self._device_limit

    async def route(
        self,
        text: str,
        candidates: Sequence[HomeCandidate],
        *,
        history: Sequence[ChatMessage] = (),
        tracker: MessagePerformanceTracker | None = None,
    ) -> HomeRouteDecision:
        candidates = limit_candidates_by_device(
            candidates,
            limit=self._device_limit,
        )
        if not candidates:
            return HomeRouteDecision(commands=(), source="llm")
        candidate_indexes = {
            candidate.entity.entity_id: index
            for index, candidate in enumerate(candidates)
        }
        payload = {
            "history": list(history),
            "content": text,
            "devices": [
                {
                    "name": _device_name(group),
                    "area": _device_area(group),
                    "entities": [
                        {
                            "candidateIndex": candidate_indexes[
                                candidate.entity.entity_id
                            ],
                            "name": candidate.entity.aliases[0],
                            "domain": candidate.entity.domain,
                            "state": candidate.entity.state,
                            "allowedOperations": sorted(
                                operation.value
                                for operation in candidate.entity.operations
                            ),
                        }
                        for candidate in group
                    ],
                }
                for group in group_candidates_by_device(candidates)
            ],
        }
        messages: list[ChatMessage] = [
            {"role": "system", "content": HOME_ROUTER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ]
        request_payload = self._llm.json_request_payload(messages)
        llm_started_ns = (
            tracker.now_ns() if tracker is not None else time.perf_counter_ns()
        )
        try:
            completion = await self._llm.complete_json(messages)
        except Exception as error:
            if tracker is not None:
                tracker.record(
                    "llm.home_router",
                    started_ns=llm_started_ns,
                    finished_ns=tracker.now_ns(),
                    input_payload=request_payload,
                    error=f"{type(error).__name__}: {error}",
                )
            raise
        if tracker is not None:
            tracker.record(
                "llm.home_router",
                started_ns=llm_started_ns,
                finished_ns=tracker.now_ns(),
                input_payload=request_payload,
                output_payload={
                    "content": completion.content,
                    "input_tokens": completion.input_tokens,
                    "output_tokens": completion.output_tokens,
                },
            )
        value = json.loads(completion.content)
        raw_commands = value.get("commands")
        if (
            not isinstance(raw_commands, list)
            or len(raw_commands) > self._command_limit
        ):
            raise ValueError("commands must not exceed the command count limit")
        commands: list[HomeAction | HomeSetCommand] = []
        skipped_commands: list[HomeAction] = []
        by_operation: dict[tuple[str, HomeOperation], HomeAction] = {}
        operations_by_entity: dict[str, set[HomeOperation]] = {}
        for raw in raw_commands:
            if not isinstance(raw, dict):
                raise ValueError("command must be an object")
            operation = HomeOperation(cast(str, raw.get("operation")))
            target = raw.get("target")
            if not isinstance(target, dict):
                raise ValueError("target must be an object")
            kind = target.get("kind")
            if kind == "entities":
                if any(
                    field in target
                    for field in ("selection", "area", "domain", "count")
                ):
                    raise ValueError("entities target contains set fields")
                indexes = target.get("candidateIndexes")
                if not isinstance(indexes, list) or not indexes:
                    raise ValueError("candidateIndexes must be a non-empty array")
                selected_candidates: list[HomeCandidate] = []
                for index in indexes:
                    if isinstance(index, bool) or not isinstance(index, int):
                        raise ValueError("candidateIndex must be an integer")
                    if not 0 <= index < len(candidates):
                        raise ValueError(
                            "candidateIndex is outside the candidate array"
                        )
                    selected_candidates.append(candidates[index])
                for candidate in selected_candidates:
                    entity = candidate.entity
                    value = (
                        resolve_brightness_value(raw.get("value"), entity, text=text)
                        if operation is HomeOperation.SET_BRIGHTNESS
                        else raw.get("value")
                    )
                    action = HomeAction(
                        entity.entity_id,
                        operation,
                        cast(str | int | float | bool | None, value),
                    )
                    if operation not in entity.operations:
                        # One natural-language command may select devices with
                        # different capabilities. Keep valid siblings executable
                        # and expose unsupported endpoints as a partial result.
                        if action not in skipped_commands:
                            skipped_commands.append(action)
                        continue
                    _append_entity_action(
                        action,
                        commands=commands,
                        by_operation=by_operation,
                        operations_by_entity=operations_by_entity,
                    )
                continue
            if kind == "set":
                if "candidateIndexes" in target:
                    raise ValueError("set target contains entity fields")
                set_target = _parse_set_target(target)
                validate_set_target(text, set_target, candidates)
                commands.append(
                    HomeSetCommand(
                        target=set_target,
                        operation=operation,
                        value=raw.get("value"),
                    )
                )
                continue
            raise ValueError("target.kind must be entities or set")
        return HomeRouteDecision(
            commands=tuple(commands),
            source="llm",
            skipped_commands=tuple(skipped_commands),
        )


def _device_name(group: Sequence[HomeCandidate]) -> str:
    return next(
        (
            candidate.entity.device_name
            for candidate in group
            if candidate.entity.device_name
        ),
        group[0].entity.aliases[0],
    )


def _device_area(group: Sequence[HomeCandidate]) -> str | None:
    return next(
        (
            candidate.entity.area_name
            for candidate in group
            if candidate.entity.area_name
        ),
        None,
    )


def _parse_set_target(value: dict[str, object]) -> HomeSetTarget:
    selection = value.get("selection")
    area = value.get("area")
    domain = value.get("domain")
    count = value.get("count")
    if not isinstance(selection, str):
        raise ValueError("set selection must be a string")
    if area is not None and not isinstance(area, str):
        raise ValueError("set area must be a string or null")
    if domain is not None and not isinstance(domain, str):
        raise ValueError("set domain must be a string or null")
    if isinstance(count, bool) or (count is not None and not isinstance(count, int)):
        raise ValueError("set count must be an integer or null")
    return HomeSetTarget(
        selection=selection,
        area=area,
        domain=domain,
        count=cast(int | None, count),
    )


def _append_entity_action(
    action: HomeAction,
    *,
    commands: list[HomeAction | HomeSetCommand],
    by_operation: dict[tuple[str, HomeOperation], HomeAction],
    operations_by_entity: dict[str, set[HomeOperation]],
) -> None:
    operation_key = (action.entity_id, action.operation)
    existing = by_operation.get(operation_key)
    if existing == action:
        return
    if existing is not None:
        raise ValueError("conflicting values for one entity operation")
    entity_operations = operations_by_entity.setdefault(action.entity_id, set())
    if any(
        action.operation in pair and not pair.isdisjoint(entity_operations)
        for pair in _MUTUALLY_EXCLUSIVE_OPERATIONS
    ):
        raise ValueError("conflicting actions for one entity")
    by_operation[operation_key] = action
    entity_operations.add(action.operation)
    commands.append(action)
