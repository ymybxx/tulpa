from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import TypeVar, cast

from fish_memory.answer.context import ChatMessage
from fish_memory.home.adapters import supported_domains
from fish_memory.home.models import (
    HomeAreaFacet,
    HomeControlPlan,
    HomeIntentType,
    HomeOperation,
    HomeQueryDetail,
    HomeQueryPlan,
    HomeScopeKind,
    HomeScopePlan,
    HomeSelection,
    HomeStateFilter,
)
from fish_memory.llm.client import JsonCompletionClient
from fish_memory.performance.tracker import MessagePerformanceTracker

_SHARED_KEYS = frozenset({"scope", "areaIndexes", "domains", "selection", "count"})
_QUERY_KEYS = _SHARED_KEYS | {"stateFilter", "detail"}
_CONTROL_KEYS = _SHARED_KEYS | {"operation", "value"}
_EnumT = TypeVar("_EnumT")

_QUERY_EXAMPLE = (
    '{"scope":"area","areaIndexes":[0],"domains":["light"],'
    '"selection":"all","count":null,"stateFilter":"active",'
    '"detail":"summary"}'
)
_CONTROL_EXAMPLE = (
    '{"scope":"specific","areaIndexes":[],"domains":["light"],'
    '"selection":"specific","count":null,"operation":"turn_off",'
    '"value":null}'
)
_AREA_SPECIFIC_EXAMPLE = (
    '{"scope":"area","areaIndexes":[0],"domains":["light"],'
    '"selection":"specific","count":null,"operation":"turn_off",'
    '"value":null}'
)
_SELECTION_RULE = (
    "- 只有明确的全部、所有或 every/all 才能使用 all；"
    "按名称、位置或特征描述的目标必须使用 specific。"
)
_GENERIC_DOMAIN_RULE = (
    "- 泛指设备且没有设备类型时 domains 必须为空数组；"
    "开着、关着、状态、离线等状态词不能推断为 light 或 switch。"
)
_ABNORMAL_RULE = (
    "- 离线、不可用、故障或异常必须使用 abnormal；"
    "inactive 只表示用户明确询问关闭、停止或普通未运行状态。"
)

_SYSTEM_PROMPT = f"""你负责把智能家居请求转换为结构化范围计划。
只解释当前请求需要处理的范围、设备类型和选择方式，不选择实体 ID。
areaIndexes 只能引用输入 areas 的 candidateIndex；domains 只能使用输入 domains。
scope 只能是 home、area、specific；selection 只能是 all、specific、count。
selection=count 时 count 必须是正整数，其他 selection 的 count 必须是 null。
{_SELECTION_RULE}
- 明确数量使用 count；不要因为某个房间或类型可能有多个设备就自动使用 all。
{_GENERIC_DOMAIN_RULE}

query 请求还必须返回 stateFilter 和 detail：
- stateFilter 只能是 interesting、all、active、inactive、abnormal；
{_ABNORMAL_RULE}
- detail 只能是 summary、full，只有用户明确要求全部、逐个或完整列表时使用 full。
query 示例：
{_QUERY_EXAMPLE}

control 请求还必须返回 operation 和 value：
- operation 只能使用输入 operations；
- value 没有参数时为 null。
control 示例：
{_CONTROL_EXAMPLE}
位置或特征限定目标的 control 示例（例如“关闭客厅电视后面的灯”）：
{_AREA_SPECIFIC_EXAMPLE}

只返回 JSON 对象，不要输出 entityId、设备索引、解释、Markdown 或协议外字段。"""


class HomeScopePlanner:
    def __init__(self, llm: JsonCompletionClient) -> None:
        self._llm = llm

    async def plan(
        self,
        text: str,
        intent: HomeIntentType,
        *,
        areas: Sequence[HomeAreaFacet],
        history: Sequence[ChatMessage] = (),
        tracker: MessagePerformanceTracker | None = None,
    ) -> HomeControlPlan | HomeQueryPlan:
        if not text.strip():
            raise ValueError("Home scope content must not be blank")
        payload: dict[str, object] = {
            "history": list(history),
            "content": text,
            "intent": intent.value,
            "areas": [
                {"candidateIndex": index, "name": area.name}
                for index, area in enumerate(areas)
            ],
            "domains": sorted(supported_domains()),
        }
        if intent is HomeIntentType.CONTROL:
            payload["operations"] = sorted(
                operation.value
                for operation in HomeOperation
                if operation is not HomeOperation.QUERY
            )
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
            parsed = json.loads(completion.content)
            return _parse_plan(parsed, intent, areas)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
            raise
        finally:
            if tracker is not None:
                tracker.record(
                    "llm.home_scope",
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


def _parse_plan(
    value: object,
    intent: HomeIntentType,
    areas: Sequence[HomeAreaFacet],
) -> HomeControlPlan | HomeQueryPlan:
    if not isinstance(value, dict):
        raise ValueError("Home scope response must be an object")
    parsed = cast(dict[str, object], value)
    expected = _QUERY_KEYS if intent is HomeIntentType.QUERY else _CONTROL_KEYS
    fields = set(parsed)
    unknown = fields - expected
    if unknown:
        raise ValueError(f"Home scope response has unknown fields: {sorted(unknown)}")
    missing = expected - fields
    if missing:
        raise ValueError(f"Home scope response is missing fields: {sorted(missing)}")

    scope = _parse_scope(parsed, areas)
    if intent is HomeIntentType.QUERY:
        return HomeQueryPlan(
            scope=scope,
            state_filter=_enum_value(
                HomeStateFilter,
                parsed["stateFilter"],
                "stateFilter",
            ),
            detail=_enum_value(HomeQueryDetail, parsed["detail"], "detail"),
        )

    operation = _enum_value(HomeOperation, parsed["operation"], "operation")
    if operation is HomeOperation.QUERY:
        raise ValueError("control operation must not be query")
    raw_value = parsed["value"]
    if raw_value is not None and not isinstance(raw_value, (str, int, float, bool)):
        raise ValueError("control value must be a scalar or null")
    return HomeControlPlan(scope=scope, operation=operation, value=raw_value)


def _parse_scope(
    value: Mapping[str, object],
    areas: Sequence[HomeAreaFacet],
) -> HomeScopePlan:
    kind = _enum_value(HomeScopeKind, value["scope"], "scope")
    selection = _enum_value(HomeSelection, value["selection"], "selection")
    area_indexes = _integer_tuple(value["areaIndexes"], "areaIndexes")
    if len(set(area_indexes)) != len(area_indexes):
        raise ValueError("area indexes must not contain duplicates")
    if any(index < 0 or index >= len(areas) for index in area_indexes):
        raise ValueError("area index is outside the supplied area array")
    if kind is HomeScopeKind.HOME and area_indexes:
        raise ValueError("home scope must not contain area indexes")
    if kind is HomeScopeKind.AREA and not area_indexes:
        raise ValueError("area scope requires at least one area index")

    domains_value = value["domains"]
    if not isinstance(domains_value, list) or any(
        not isinstance(domain, str) for domain in domains_value
    ):
        raise ValueError("domains must be a string array")
    domains = tuple(cast(str, domain) for domain in domains_value)
    if len(set(domains)) != len(domains):
        raise ValueError("domains must not contain duplicates")
    allowed_domains = supported_domains()
    if any(domain not in allowed_domains for domain in domains):
        raise ValueError("domain is outside the supported Home domains")

    count = value["count"]
    if selection is HomeSelection.COUNT:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("count selection requires a positive count")
    elif count is not None:
        raise ValueError("count is only valid for count selection")
    return HomeScopePlan(
        kind=kind,
        area_indexes=area_indexes,
        domains=domains,
        selection=selection,
        count=cast(int | None, count),
    )


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{field} must be an integer array")
    return tuple(cast(int, item) for item in value)


def _enum_value(
    enum_type: type[_EnumT],
    value: object,
    field: str,
) -> _EnumT:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    try:
        return enum_type(value)
    except ValueError as exception:
        raise ValueError(f"{field} has an unsupported value") from exception
