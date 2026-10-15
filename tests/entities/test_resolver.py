import json
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

import pytest

from fish_memory.answer.context import ChatMessage
from fish_memory.entities.models import (
    EntityCandidate,
    EntityContextMessage,
    SnapshotEntityContext,
)
from fish_memory.entities.resolver import (
    EntityResolutionContext,
    EntityResolver,
    build_entity_resolution_context,
)
from fish_memory.llm.client import JsonCompletion
from fish_memory.models import Role
from fish_memory.performance.models import JsonValue
from fish_memory.snapshots.models import (
    SnapshotFact,
    SnapshotItemState,
)

NOW = datetime(2026, 8, 1, 12, tzinfo=UTC)
FISH_ID = UUID(int=1)


class FakeJsonCompletionClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[Sequence[ChatMessage]] = []

    async def complete_json(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonCompletion:
        self.calls.append(messages)
        return JsonCompletion(content=self.content, input_tokens=10, output_tokens=5)

    def json_request_payload(
        self,
        messages: Sequence[ChatMessage],
    ) -> JsonValue:
        return {"model": "test", "messages": [dict(message) for message in messages]}


def make_fact(
    index: int,
    text: str,
    entities: tuple[str, ...],
    role: Role = Role.USER,
) -> SnapshotFact:
    return SnapshotFact(
        id=UUID(int=100 + index),
        sequence=index + 1,
        fish_id=FISH_ID,
        source_message_id=UUID(int=200 + index),
        source_role=role,
        text=text,
        occurred_start=None,
        occurred_end=None,
        mentioned_at=NOW,
        entities=entities,
    )


def make_item(text: str, fact_id: UUID) -> SnapshotItemState:
    return SnapshotItemState(
        id=fact_id.int + 1,
        summary=text,
        support_fact_ids=(fact_id,),
        lineage_fact_ids=(fact_id,),
        last_supported_at=NOW,
        item_version=1,
    )


def make_normalized_item(
    summary: str,
    *,
    support_fact_id: UUID,
    lineage_fact_id: UUID,
) -> SnapshotItemState:
    return SnapshotItemState(
        id=61,
        summary=summary,
        support_fact_ids=(support_fact_id,),
        lineage_fact_ids=(support_fact_id, lineage_fact_id),
        last_supported_at=NOW,
        item_version=4,
    )


def test_normalized_entity_evidence_uses_current_support_and_summary_only() -> None:
    """Catches lineage evidence leaking back after support was removed."""
    current_fact_id = UUID(int=170)
    removed_fact_id = UUID(int=171)
    new_fact_id = UUID(int=172)
    context = SnapshotEntityContext(
        candidates=(
            EntityCandidate(UUID(int=70), ("安安",)),
            EntityCandidate(UUID(int=71), ("小美",)),
            EntityCandidate(UUID(int=72), ("妮妮",)),
        ),
        fact_entity_indexes={
            current_fact_id: (0,),
            removed_fact_id: (1,),
            new_fact_id: (2,),
        },
        fact_source_roles={
            current_fact_id: Role.USER,
            removed_fact_id: Role.USER,
            new_fact_id: Role.USER,
        },
        fact_texts={
            current_fact_id: "用户当前叫助手安安",
        },
    )

    result = build_entity_resolution_context(
        (
            make_normalized_item(
                "助手当前叫安安，曾经叫小美",
                support_fact_id=current_fact_id,
                lineage_fact_id=removed_fact_id,
            ),
        ),
        (
            SnapshotFact(
                id=new_fact_id,
                sequence=10,
                fish_id=FISH_ID,
                source_message_id=UUID(int=272),
                source_role=Role.USER,
                text="用户也叫助手妮妮",
                occurred_start=None,
                occurred_end=None,
                mentioned_at=NOW,
                entities=("妮妮",),
            ),
        ),
        context,
    )

    assert [candidate.aliases for candidate in result.candidates] == [
        ("安安",),
        ("妮妮",),
    ]
    assert result.candidates[0].evidence_texts == (
        "用户当前叫助手安安",
        "助手当前叫安安，曾经叫小美",
    )


def make_context() -> SnapshotEntityContext:
    old_bb = UUID(int=10)
    old_fat = UUID(int=11)
    new_fat = UUID(int=101)
    return SnapshotEntityContext(
        candidates=(
            EntityCandidate(old_bb, ("bb", "笨笨")),
            EntityCandidate(old_fat, ("胖胖",)),
        ),
        fact_entity_indexes={
            UUID(int=90): (0,),
            new_fat: (1,),
        },
        fact_source_roles={
            UUID(int=90): Role.USER,
            new_fat: Role.USER,
        },
    )


def test_builds_self_contained_candidates_from_new_facts_and_matching_items() -> None:
    context = make_context()
    new_fact = make_fact(1, "用户又叫助手胖胖", ("胖胖",))
    result = build_entity_resolution_context(
        (
            make_item("用户给助手取名为 BB", UUID(int=90)),
            make_item("这是与项目无关的内容", UUID(int=90)),
        ),
        (new_fact,),
        context,
    )

    assert result == EntityResolutionContext(
        candidates=(
            result.candidates[0],
            result.candidates[1],
        ),
        touched_keys=frozenset({"c1"}),
    )
    assert result.candidates[0].key == "c0"
    assert result.candidates[0].aliases == ("bb", "笨笨")
    assert result.candidates[0].evidence_texts == ("用户给助手取名为 BB",)
    assert result.candidates[1].evidence_texts == ("用户又叫助手胖胖",)


def test_does_not_send_unrelated_or_duplicate_candidate_evidence() -> None:
    context = SnapshotEntityContext(
        candidates=(
            EntityCandidate(UUID(int=20), ("a",)),
            EntityCandidate(UUID(int=21), ("b",)),
        ),
        fact_entity_indexes={UUID(int=120): (0,), UUID(int=121): (1,)},
        fact_source_roles={UUID(int=120): Role.USER, UUID(int=121): Role.USER},
    )
    result = build_entity_resolution_context(
        (
            make_item("a appears here", UUID(int=120)),
            make_item("a appears here", UUID(int=120)),
        ),
        (make_fact(20, "a appears again", ("a",)),),
        context,
    )

    assert [candidate.key for candidate in result.candidates] == ["c0"]
    assert result.candidates[0].evidence_texts == ("a appears again", "a appears here")
    assert result.touched_keys == frozenset({"c0"})


def test_keeps_source_fact_evidence_when_snapshot_text_uses_new_alias() -> None:
    old_fact_id = UUID(int=125)
    new_fact_id = UUID(int=126)
    context = SnapshotEntityContext(
        candidates=(
            EntityCandidate(UUID(int=25), ("小美",)),
            EntityCandidate(UUID(int=26), ("安安",)),
        ),
        fact_entity_indexes={old_fact_id: (0,), new_fact_id: (1,)},
        fact_source_roles={old_fact_id: Role.USER, new_fact_id: Role.USER},
        fact_texts={old_fact_id: "用户给助手取名为小美"},
    )

    result = build_entity_resolution_context(
        (make_item("助手现在叫安安", old_fact_id),),
        (make_fact(26, "用户选择安安作为助手的称呼", ("安安",)),),
        context,
    )

    assert [candidate.aliases for candidate in result.candidates] == [
        ("小美",),
        ("安安",),
    ]
    assert result.candidates[0].evidence_texts == ("用户给助手取名为小美",)
    assert result.touched_keys == frozenset({"c1"})


def test_ignores_entities_grounded_only_by_assistant_facts() -> None:
    assistant_fact_id = UUID(int=122)
    first_user_fact_id = UUID(int=123)
    second_user_fact_id = UUID(int=124)
    context = SnapshotEntityContext(
        candidates=(
            EntityCandidate(UUID(int=22), ("知识名词",)),
            EntityCandidate(UUID(int=23), ("胖胖",)),
            EntityCandidate(UUID(int=24), ("笨笨",)),
        ),
        fact_entity_indexes={
            assistant_fact_id: (0,),
            first_user_fact_id: (1,),
            second_user_fact_id: (2,),
        },
        fact_source_roles={
            assistant_fact_id: Role.ASSISTANT,
            first_user_fact_id: Role.USER,
            second_user_fact_id: Role.USER,
        },
    )

    result = build_entity_resolution_context(
        (),
        (
            make_fact(22, "助手介绍了知识名词", ("知识名词",), Role.ASSISTANT),
            make_fact(23, "用户叫助手胖胖", ("胖胖",)),
            make_fact(24, "用户叫助手笨笨", ("笨笨",)),
        ),
        context,
    )

    assert [candidate.aliases for candidate in result.candidates] == [
        ("胖胖",),
        ("笨笨",),
    ]
    assert result.touched_keys == frozenset({"c0", "c1"})


@pytest.mark.asyncio
async def test_resolver_returns_empty_without_two_touched_candidates() -> None:
    client = FakeJsonCompletionClient('{"groups":[["c0","c1"]]}')
    resolver = EntityResolver(client)
    context = SnapshotEntityContext(
        candidates=(EntityCandidate(UUID(int=30), ("only",)),),
        fact_entity_indexes={UUID(int=130): (0,)},
    )

    result = await resolver.resolve(
        (make_item("only", UUID(int=130)),),
        (make_fact(30, "only", ("only",)),),
        context,
    )

    assert result.groups == ()
    assert result.skipped is True
    assert client.calls == []


@pytest.mark.asyncio
async def test_resolver_maps_valid_keys_and_merges_overlapping_groups() -> None:
    client = FakeJsonCompletionClient(
        json.dumps({"groups": [["c0", "c1"], ["c1", "c2"]]})
    )
    resolver = EntityResolver(client)
    context = SnapshotEntityContext(
        candidates=(
            EntityCandidate(UUID(int=40), ("a",)),
            EntityCandidate(UUID(int=41), ("b",)),
            EntityCandidate(UUID(int=42), ("c",)),
        ),
        fact_entity_indexes={
            UUID(int=140): (0,),
            UUID(int=141): (1,),
            UUID(int=142): (2,),
        },
    )
    facts = (
        make_fact(40, "a", ("a",)),
        make_fact(41, "b", ("b",)),
        make_fact(42, "c", ("c",)),
    )

    result = await resolver.resolve((), facts, context)

    assert len(result.groups) == 1
    assert result.groups[0].aliases == ("a", "b", "c")
    payload = client.calls[0][1]["content"]
    assert json.loads(payload)["candidates"] == [
        {"key": "c0", "aliases": ["a"], "evidenceTexts": ["a"]},
        {"key": "c1", "aliases": ["b"], "evidenceTexts": ["b"]},
        {"key": "c2", "aliases": ["c"], "evidenceTexts": ["c"]},
    ]


@pytest.mark.asyncio
async def test_prompt_treats_same_target_name_change_as_merge_evidence() -> None:
    client = FakeJsonCompletionClient('{"groups":[]}')
    resolver = EntityResolver(client)
    context = SnapshotEntityContext(
        candidates=(
            EntityCandidate(UUID(int=43), ("小美",)),
            EntityCandidate(UUID(int=44), ("安安",)),
        ),
        fact_entity_indexes={
            UUID(int=143): (0,),
            UUID(int=144): (1,),
        },
        fact_source_roles={
            UUID(int=143): Role.USER,
            UUID(int=144): Role.USER,
        },
    )

    await resolver.resolve(
        (),
        (
            make_fact(43, "用户给助手取名为小美", ("小美",)),
            make_fact(44, "用户选择安安作为助手的称呼", ("安安",)),
        ),
        context,
    )

    system_prompt = client.calls[0][0]["content"]
    assert "改名为" in system_prompt
    assert "只有证据明确表达" in system_prompt
    assert "共同出现在同一事实" in system_prompt
    assert "密歇根湖" in system_prompt
    assert "默认不合并" in system_prompt


@pytest.mark.asyncio
async def test_prompt_explains_unique_target_and_multi_instance_role_rules() -> None:
    client = FakeJsonCompletionClient('{"groups":[]}')
    resolver = EntityResolver(client)
    context = SnapshotEntityContext(
        candidates=(
            EntityCandidate(UUID(int=45), ("甲",)),
            EntityCandidate(UUID(int=46), ("乙",)),
        ),
        fact_entity_indexes={
            UUID(int=145): (0,),
            UUID(int=146): (1,),
        },
        fact_source_roles={
            UUID(int=145): Role.USER,
            UUID(int=146): Role.USER,
        },
    )

    await resolver.resolve(
        (),
        (
            make_fact(45, "该助手使用甲作为名字", ("甲",)),
            make_fact(46, "该助手的名字是乙", ("乙",)),
        ),
        context,
    )

    system_prompt = client.calls[0][0]["content"]
    assert "同一个唯一目标" in system_prompt
    assert "可能存在多个实例" in system_prompt
    assert "分别明确" in system_prompt


@pytest.mark.asyncio
async def test_resolution_request_preserves_recent_context() -> None:
    client = FakeJsonCompletionClient('{"groups":[]}')
    resolver = EntityResolver(client)
    context = SnapshotEntityContext(
        candidates=(
            EntityCandidate(UUID(int=47), ("西湖",)),
            EntityCandidate(UUID(int=48), ("West Lake",)),
        ),
        fact_entity_indexes={
            UUID(int=147): (0,),
            UUID(int=148): (1,),
        },
        fact_source_roles={
            UUID(int=147): Role.USER,
            UUID(int=148): Role.USER,
        },
        recent_context=(
            EntityContextMessage(Role.USER, "用户：西湖的英文怎么说？"),
            EntityContextMessage(Role.ASSISTANT, "助手：西湖的英文是 West Lake。"),
        ),
    )

    await resolver.resolve(
        (),
        (
            make_fact(47, "该用户询问西湖的英文表达", ("西湖",)),
            make_fact(48, "该用户询问West Lake是否好玩", ("West Lake",)),
        ),
        context,
    )

    payload = json.loads(client.calls[0][1]["content"])
    assert payload["recentContext"] == [
        {"role": "user", "content": "用户：西湖的英文怎么说？"},
        {"role": "assistant", "content": "助手：西湖的英文是 West Lake。"},
    ]
    assert all("recentContext" not in candidate for candidate in payload["candidates"])


@pytest.mark.asyncio
async def test_resolver_drops_unknown_or_single_candidate_groups() -> None:
    client = FakeJsonCompletionClient(
        '{"groups":[["c0","unknown"],["c1"],["c0","c1"]]}'
    )
    resolver = EntityResolver(client)
    context = SnapshotEntityContext(
        candidates=(
            EntityCandidate(UUID(int=50), ("a",)),
            EntityCandidate(UUID(int=51), ("b",)),
        ),
        fact_entity_indexes={UUID(int=150): (0,), UUID(int=151): (1,)},
    )

    result = await resolver.resolve(
        (),
        (
            make_fact(50, "a", ("a",)),
            make_fact(51, "b", ("b",)),
        ),
        context,
    )

    assert [group.aliases for group in result.groups] == [("a", "b")]
    assert len(result.dropped_groups) == 2
