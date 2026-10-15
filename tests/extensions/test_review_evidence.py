import copy
import json
from datetime import UTC, datetime, timedelta

import pytest

from fish_memory.extensions.review_evidence import pack_review_evidence


def event(index, text="还没测耐久，就先别下结论。"):
    return {
        "id": f"E{index}",
        "source": "plugin",
        "type": "live.idle",
        "at": (
            datetime(2026, 9, 11, tzinfo=UTC) + timedelta(seconds=index)
        ).isoformat(),
        "status": "completed",
        "text": text,
        "plan": {"point": "这句还没说出来"},
        "data": {"contribution": {"point": "这句还没说出来"}},
        "context_audit": {
            "activity": {"role": "主播"},
            "latest_user_input": "围绕手机聊",
        },
        "perception": {
            "available": True,
            "omitted_chars": 123,
            "earlier_excerpt": "之前看过的视频。" * 30,
            "text": "\n\n".join(
                ["电脑操作：仍在看视频。" * 40, "视频里的讲解者展示分屏。" * 80]
            ),
        },
    }


def test_shared_context_preserves_all_speech_and_user_instructions():
    events = [event(i) for i in range(1, 16)]
    instruction = {
        "id": "U1",
        "source": "user",
        "type": "chat.user",
        "at": events[0]["at"],
        "text": "继续聊手机，但可以改变刚才的判断。",
    }
    evidence = [instruction, *events]
    original = copy.deepcopy(evidence)
    result = pack_review_evidence(evidence)
    assert {e["id"] for e in result["evidence"]} == {e["id"] for e in evidence}
    assert [e["text"] for e in result["evidence"] if e["source"] == "plugin"] == [
        e["text"] for e in events
    ]
    assert len(result["contexts"]) == 4
    assert len(json.dumps(result, ensure_ascii=False)) < 12_000
    assert "这句还没说出来" not in json.dumps(result, ensure_ascii=False)
    assert evidence == original


def test_budget_discards_auxiliary_context_not_turns_or_instructions():
    evidence = [event(i) for i in range(3)]
    for i, item in enumerate(evidence):
        item["perception"]["text"] = str(i) + "长画面材料。" * 1000
    result = pack_review_evidence(evidence, max_chars=1800)
    assert len(result["evidence"]) == len(evidence)
    assert all(
        e["text"] == evidence[i]["text"] for i, e in enumerate(result["evidence"])
    )
    assert result["context_compression"]["omitted_fragments"] > 0
    assert any(e.get("context_budget_omitted") for e in result["evidence"])
    assert all(
        ref in result["contexts"]
        for e in result["evidence"]
        for ref in e.get("context_refs", [])
    )


def test_turns_only_reference_context_available_at_their_own_time():
    early, later = event(1), event(2)
    early["perception"]["text"] = "目前还没有展示分屏。"
    later["perception"]["text"] = "后来确实展示了分屏。"
    result = pack_review_evidence([early, later])
    before = [result["contexts"][ref] for ref in result["evidence"][0]["context_refs"]]
    after = [result["contexts"][ref] for ref in result["evidence"][1]["context_refs"]]
    assert "后来确实展示了分屏" not in str(before)
    assert "后来确实展示了分屏" in str(after)


def test_v5_audit_perception_uses_shared_fragments_without_changing_source():
    early, later = event(1), event(2)
    early["perception"]["text"] = "相同背景。\n\n还没有展示分屏。"
    later["perception"]["text"] = "相同背景。\n\n后来展示了分屏。"
    for item in (early, later):
        item["context_audit"]["perception"] = copy.deepcopy(item["perception"])
    original = copy.deepcopy([early, later])
    packed = pack_review_evidence([early, later])
    audits = [c for c in packed["contexts"].values() if c["kind"] == "context_audit"]
    assert len(audits) == 1 and "perception" not in audits[0]["value"]
    first = [packed["contexts"][ref] for ref in packed["evidence"][0]["context_refs"]]
    assert "还没有展示分屏" in str(first) and "后来展示" not in str(first)
    assert [early, later] == original
    # Older saved reports may contain the snapshot only inside the audit.
    del later["perception"]
    assert "后来展示了分屏" in str(pack_review_evidence([later]))


def test_oversized_speech_is_explicit_failure_instead_of_silent_sampling():
    with pytest.raises(ValueError, match="no turns were dropped"):
        pack_review_evidence([event(1, "实际发言" * 1000)], max_chars=100)


def test_external_message_data_is_kept_but_idle_plan_is_not():
    item = event(1)
    item["type"] = "live.reply"
    item["data"] = {"viewer_id": "viewer", "text": "我想听个不同的观点"}
    result = pack_review_evidence([item])
    assert result["evidence"][0]["data"] == item["data"]


def test_declared_references_are_review_material_not_spoken_content():
    from fish_memory.extensions.store import event_references

    source = {
        "research": [{"run_id": "run-a", "text": "外部资料"}],
        "plan": "准备说但没说的观点",
    }
    schema = {
        "properties": {
            "research": {"type": "array", "x-fish-evidence": "reference"},
            "plan": {"type": "string"},
        }
    }
    selected = event_references(source, schema)
    assert selected == {"research": source["research"]}
    packed = pack_review_evidence(
        [
            {
                "id": "E1",
                "source": "plugin",
                "at": "2026-09-12T00:00:00+00:00",
                "text": "实际说过的话",
                "references": selected,
                "data": source,
            }
        ]
    )
    assert packed["evidence"][0]["text"] == "实际说过的话"
    assert "准备说但没说" not in str(packed)
    assert any(c["kind"] == "event_references" for c in packed["contexts"].values())
