from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from fish_memory.observations.environment_history_models import SemanticDocument
from fish_memory.observations.environment_history_narrative import (
    NarrativeParagraph,
    parse_narrative,
    render_paragraphs,
)
from fish_memory.observations.environment_history_store import _redacted_document


def test_paragraph_provenance_is_not_rendered_as_a_state_or_unresolved_word_list():
    sources = {"a": "原话甲", "b": "原话乙", "c": "一句未引用的原话"}
    paragraphs = parse_narrative(
        json.dumps(
            {
                "paragraphs": [
                    {
                        "text": "先提出请求，后来被拒绝，事情仍未办成。",
                        "sources": [1, 2],
                    }
                ]
            }
        ),
        sources,
    )
    document = SemanticDocument(
        summary_version=3,
        audiovisual_summary=render_paragraphs(paragraphs),
        narrative_paragraphs=paragraphs,
        narrative_sources=sources,
    )
    assert document.audiovisual_summary == "先提出请求，后来被拒绝，事情仍未办成。"
    assert "原话" not in document.summary and "尚不确定" not in document.summary
    assert document.narrative_sources["c"] == "一句未引用的原话"
    encoded = document.model_dump_json()
    assert "audiovisualSummary" not in document.model_dump()
    assert encoded.count("原话甲") == 1
    assert SemanticDocument.model_validate_json(encoded) == document


@pytest.mark.parametrize("indexes", [[0], [4], [True], ["1"], [], [None]])
def test_invalid_current_source_cannot_be_promoted_into_a_narrative(indexes):
    with pytest.raises(ValueError):
        parse_narrative(
            json.dumps({"paragraphs": [{"text": "发生了。", "sources": indexes}]}),
            {"source": "原话"},
        )


def test_blank_story_and_missing_provenance_are_rejected():
    with pytest.raises(ValueError):
        parse_narrative('{"paragraphs":[{"text":"  ","sources":[1]}]}', {"a": "原话"})
    with pytest.raises(ValidationError):
        SemanticDocument(
            summary_version=3,
            narrative_paragraphs=(
                NarrativeParagraph(text="有人说话。", source_ids=("missing",)),
            ),
        )


def test_redaction_covers_prose_and_original_sources_without_changing_reference_ids():
    document = SemanticDocument(
        summary_version=3,
        narrative_paragraphs=(
            NarrativeParagraph(
                text="说出了 password=paragraph-secret", source_ids=("stable-source",)
            ),
        ),
        narrative_sources={"stable-source": "password=source-secret"},
    )
    redacted = _redacted_document(document)
    assert "paragraph-secret" not in redacted.summary
    assert "source-secret" not in redacted.model_dump_json()
    assert redacted.narrative_paragraphs[0].source_ids == ("stable-source",)
    assert set(redacted.narrative_sources) == {"stable-source"}
    assert SemanticDocument.model_validate_json(redacted.model_dump_json()) == redacted


def test_display_removes_only_redundant_machine_voice_annotations():
    sources = {
        "first": "[00:00.00—00:01.00] 声音abc-T01：请等一下。",
        "second": "[00:00.00—00:01.00] 声音abc-T02：好的。",
    }
    text = "来访者（声音abc-T01）请对方等待。声音abc-T02答应了。"
    paragraphs = parse_narrative(
        json.dumps({"paragraphs": [{"text": text, "sources": [1, 2]}]}), sources
    )
    assert paragraphs[0].text == "来访者请对方等待。声音abc-T02答应了。"
    assert paragraphs[0].source_ids == ("first", "second")
    assert "声音abc-T01" in sources["first"]


def test_actually_spoken_voice_label_is_not_deleted_from_narrative():
    sources = {"first": "[00:00.00—00:01.00] 声音abc-T01：标签叫声音abc-T01。"}
    text = "当事人说明了自己的标签（声音abc-T01）。"
    paragraphs = parse_narrative(
        json.dumps({"paragraphs": [{"text": text, "sources": [1]}]}), sources
    )
    assert paragraphs[0].text == text
