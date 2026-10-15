from uuid import UUID

import pytest
from pydantic import ValidationError

from fish_memory.answer.voice_draft_types import (
    VoiceDraftState,
    exact_content_hash,
    voice_answer_message_id,
)
from fish_memory.api.models import VoiceDraftUpsertRequest, VoiceTurnCommitRequest


def test_voice_draft_identity_is_stable_and_content_exact() -> None:
    turn_id = UUID("00000000-0000-0000-0000-000000000123")
    assert voice_answer_message_id(turn_id) == voice_answer_message_id(turn_id)
    assert voice_answer_message_id(turn_id) != turn_id
    assert exact_content_hash("鱼。") != exact_content_hash("鱼")


def test_voice_draft_request_rejects_revision_zero() -> None:
    with pytest.raises(ValidationError):
        VoiceDraftUpsertRequest(revision=0, content="问题")


def test_voice_draft_request_rejects_empty_or_oversized_content() -> None:
    with pytest.raises(ValidationError):
        VoiceDraftUpsertRequest(revision=1, content="")
    with pytest.raises(ValidationError):
        VoiceDraftUpsertRequest(revision=1, content="x" * 32_001)


@pytest.mark.parametrize("promote_revision", [None, 0])
def test_voice_commit_allows_no_promotion_lease(
    promote_revision: int | None,
) -> None:
    request = VoiceTurnCommitRequest(
        revision=2,
        promoteRevision=promote_revision,
        content="问题",
    )

    assert request.promote_revision == promote_revision


def test_voice_commit_requires_promotion_revision_before_commit() -> None:
    with pytest.raises(ValidationError):
        VoiceTurnCommitRequest(
            revision=2,
            promoteRevision=2,
            content="问题",
        )
    with pytest.raises(ValidationError):
        VoiceTurnCommitRequest(
            revision=2,
            promoteRevision=3,
            content="问题",
        )


def test_voice_draft_state_is_closed() -> None:
    assert VoiceDraftState.ACTIVE.value == "active"
