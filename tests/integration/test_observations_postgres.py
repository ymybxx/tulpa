import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

import asyncpg
import pytest

from fish_memory.db.migrations import MigrationPool, run_migrations
from fish_memory.observations.computer_history import (
    COMPUTER_HISTORY_FORMAT,
    COMPUTER_HISTORY_OBSERVATION_TYPE,
    COMPUTER_HISTORY_SCOPE_KEY,
    COMPUTER_HISTORY_SOURCE,
    ComputerHistorySegmentConflictError,
    ComputerHistorySummaryCheckpoint,
)
from fish_memory.observations.models import Observation, ObservationKind
from fish_memory.observations.service import ObservationService
from fish_memory.observations.store import (
    ObservationFishNotFoundError,
    ObservationStore,
)

FISH_ID = UUID("00000000-0000-0000-0000-000000000001")


async def migrated_service(
    pool: asyncpg.Pool,
) -> tuple[ObservationService, ObservationStore]:
    await run_migrations(cast(MigrationPool, pool))
    await pool.execute("INSERT INTO fishes (id) VALUES ($1)", FISH_ID)
    store = ObservationStore(pool)
    return ObservationService(store), store


@pytest.mark.asyncio
async def test_state_intervals_extend_and_events_remain_independent(
    postgres_pool: asyncpg.Pool,
) -> None:
    service, _store = await migrated_service(postgres_pool)
    started = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)

    first = await service.observe_state(
        FISH_ID,
        source="workspace",
        observation_type="activity",
        scope_key="active-window",
        semantic_key="editing:fish-memory",
        summary="Editing Fish Memory",
        payload={"document": "observations.py"},
        app_bundle_id="com.microsoft.VSCode",
        app_name="Visual Studio Code",
        observed_at=started,
    )
    extended = await service.observe_state(
        FISH_ID,
        source="workspace",
        observation_type="activity",
        scope_key="active-window",
        semantic_key="editing:fish-memory",
        summary="Editing Observation tests",
        payload={"document": "test_observations_postgres.py"},
        app_bundle_id="com.microsoft.vscode",
        app_name="Visual Studio Code",
        observed_at=started + timedelta(seconds=20),
    )
    changed = await service.observe_state(
        FISH_ID,
        source="workspace",
        observation_type="activity",
        scope_key="active-window",
        semantic_key="running-tests",
        summary="Running focused tests",
        app_bundle_id="com.apple.terminal",
        app_name="Terminal",
        observed_at=started + timedelta(seconds=21),
    )
    event = await service.record_important_event(
        FISH_ID,
        source="accessibility",
        observation_type="error",
        semantic_key="test-failure",
        summary="A focused test failed",
        app_bundle_id="com.apple.terminal",
        app_name="Terminal",
        occurred_at=started + timedelta(seconds=25),
    )

    assert first is not None
    assert extended is not None
    assert changed is not None
    assert event is not None
    assert extended.id == first.id
    assert extended.started_at == started
    assert extended.ended_at == started + timedelta(seconds=20)
    assert changed.id != first.id
    assert event.kind is ObservationKind.EVENT
    assert await service.latest_for_stream(FISH_ID, first) == changed

    counts = await postgres_pool.fetchrow(
        """
        SELECT
            (SELECT count(*) FROM observations) AS observations,
            (SELECT count(*) FROM messages) AS messages,
            (SELECT count(*) FROM facts) AS facts,
            (SELECT count(*) FROM memory_snapshots) AS snapshots
        """
    )
    assert counts is not None
    assert counts["observations"] == 3
    assert counts["messages"] == 0
    assert counts["facts"] == 0
    assert counts["snapshots"] == 0


@pytest.mark.asyncio
async def test_queries_delete_and_app_policy_are_fish_scoped(
    postgres_pool: asyncpg.Pool,
) -> None:
    service, store = await migrated_service(postgres_pool)
    now = datetime(2026, 9, 1, 8, 0, tzinfo=UTC)
    blocked_app = "com.example.private"

    assert await store.is_app_recording_enabled(FISH_ID, blocked_app)
    policy = await service.set_app_recording(
        FISH_ID,
        blocked_app,
        enabled=False,
        app_name="Private App",
    )
    assert not policy.recording_enabled
    assert not await store.is_app_recording_enabled(FISH_ID, blocked_app)
    assert not await store.is_app_recording_enabled(
        FISH_ID,
        " COM.Example.Private ",
    )
    with pytest.raises(ValueError, match="valid bundle identifier"):
        await store.is_app_recording_enabled(FISH_ID, "not a bundle id")
    with pytest.raises(ObservationFishNotFoundError):
        await store.is_app_recording_enabled(UUID(int=999), blocked_app)
    blocked = await service.record_important_event(
        FISH_ID,
        source="accessibility",
        observation_type="error",
        semantic_key="private-error",
        summary="Must not be stored",
        app_bundle_id=blocked_app,
        occurred_at=now,
    )
    assert blocked is None

    await service.set_app_recording(FISH_ID, blocked_app, enabled=True)
    stored = await service.record_important_event(
        FISH_ID,
        source="accessibility",
        observation_type="error",
        semantic_key="public-error",
        summary="An allowed event",
        app_bundle_id=blocked_app,
        occurred_at=now + timedelta(minutes=1),
    )
    assert stored is not None

    results = await service.query(
        FISH_ID,
        start_at=now,
        end_at=now + timedelta(minutes=2),
        app_bundle_id=blocked_app,
        observation_type="error",
    )
    assert [item.id for item in results] == [stored.id]
    assert await service.delete(FISH_ID, stored.id)
    assert await service.get(FISH_ID, stored.id) is None
    assert not await service.delete(FISH_ID, uuid4())


@pytest.mark.asyncio
async def test_computer_history_is_idempotent_and_memory_pipeline_isolated(
    postgres_pool: asyncpg.Pool,
) -> None:
    _service, store = await migrated_service(postgres_pool)
    segment_id = uuid4()
    started = datetime(2026, 9, 2, 8, 0, tzinfo=UTC)
    ended = started + timedelta(minutes=10)
    digest = "a" * 64
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    payload: dict[str, object] = {
        "schemaVersion": 1,
        "format": COMPUTER_HISTORY_FORMAT,
        "documentMarkdown": f"# Summary\nAPI_KEY={secret}\n{digest}\n",
        "title": f"Summary token {secret}",
        "description": "Description",
        "applications": ["com.example.editor"],
        "segmentId": str(segment_id),
        "segmentStartedAt": started.isoformat().replace("+00:00", "Z"),
        "segmentEndedAt": ended.isoformat().replace("+00:00", "Z"),
        "eventCount": 1,
        "suppressedEventCount": 0,
        "eventDigest": digest,
        "citations": [
            f"segment:{segment_id}",
            f"event-digest:sha256:{digest}",
        ],
    }

    async def write(
        *,
        event_digest: str = digest,
        value: Mapping[str, object] = payload,
    ) -> Observation:
        return await store.record_computer_history_segment(
            FISH_ID,
            segment_id=segment_id,
            started_at=started,
            ended_at=ended,
            event_digest=event_digest,
            summary=f"Description\n\nAPI_KEY={secret}",
            payload=value,
        )

    first, retry = await asyncio.gather(write(), write())
    recovered = await store.get_computer_history_segment(FISH_ID, segment_id)
    inherited = await store.list_recent_computer_history_summaries(
        FISH_ID,
        before=ended,
        limit=3,
    )

    assert retry.id == first.id
    assert recovered == first
    assert inherited == (first,)
    assert first.kind is ObservationKind.STATE
    assert first.source == COMPUTER_HISTORY_SOURCE
    assert first.observation_type == COMPUTER_HISTORY_OBSERVATION_TYPE
    assert first.scope_key == COMPUTER_HISTORY_SCOPE_KEY
    assert first.semantic_key == f"segment:{segment_id}"
    assert first.started_at == started
    assert first.ended_at == ended
    assert secret not in first.summary
    assert secret not in str(first.payload)
    assert digest in str(first.payload["documentMarkdown"])

    conflicting_digest = {**payload, "eventDigest": "b" * 64}
    with pytest.raises(ComputerHistorySegmentConflictError):
        await write(event_digest="b" * 64, value=conflicting_digest)

    conflicting_suppressed = {**payload, "suppressedEventCount": 1}
    with pytest.raises(ComputerHistorySegmentConflictError):
        await write(value=conflicting_suppressed)

    counts = await postgres_pool.fetchrow(
        """
        SELECT
            (SELECT count(*) FROM observations) AS observations,
            (SELECT count(*) FROM messages) AS messages,
            (SELECT count(*) FROM facts) AS facts,
            (SELECT count(*) FROM fact_embeddings) AS fact_embeddings,
            (SELECT count(*) FROM memory_snapshots) AS snapshots,
            (SELECT count(*) FROM memory_snapshot_items) AS recall_items,
            (SELECT count(*) FROM memory_snapshot_item_embeddings)
                AS recall_embeddings,
            (SELECT count(*) FROM background_assistant_deliveries)
                AS proactive_deliveries
        """
    )
    assert counts is not None
    assert dict(counts) == {
        "observations": 1,
        "messages": 0,
        "facts": 0,
        "fact_embeddings": 0,
        "snapshots": 0,
        "recall_items": 0,
        "recall_embeddings": 0,
        "proactive_deliveries": 0,
    }


@pytest.mark.asyncio
async def test_computer_history_checkpoint_cas_and_finalization_are_atomic(
    postgres_pool: asyncpg.Pool,
) -> None:
    _service, store = await migrated_service(postgres_pool)
    segment_id = uuid4()
    started = datetime(2026, 9, 2, 9, 0, tzinfo=UTC)
    ended = started + timedelta(minutes=10)
    request_id = uuid4()
    delta_digest = "a" * 64
    chain_digest = "b" * 64
    final_digest = "c" * 64
    observation_id = uuid4()
    summary = {
        "title": "Project work",
        "description": "The user edited a local project.",
        "memorySummary": "The user reviewed and edited project files.",
        "relevantPriorContext": [],
        "importantContext": [],
        "recordingSections": [
            {"heading": "Editing", "details": ["Reviewed project files."]}
        ],
        "suggestion": None,
    }

    def checkpoint(
        *,
        revision: int,
        status: str = "active",
    ) -> ComputerHistorySummaryCheckpoint:
        finalized = status == "finalized"
        return ComputerHistorySummaryCheckpoint.model_validate(
            {
                "schemaVersion": 1,
                "fishId": str(FISH_ID),
                "segmentId": str(segment_id),
                "startedAt": started.isoformat(),
                "endedAt": ended.isoformat(),
                "revision": revision,
                "throughEventCount": 1,
                "throughByteOffset": 256,
                "throughSuppressedEventCount": 2,
                "deltaChainDigest": chain_digest,
                "fullSummary": summary,
                "applications": ["Editor"],
                "inheritedObservationIds": [],
                "deltaCapsules": [
                    {
                        "requestId": str(request_id),
                        "fromEventCount": 0,
                        "throughEventCount": 1,
                        "deltaDigest": delta_digest,
                        "summary": summary,
                        "applications": ["Editor"],
                        "createdAt": (started + timedelta(minutes=1)).isoformat(),
                    }
                ],
                "lastRequestId": str(request_id),
                "lastDeltaDigest": delta_digest,
                "lastSuccessfulSummaryAt": (started + timedelta(minutes=1)).isoformat(),
                "status": status,
                "finalizedAt": ended.isoformat() if finalized else None,
                "observationId": str(observation_id) if finalized else None,
                "finalEventDigest": final_digest if finalized else None,
                "createdAt": started.isoformat(),
                "updatedAt": ended.isoformat() if finalized else started.isoformat(),
            }
        )

    active = checkpoint(revision=1)
    stored = await store.save_computer_history_checkpoint(
        active,
        expected_revision=0,
    )
    latest = await store.get_latest_computer_history_checkpoint(FISH_ID)

    assert stored == active
    assert latest == active
    assert (
        await store.delete_stale_active_computer_history_checkpoints(
            FISH_ID,
            updated_before=started,
        )
        == 0
    )
    assert (
        await store.delete_stale_active_computer_history_checkpoints(
            FISH_ID,
            updated_before=started + timedelta(seconds=1),
        )
        == 1
    )
    assert await store.get_computer_history_checkpoint(FISH_ID, segment_id) is None
    stored = await store.save_computer_history_checkpoint(
        active,
        expected_revision=0,
    )
    with pytest.raises(ComputerHistorySegmentConflictError, match="acknowledgement"):
        await store.acknowledge_finalized_computer_history_checkpoint(
            FISH_ID,
            segment_id,
            observation_id=observation_id,
            final_event_digest=final_digest,
        )
    with pytest.raises(ComputerHistorySegmentConflictError, match="stale"):
        await store.save_computer_history_checkpoint(active, expected_revision=0)

    final = checkpoint(revision=2, status="finalized")
    payload: dict[str, object] = {
        "schemaVersion": 1,
        "format": COMPUTER_HISTORY_FORMAT,
        "documentMarkdown": "# Project work\n",
        "title": "Project work",
        "description": "The user edited a local project.",
        "applications": ["Editor"],
        "segmentId": str(segment_id),
        "segmentStartedAt": started.isoformat().replace("+00:00", "Z"),
        "segmentEndedAt": ended.isoformat().replace("+00:00", "Z"),
        "eventCount": 1,
        "suppressedEventCount": 2,
        "eventDigest": final_digest,
        "checkpointRevision": 2,
        "citations": [f"segment:{segment_id}"],
    }
    finalized, observation = await store.finalize_computer_history_checkpoint(
        final,
        expected_revision=1,
        event_digest=final_digest,
        summary="The user edited a local project.",
        payload=payload,
    )
    retried, retried_observation = await store.finalize_computer_history_checkpoint(
        final,
        expected_revision=1,
        event_digest=final_digest,
        summary="The user edited a local project.",
        payload=payload,
    )

    assert finalized.status == "finalized"
    assert finalized.observation_id == observation_id
    assert observation.id == observation_id
    assert retried == finalized
    assert retried_observation == observation
    assert await store.get_latest_computer_history_checkpoint(FISH_ID) is None
    assert await store.get_computer_history_segment(FISH_ID, segment_id) == observation
    assert await store.acknowledge_finalized_computer_history_checkpoint(
        FISH_ID,
        segment_id,
        observation_id=observation_id,
        final_event_digest=final_digest,
    )
    assert not await store.acknowledge_finalized_computer_history_checkpoint(
        FISH_ID,
        segment_id,
        observation_id=observation_id,
        final_event_digest=final_digest,
    )
    assert await store.get_computer_history_segment(FISH_ID, segment_id) == observation
    counts = await postgres_pool.fetchrow(
        """
        SELECT
            (SELECT count(*) FROM computer_history_summary_checkpoints) checkpoints,
            (SELECT count(*) FROM observations) observations,
            (SELECT count(*) FROM messages) messages,
            (SELECT count(*) FROM facts) facts,
            (SELECT count(*) FROM memory_snapshots) snapshots
        """
    )
    assert counts is not None
    assert dict(counts) == {
        "checkpoints": 0,
        "observations": 1,
        "messages": 0,
        "facts": 0,
        "snapshots": 0,
    }
