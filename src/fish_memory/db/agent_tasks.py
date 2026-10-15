"""Persistence for Agent task, run, and interaction lifecycles."""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from datetime import datetime
from typing import cast
from uuid import UUID, uuid4

import asyncpg

from fish_memory.agent.models import (
    AgentActivationPolicy,
    AgentEndpoint,
    AgentHistorySource,
    AgentInteraction,
    AgentInteractionKind,
    AgentInteractionWorkItem,
    AgentMessageEnvelope,
    AgentMessageSubject,
    AgentMessageSubjectKind,
    AgentRun,
    AgentRunStatus,
    AgentRunWorkItem,
    AgentTarget,
    AgentTask,
    AgentTaskCandidate,
    ConnectorTypeSlug,
)
from fish_memory.agent.replies import (
    AgentReplyConflictError,
    AgentReplyContext,
    AgentReplyMutation,
    AgentReplyReplay,
    AgentReplySubjectSnapshot,
)
from fish_memory.background_messages.store import BackgroundMessageStore
from fish_memory.db.json_values import canonical_json_object, canonical_json_size
from fish_memory.db.messages import MessageIdentityConflictError, MessagesStore
from fish_memory.models import Message, MessageStatus, Role

_MAX_TITLE_CHARS = 200
_MAX_MATERIALIZED_CHARS = 32_000
_MAX_EXTERNAL_ID_CHARS = 512
_MAX_ERROR_CHARS = 4_000
_MAX_PROVIDER_BYTES = 65_536
_RECOVERY_METADATA_RESERVE_BYTES = canonical_json_size(
    {
        "state": "unknown",
        "error_code": "\x01" * _MAX_EXTERNAL_ID_CHARS,
        "error_message": "\x01" * _MAX_ERROR_CHARS,
    }
) - canonical_json_size({"state": "queued"})
_ACTIVE_STATUS_VALUES = (
    "queued",
    "dispatching",
    "running",
    "waiting_user",
    "unknown",
)
_MUTABLE_ACTIVE_STATUS_VALUES = (
    "dispatching",
    "running",
    "waiting_user",
    "unknown",
)

_TASK_COLUMNS = """
    id, fish_id, session_id, target_id, external_session_id, title, created_at,
    updated_at
"""
_RUN_COLUMNS = """
    id, task_id, target_id, fish_id, session_id, previous_run_id, source_message_id,
confirmation_message_id, input, external_run_id, history_source, status,
output, result_message_id, error_code, error_message, provider_data,
created_at, started_at, updated_at, finished_at, origin, origin_context
"""
_INTERACTION_COLUMNS = """
    id, run_id, fish_id, session_id, kind, request_message_id,
    response_message_id, next_run_id,
provider_data, created_at, resolved_at, closed_at
"""
_QUALIFIED_RUN_COLUMNS = ", ".join(
    "run." + item.strip() for item in _RUN_COLUMNS.split(",")
)
_WORK_COLUMNS = """
    run.id AS run_id,
    run.task_id AS run_task_id,
    run.target_id AS run_target_id,
    run.fish_id AS run_fish_id,
    run.session_id AS run_session_id,
run.previous_run_id AS run_previous_run_id,
run.source_message_id AS run_source_message_id,
run.confirmation_message_id AS run_confirmation_message_id,
run.input AS run_input,
run.external_run_id AS run_external_run_id,
run.history_source AS run_history_source,
run.status AS run_status,
run.output AS run_output,
run.result_message_id AS run_result_message_id,
run.error_code AS run_error_code,
run.error_message AS run_error_message,
run.provider_data AS run_provider_data,
run.created_at AS run_created_at,
run.started_at AS run_started_at,
run.updated_at AS run_updated_at,
run.finished_at AS run_finished_at,
run.origin AS run_origin,
run.origin_context AS run_origin_context,
    task.id AS task_id,
    task.fish_id AS task_fish_id,
    task.session_id AS task_session_id,
task.target_id AS task_target_id,
task.external_session_id AS task_external_session_id,
task.title AS task_title,
task.created_at AS task_created_at,
task.updated_at AS task_updated_at,
target.id AS target_id,
target.endpoint_id AS target_endpoint_id,
target.external_key AS target_external_key,
target.route_prefix AS target_route_prefix,
target.credential_ref AS target_credential_ref,
target.connector_config AS target_connector_config,
target.display_name AS target_display_name,
target.enabled AS target_enabled,
target.created_at AS target_created_at,
target.updated_at AS target_updated_at,
endpoint.id AS endpoint_id,
endpoint.connector_type AS endpoint_connector_type,
endpoint.base_url AS endpoint_base_url,
endpoint.created_at AS endpoint_created_at,
endpoint.updated_at AS endpoint_updated_at
"""
_CANDIDATE_INTERACTION_COLUMNS = """
    interaction.id AS interaction_id,
    interaction.run_id AS interaction_run_id,
    interaction.fish_id AS interaction_fish_id,
    interaction.session_id AS interaction_session_id,
interaction.kind AS interaction_kind,
interaction.request_message_id AS interaction_request_message_id,
interaction.response_message_id AS interaction_response_message_id,
interaction.next_run_id AS interaction_next_run_id,
interaction.provider_data AS interaction_provider_data,
interaction.created_at AS interaction_created_at,
interaction.resolved_at AS interaction_resolved_at,
interaction.closed_at AS interaction_closed_at
"""
_MAX_TASK_CANDIDATES = 8
_MAX_RECENT_TERMINAL_CANDIDATES = 5
_MAX_TASK_CANDIDATE_REQUEST_CHARS = 4_000
_MAX_RECENT_AGENT_REPLY_LOOKBACK = 20


class AgentRunConflictError(AgentReplyConflictError):
    pass


class AgentInteractionConflictError(AgentReplyConflictError):
    pass


class AgentMessageSubjectConflictError(AgentReplyConflictError):
    pass


class TaskStore:
    def __init__(
        self,
        pool: asyncpg.Pool,
        *,
        messages_store: MessagesStore | None = None,
        background_message_store: BackgroundMessageStore | None = None,
    ) -> None:
        self._pool = pool
        self._messages = messages_store or MessagesStore(pool)
        self._background_messages = background_message_store or BackgroundMessageStore(
            pool,
            messages_store=self._messages,
        )

    async def load_reply_subject(
        self,
        context: AgentReplyContext,
        fish_id: UUID,
        session_id: UUID,
        *,
        message_id: UUID | None = None,
        content: str | None = None,
    ) -> AgentReplySubjectSnapshot:
        """Load one bounded exact subject or a durable replay disposition."""

        if context.kind is AgentMessageSubjectKind.INTERACTION:
            return await self._load_interaction_reply_subject(
                context,
                fish_id,
                session_id,
                message_id=message_id,
                content=content,
            )
        if context.kind is AgentMessageSubjectKind.RUN_RESULT:
            return await self._load_result_reply_subject(
                context,
                fish_id,
                session_id,
                message_id=message_id,
                content=content,
            )
        raise AgentReplyConflictError("reply subject kind is unsupported")

    async def _load_interaction_reply_subject(
        self,
        context: AgentReplyContext,
        fish_id: UUID,
        session_id: UUID,
        *,
        message_id: UUID | None,
        content: str | None,
    ) -> AgentReplySubjectSnapshot:
        row = await self._pool.fetchrow(
            """
            SELECT interaction.kind AS interaction_kind,
                   interaction.response_message_id,
                   interaction.next_run_id,
                   interaction.provider_data,
                   interaction.resolved_at,
                   interaction.closed_at,
                   task.fish_id AS task_fish_id,
                   request.id AS source_message_id,
                   LEFT(request.content, $5) AS source_content,
                   char_length(request.content) AS source_content_length,
                   request.fish_id AS source_fish_id,
                   request.session_id AS source_session_id,
                   request.role AS source_role,
                   request.status AS source_status,
                   request.consumed_offset AS source_consumed_offset,
                   request.fact_extraction_status AS source_fact_status,
                   existing.id AS existing_id,
                   existing.fish_id AS existing_fish_id,
                   existing.session_id AS existing_session_id,
                   existing.sequence AS existing_sequence,
                   existing.role AS existing_role,
                   existing.content AS existing_content,
                   existing.status AS existing_status,
                   existing.created_at AS existing_created_at,
                   existing.context_through_sequence AS existing_context_sequence,
                   existing.consumed_offset AS existing_consumed_offset,
                   existing.finished_at AS existing_finished_at,
                   existing.history_token_count AS existing_history_tokens
            FROM agent_interactions AS interaction
            JOIN agent_runs AS run ON run.id = interaction.run_id
            JOIN agent_tasks AS task ON task.id = run.task_id
            LEFT JOIN messages AS request
              ON request.id = interaction.request_message_id
            LEFT JOIN messages AS existing ON existing.id = $4
            WHERE interaction.id = $1
              AND task.fish_id = $2
              AND task.session_id = $3
              AND request.session_id = $3
            """,
            context.id,
            fish_id,
            session_id,
            message_id,
            _MAX_MATERIALIZED_CHARS,
        )
        if row is None:
            raise AgentReplyConflictError("interaction is not replyable")
        _require_reply_source(row, fish_id, session_id)
        existing = _existing_reply_message(
            row,
            fish_id,
            session_id,
            message_id=message_id,
            content=content,
        )
        response_message_id = cast(UUID | None, row["response_message_id"])
        if existing is not None:
            interaction_kind = AgentInteractionKind(cast(str, row["interaction_kind"]))
            handled = await self._authoritative_reply_handled(
                self._pool,
                existing.id,
                context,
            )
            return AgentReplySubjectSnapshot(
                context=context,
                source_message_id=cast(UUID, row["source_message_id"]),
                source_content=cast(str, row["source_content"]),
                interaction_kind=interaction_kind,
                allowed_decisions=_allowed_decisions(row["provider_data"]),
                recovered_without_details=_recovered_without_details(
                    row["provider_data"]
                ),
                replay=AgentReplyReplay(existing, handled),
            )
        if (
            response_message_id is not None
            or row["resolved_at"] is not None
            or row["closed_at"] is not None
        ):
            raise AgentReplyConflictError("interaction is not replyable")
        return AgentReplySubjectSnapshot(
            context=context,
            source_message_id=cast(UUID, row["source_message_id"]),
            source_content=cast(str, row["source_content"]),
            interaction_kind=AgentInteractionKind(cast(str, row["interaction_kind"])),
            allowed_decisions=_allowed_decisions(row["provider_data"]),
            recovered_without_details=_recovered_without_details(row["provider_data"]),
            replay=None,
        )

    async def _load_result_reply_subject(
        self,
        context: AgentReplyContext,
        fish_id: UUID,
        session_id: UUID,
        *,
        message_id: UUID | None,
        content: str | None,
    ) -> AgentReplySubjectSnapshot:
        row = await self._pool.fetchrow(
            """
            SELECT run.status AS run_status,
                   task.fish_id AS task_fish_id,
                   result.id AS source_message_id,
                   LEFT(result.content, $5) AS source_content,
                   char_length(result.content) AS source_content_length,
                   result.fish_id AS source_fish_id,
                   result.session_id AS source_session_id,
                   result.role AS source_role,
                   result.status AS source_status,
                   result.consumed_offset AS source_consumed_offset,
                   result.fact_extraction_status AS source_fact_status,
                   child.id AS child_run_id,
                   child.source_message_id AS child_source_message_id,
                   existing.id AS existing_id,
                   existing.fish_id AS existing_fish_id,
                   existing.session_id AS existing_session_id,
                   existing.sequence AS existing_sequence,
                   existing.role AS existing_role,
                   existing.content AS existing_content,
                   existing.status AS existing_status,
                   existing.created_at AS existing_created_at,
                   existing.context_through_sequence AS existing_context_sequence,
                   existing.consumed_offset AS existing_consumed_offset,
                   existing.finished_at AS existing_finished_at,
                   existing.history_token_count AS existing_history_tokens
            FROM agent_runs AS run
            JOIN agent_tasks AS task ON task.id = run.task_id
            LEFT JOIN messages AS result ON result.id = run.result_message_id
            LEFT JOIN agent_runs AS child ON child.previous_run_id = run.id
            LEFT JOIN messages AS existing ON existing.id = $4
            WHERE run.id = $1
              AND task.fish_id = $2
              AND task.session_id = $3
              AND result.session_id = $3
            """,
            context.id,
            fish_id,
            session_id,
            message_id,
            _MAX_MATERIALIZED_CHARS,
        )
        if row is None:
            raise AgentReplyConflictError("run result is not replyable")
        _require_reply_source(row, fish_id, session_id)
        if row["run_status"] not in {"completed", "failed", "cancelled"}:
            raise AgentReplyConflictError("run result is not replyable")
        existing = _existing_reply_message(
            row,
            fish_id,
            session_id,
            message_id=message_id,
            content=content,
        )
        if existing is not None:
            handled = await self._authoritative_reply_handled(
                self._pool,
                existing.id,
                context,
            )
            return AgentReplySubjectSnapshot(
                context=context,
                source_message_id=cast(UUID, row["source_message_id"]),
                source_content=cast(str, row["source_content"]),
                interaction_kind=None,
                allowed_decisions=(),
                recovered_without_details=False,
                replay=AgentReplyReplay(existing, handled),
            )
        if row["child_run_id"] is not None:
            raise AgentReplyConflictError("run result is not replyable")
        return AgentReplySubjectSnapshot(
            context=context,
            source_message_id=cast(UUID, row["source_message_id"]),
            source_content=cast(str, row["source_content"]),
            interaction_kind=None,
            allowed_decisions=(),
            recovered_without_details=False,
            replay=None,
        )

    async def create_task_with_first_run(
        self,
        fish_id: UUID,
        target_id: UUID,
        title: str,
        source_message_id: UUID,
        confirmation_message_id: UUID | None,
        input: str,
    ) -> tuple[AgentTask, AgentRun]:
        _bounded("title", title, _MAX_TITLE_CHARS)
        _bounded("input", input, _MAX_MATERIALIZED_CHARS)
        task_id = uuid4()
        run_id = uuid4()
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                target = await connection.fetchrow(
                    """
                    SELECT id
                    FROM agent_targets
                    WHERE id = $1 AND enabled
                    FOR SHARE
                    """,
                    target_id,
                )
                if target is None:
                    raise AgentRunConflictError("Agent target is unavailable")
                task_session_id = await self._lock_run_messages(
                    connection,
                    fish_id,
                    source_message_id,
                    confirmation_message_id,
                )
                task_row = await connection.fetchrow(
                    f"""
                    INSERT INTO agent_tasks (
                        id, fish_id, session_id, target_id, external_session_id, title
                    )
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING {_TASK_COLUMNS}
                    """,
                    task_id,
                    fish_id,
                    task_session_id,
                    target_id,
                    f"fm-task-{task_id}",
                    title,
                )
                run_row = await connection.fetchrow(
                    f"""
                    INSERT INTO agent_runs (
                        id, task_id, target_id, fish_id, session_id, source_message_id,
                        confirmation_message_id, input, status
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'queued')
                    RETURNING {_RUN_COLUMNS}
                    """,
                    run_id,
                    task_id,
                    target_id,
                    fish_id,
                    task_session_id,
                    source_message_id,
                    confirmation_message_id,
                    input,
                )
        return _required_task(task_row), _required_run(run_row)

    async def create_continuation_run(
        self,
        task_id: UUID,
        expected_previous_run_id: UUID,
        source_message_id: UUID,
        input: str,
        confirmation_message_id: UUID | None,
    ) -> AgentRun:
        _bounded("input", input, _MAX_MATERIALIZED_CHARS)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                task = await self._lock_task_with_enabled_target(connection, task_id)
                if task is None:
                    raise AgentRunConflictError("Agent task target is unavailable")
                tails = await connection.fetch(
                    f"""
                    SELECT {_QUALIFIED_RUN_COLUMNS}
                    FROM agent_runs AS run
                    WHERE run.task_id = $1
                      AND NOT EXISTS (
                          SELECT 1
                          FROM agent_runs AS child
                          WHERE child.previous_run_id = run.id
                      )
                    FOR UPDATE OF run
                    """,
                    task_id,
                )
                if len(tails) != 1:
                    raise AgentRunConflictError(
                        "Agent task does not have exactly one lineage tail"
                    )
                previous_run = _run_from_record(tails[0])
                if previous_run.id != expected_previous_run_id:
                    raise AgentRunConflictError("Agent task lineage tail changed")
                if previous_run.status.value in _ACTIVE_STATUS_VALUES:
                    raise AgentRunConflictError("Agent task already has an active run")
                await self._lock_run_messages(
                    connection,
                    cast(UUID, task["fish_id"]),
                    source_message_id,
                    confirmation_message_id,
                    expected_session_id=cast(UUID, task["session_id"]),
                )
                row = await self._insert_run(
                    connection,
                    task_id=task_id,
                    target_id=cast(UUID, task["target_id"]),
                    fish_id=cast(UUID, task["fish_id"]),
                    session_id=cast(UUID, task["session_id"]),
                    previous_run_id=previous_run.id,
                    source_message_id=source_message_id,
                    confirmation_message_id=confirmation_message_id,
                    input=input,
                )
                return row

    async def list_task_candidates(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        recent_terminal_limit: int = _MAX_RECENT_TERMINAL_CANDIDATES,
        limit: int = _MAX_TASK_CANDIDATES,
        supported_connector_types: Collection[str] | None = None,
    ) -> Sequence[AgentTaskCandidate]:
        """Load one stable lineage-tail candidate per Task in one Fish Session.

        Active Runs and open Interactions are always prioritized. Fully consumed
        terminal results contribute only the newest bounded tail set. The final
        list is active-first, newest-first within each group, connector-filtered,
        and capped by the requested limit.
        """

        _bounded_integer(
            "recent_terminal_limit",
            recent_terminal_limit,
            minimum=0,
            maximum=_MAX_RECENT_TERMINAL_CANDIDATES,
        )
        _bounded_integer("limit", limit, minimum=1, maximum=_MAX_TASK_CANDIDATES)
        connector_types = _validated_connector_type_filter(supported_connector_types)
        active_statuses = (
            "(" + ", ".join(f"'{status}'" for status in _ACTIVE_STATUS_VALUES) + ")"
        )
        rows = await self._pool.fetch(
            f"""
            WITH recent_assistant AS MATERIALIZED (
                SELECT id, sequence
                FROM messages
                WHERE fish_id = $1
                  AND session_id = $2
                  AND role = 'assistant'
                  AND status IN ('completed', 'interrupted')
                ORDER BY sequence DESC
                LIMIT $3
            ),
            recent_task_reply_rows AS (
                SELECT run.task_id, recent.sequence AS reply_sequence
                FROM recent_assistant AS recent
                JOIN agent_runs AS run
                  ON run.confirmation_message_id = recent.id
                WHERE run.fish_id = $1
                  AND run.session_id = $2
                UNION ALL
                SELECT run.task_id, recent.sequence AS reply_sequence
                FROM recent_assistant AS recent
                JOIN agent_runs AS run
                  ON run.result_message_id = recent.id
                WHERE run.fish_id = $1
                  AND run.session_id = $2
                UNION ALL
                SELECT run.task_id, recent.sequence AS reply_sequence
                FROM recent_assistant AS recent
                JOIN agent_interactions AS interaction
                  ON interaction.request_message_id = recent.id
                JOIN agent_runs AS run
                  ON run.id = interaction.run_id
                WHERE interaction.fish_id = $1
                  AND interaction.session_id = $2
            ),
            recent_task_replies AS (
                SELECT task_id, max(reply_sequence) AS recent_reply_sequence
                FROM recent_task_reply_rows
                GROUP BY task_id
            ),
            session_interactions AS (
                SELECT interaction.*,
                       LEFT(
                           request.content,
                           {_MAX_TASK_CANDIDATE_REQUEST_CHARS}
                       ) AS request_content,
                       request.session_id AS request_session_id,
                       request.sequence AS request_sequence,
                       row_number() OVER (
                           ORDER BY request.sequence DESC, interaction.id
                       ) AS session_rank
                FROM agent_interactions AS interaction
                JOIN agent_runs AS interaction_run
                  ON interaction_run.id = interaction.run_id
                JOIN agent_tasks AS interaction_task
                  ON interaction_task.id = interaction_run.task_id
                JOIN messages AS request
                  ON request.id = interaction.request_message_id
                WHERE interaction_task.fish_id = $1
                  AND interaction_task.session_id = $2
                  AND request.session_id = $2
            ),
            focused_interactions AS (
                SELECT session_interactions.*,
                       row_number() OVER (
                           PARTITION BY run_id
                           ORDER BY request_sequence DESC, id
                       ) AS focus_rank
                FROM session_interactions
                WHERE response_message_id IS NULL
                  AND resolved_at IS NULL
                  AND closed_at IS NULL
                  AND (
                      created_at >= now() - interval '1 hour'
                      OR session_rank = 1
                  )
            ),
            lineage_tails AS (
                SELECT run.*
                FROM agent_runs AS run
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM agent_runs AS child
                    WHERE child.previous_run_id = run.id
                )
            ),
            scoped AS (
                SELECT
                    {_WORK_COLUMNS},
                    {_CANDIDATE_INTERACTION_COLUMNS},
                    interaction.request_content AS interaction_request_content,
                    interaction.request_session_id AS interaction_request_session_id,
                    recent_reply.recent_reply_sequence,
                    CASE
                        WHEN run.status IN {active_statuses}
                          OR interaction.id IS NOT NULL
                        THEN 0
                        ELSE 1
                    END AS candidate_priority
                FROM lineage_tails AS run
                JOIN agent_tasks AS task ON task.id = run.task_id
                JOIN agent_targets AS target ON target.id = run.target_id
                JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
                JOIN messages AS source ON source.id = run.source_message_id
                LEFT JOIN focused_interactions AS interaction
                  ON interaction.run_id = run.id
                 AND interaction.focus_rank = 1
                LEFT JOIN messages AS result ON result.id = run.result_message_id
                LEFT JOIN recent_task_replies AS recent_reply
                  ON recent_reply.task_id = task.id
                WHERE task.fish_id = $1
                  AND task.session_id = $2
                  AND source.session_id = $2
                  AND target.enabled
                  AND (
                      $5::text[] IS NULL
                      OR endpoint.connector_type = ANY($5::text[])
                  )
                  AND (
                      run.status IN {active_statuses}
                      OR interaction.id IS NOT NULL
                      OR (
                          run.status IN ('completed', 'failed', 'cancelled')
                          AND result.id IS NOT NULL
                          AND recent_reply.task_id IS NOT NULL
                          AND result.status IN ('completed', 'interrupted')
                          AND result.fact_extraction_status <> 'waiting_consumption'
                          AND (
                              result.consumed_offset = -1
                              OR result.consumed_offset >= char_length(result.content)
                          )
                      )
                  )
            )
            SELECT *
            FROM scoped
            ORDER BY
                candidate_priority,
                CASE
                    WHEN candidate_priority = 0 THEN run_updated_at
                END DESC NULLS LAST,
                recent_reply_sequence DESC NULLS LAST,
                run_updated_at DESC,
                run_id
            LIMIT $4
            """,
            fish_id,
            session_id,
            recent_terminal_limit,
            limit,
            connector_types,
        )
        return tuple(_candidate_from_record(row) for row in rows)

    async def has_recent_agent_reply(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        before_sequence: int | None,
        limit: int = 5,
    ) -> bool:
        """Check whether recent assistant replies belong to an Agent Task."""

        _bounded_integer(
            "limit",
            limit,
            minimum=1,
            maximum=_MAX_RECENT_AGENT_REPLY_LOOKBACK,
        )
        if before_sequence is not None:
            _bounded_integer(
                "before_sequence",
                before_sequence,
                minimum=1,
                maximum=2**63 - 1,
            )
        value = await self._pool.fetchval(
            """
            WITH recent_assistant AS MATERIALIZED (
                SELECT id
                FROM messages
                WHERE fish_id = $1
                  AND session_id = $2
                  AND role = 'assistant'
                  AND status IN ('completed', 'interrupted')
                  AND ($3::bigint IS NULL OR sequence < $3)
                ORDER BY sequence DESC
                LIMIT $4
            )
            SELECT EXISTS (
                SELECT 1
                FROM recent_assistant AS recent
                WHERE EXISTS (
                    SELECT 1
                    FROM agent_runs AS run
                    WHERE run.fish_id = $1
                      AND run.session_id = $2
                      AND run.confirmation_message_id = recent.id
                )
                   OR EXISTS (
                    SELECT 1
                    FROM agent_runs AS run
                    WHERE run.fish_id = $1
                      AND run.session_id = $2
                      AND run.result_message_id = recent.id
                )
                   OR EXISTS (
                    SELECT 1
                    FROM agent_interactions AS interaction
                    WHERE interaction.fish_id = $1
                      AND interaction.session_id = $2
                      AND interaction.request_message_id = recent.id
                )
            )
            """,
            fish_id,
            session_id,
            before_sequence,
            limit,
        )
        return value is True

    async def peek_queued_run(self) -> AgentRunWorkItem | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT {_WORK_COLUMNS}
            FROM agent_runs AS run
            JOIN agent_tasks AS task ON task.id = run.task_id
            JOIN agent_targets AS target ON target.id = run.target_id
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            WHERE run.status = 'queued'
            ORDER BY run.created_at, run.id
            LIMIT 1
            """
        )
        return None if row is None else _work_item_from_record(row)

    async def peek_unmaterialized_interaction(
        self,
    ) -> AgentInteractionWorkItem | None:
        """Read one stable open Interaction without claiming or mutating it."""

        row = await self._pool.fetchrow(
            f"""
            SELECT {_WORK_COLUMNS},
                   {_CANDIDATE_INTERACTION_COLUMNS}
            FROM agent_interactions AS interaction
            JOIN agent_runs AS run ON run.id = interaction.run_id
            JOIN agent_tasks AS task ON task.id = run.task_id
            JOIN agent_targets AS target ON target.id = run.target_id
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            WHERE interaction.request_message_id IS NULL
              AND run.origin = 'user'
              AND interaction.resolved_at IS NULL
              AND interaction.closed_at IS NULL
            ORDER BY interaction.created_at, interaction.id
            LIMIT 1
            """
        )
        return None if row is None else _interaction_work_item_from_record(row)

    async def peek_unmaterialized_terminal_run(self) -> AgentRunWorkItem | None:
        """Read one stable terminal result candidate without claiming it."""

        row = await self._pool.fetchrow(
            f"""
            SELECT {_WORK_COLUMNS}
            FROM agent_runs AS run
            JOIN agent_tasks AS task ON task.id = run.task_id
            JOIN agent_targets AS target ON target.id = run.target_id
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            WHERE run.status IN ('completed', 'failed', 'cancelled')
              AND run.origin = 'user'
              AND run.result_message_id IS NULL
              AND NOT EXISTS (
                  SELECT 1
                  FROM agent_interactions AS interaction
                  WHERE interaction.run_id = run.id
                    AND interaction.kind = 'input'
              )
            ORDER BY CASE
                         WHEN run.status IN ('failed', 'cancelled') THEN 0
                         ELSE 1
                     END,
                     COALESCE(run.finished_at, run.updated_at),
                     run.id
            LIMIT 1
            """
        )
        return None if row is None else _work_item_from_record(row)

    async def fail_queued_run(
        self,
        run_id: UUID,
        error_code: str,
        error_message: str,
    ) -> AgentRun | None:
        _bounded("error_code", error_code, _MAX_EXTERNAL_ID_CHARS)
        _bounded("error_message", error_message, _MAX_ERROR_CHARS, allow_empty=True)
        row = await self._pool.fetchrow(
            f"""
            UPDATE agent_runs
            SET status = 'failed',
                error_code = $2,
                error_message = $3,
                finished_at = date_trunc('second', now()),
                updated_at = date_trunc('second', now())
            WHERE id = $1 AND status = 'queued'
            RETURNING {_RUN_COLUMNS}
            """,
            run_id,
            error_code,
            error_message,
        )
        return None if row is None else _run_from_record(row)

    async def fail_stale_run(
        self,
        run_id: UUID,
        error_code: str,
        error_message: str,
    ) -> AgentRun | None:
        """Fail one still-unattended Run and close its unresolved Interactions."""

        _bounded("error_code", error_code, _MAX_EXTERNAL_ID_CHARS)
        _bounded("error_message", error_message, _MAX_ERROR_CHARS, allow_empty=True)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    f"""
                    UPDATE agent_runs
                    SET status = 'failed',
                        error_code = $2,
                        error_message = $3,
                        finished_at = date_trunc('second', now()),
                        updated_at = date_trunc('second', now())
                    WHERE id = $1
                      AND status IN ('dispatching', 'running', 'unknown')
                    RETURNING {_RUN_COLUMNS}
                    """,
                    run_id,
                    error_code,
                    error_message,
                )
                if row is None:
                    return None
                await connection.execute(
                    """
                    UPDATE agent_interactions
                    SET closed_at = date_trunc('second', now())
                    WHERE run_id = $1
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                    """,
                    run_id,
                )
                return _run_from_record(row)

    async def claim_prepared_run(
        self,
        run_id: UUID,
        history_source: AgentHistorySource | str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRunWorkItem | None:
        history = AgentHistorySource(history_source)
        provider_json = _provider_json(provider_data)
        row = await self._pool.fetchrow(
            f"""
            WITH claimed AS (
                UPDATE agent_runs
                SET status = 'dispatching',
                    history_source = $2,
                    provider_data = $3::jsonb,
                    started_at = COALESCE(started_at, date_trunc('second', now())),
                    updated_at = date_trunc('second', now())
                WHERE id = $1 AND status = 'queued'
                RETURNING *
            )
            SELECT {_WORK_COLUMNS}
            FROM claimed AS run
            JOIN agent_tasks AS task ON task.id = run.task_id
            JOIN agent_targets AS target ON target.id = run.target_id
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            """,
            run_id,
            history.value,
            provider_json,
        )
        return None if row is None else _work_item_from_record(row)

    async def record_dispatch_accepted(
        self,
        run_id: UUID,
        external_run_id: str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun:
        _bounded("external_run_id", external_run_id, _MAX_EXTERNAL_ID_CHARS)
        provider_json = _provider_json(provider_data)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    f"""
                    UPDATE agent_runs
                    SET external_run_id = $2,
                        provider_data = $3::jsonb,
                        status = 'running',
                        started_at = COALESCE(started_at, date_trunc('second', now())),
                        updated_at = date_trunc('second', now())
                    WHERE id = $1
                      AND status = 'dispatching'
                      AND external_run_id IS NULL
                    RETURNING {_RUN_COLUMNS}
                    """,
                    run_id,
                    external_run_id,
                    provider_json,
                )
                if row is not None:
                    return _run_from_record(row)
                existing = await connection.fetchrow(
                    f"SELECT {_RUN_COLUMNS} FROM agent_runs WHERE id = $1 FOR UPDATE",
                    run_id,
                )
                existing_run = _required_run(existing)
                if existing_run.external_run_id == external_run_id:
                    return existing_run
                raise AgentRunConflictError("dispatch acceptance conflicts with run")

    async def mark_unidentified_dispatches_unknown(self) -> int:
        value = await self._pool.fetchval(
            """
            WITH updated AS (
                UPDATE agent_runs
                SET status = 'unknown',
                    error_code = 'dispatch_unknown',
                    error_message = 'Dispatch acceptance could not be identified',
                    updated_at = date_trunc('second', now())
                WHERE status = 'dispatching' AND external_run_id IS NULL
                RETURNING 1
            )
            SELECT count(*) FROM updated
            """
        )
        return int(value or 0)

    async def list_reconcilable_run_work(
        self,
        limit: int,
        after: tuple[datetime, UUID] | None = None,
    ) -> Sequence[AgentRunWorkItem]:
        if limit < 1:
            raise ValueError("limit must be positive")
        after_created_at, after_id = after if after is not None else (None, None)
        rows = await self._pool.fetch(
            f"""
            SELECT {_WORK_COLUMNS}
            FROM agent_runs AS run
            JOIN agent_tasks AS task ON task.id = run.task_id
            JOIN agent_targets AS target ON target.id = run.target_id
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            WHERE run.status IN ('dispatching', 'running', 'waiting_user', 'unknown')
              AND run.external_run_id IS NOT NULL
              AND (
                  $2::timestamptz IS NULL
                  OR (run.created_at, run.id) > ($2::timestamptz, $3::uuid)
              )
            ORDER BY run.created_at, run.id
            LIMIT $1
            """,
            limit,
            after_created_at,
            after_id,
        )
        return tuple(_work_item_from_record(row) for row in rows)

    async def list_stale_run_work(
        self,
        cutoff: datetime,
        limit: int = 1_000,
    ) -> Sequence[AgentRunWorkItem]:
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = await self._pool.fetch(
            f"""
            SELECT {_WORK_COLUMNS}
            FROM agent_runs AS run
            JOIN agent_tasks AS task ON task.id = run.task_id
            JOIN agent_targets AS target ON target.id = run.target_id
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            WHERE run.status IN ('dispatching', 'running', 'unknown')
              AND run.updated_at <= $1
            ORDER BY run.updated_at, run.id
            LIMIT $2
            """,
            cutoff,
            limit,
        )
        return tuple(_work_item_from_record(row) for row in rows)

    async def list_reconcilable_approval_responses(
        self,
        limit: int = 1_000,
        after: tuple[datetime, UUID] | None = None,
    ) -> Sequence[AgentInteractionWorkItem]:
        if limit < 1:
            raise ValueError("limit must be positive")
        after_created_at, after_id = after if after is not None else (None, None)
        rows = await self._pool.fetch(
            f"""
            SELECT
                interaction.id AS interaction_id,
                interaction.run_id AS interaction_run_id,
                interaction.fish_id AS interaction_fish_id,
                interaction.session_id AS interaction_session_id,
                interaction.kind AS interaction_kind,
                interaction.request_message_id AS interaction_request_message_id,
                interaction.response_message_id AS interaction_response_message_id,
                interaction.next_run_id AS interaction_next_run_id,
                interaction.provider_data AS interaction_provider_data,
                interaction.created_at AS interaction_created_at,
                interaction.resolved_at AS interaction_resolved_at,
                interaction.closed_at AS interaction_closed_at,
                {_WORK_COLUMNS}
            FROM agent_interactions AS interaction
            JOIN agent_runs AS run ON run.id = interaction.run_id
            JOIN agent_tasks AS task ON task.id = run.task_id
            JOIN agent_targets AS target ON target.id = run.target_id
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            WHERE interaction.kind = 'approval'
              AND interaction.response_message_id IS NOT NULL
              AND interaction.resolved_at IS NULL
              AND interaction.closed_at IS NULL
              AND interaction.provider_data #>> '{{approval_response,state}}'
                  IN ('dispatching', 'unknown')
              AND (
                  $2::timestamptz IS NULL
                  OR (interaction.created_at, interaction.id) >
                     ($2::timestamptz, $3::uuid)
              )
            ORDER BY interaction.created_at, interaction.id
            LIMIT $1
            """,
            limit,
            after_created_at,
            after_id,
        )
        return tuple(_interaction_work_item_from_record(row) for row in rows)

    async def load_local_projection(
        self,
        run_id: UUID,
        limit: int = 64,
    ) -> Sequence[tuple[str, str]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        effective_limit = min(limit, 64)
        rows = await self._pool.fetch(
            """
            WITH RECURSIVE lineage AS (
                SELECT prior.*, 0 AS lineage_depth
                FROM agent_runs AS current
                JOIN agent_runs AS prior ON prior.id = current.previous_run_id
                WHERE current.id = $1

                UNION ALL

                SELECT parent.*, child.lineage_depth + 1
                FROM lineage AS child
                JOIN agent_runs AS parent ON parent.id = child.previous_run_id
            ),
            events AS (
                SELECT lineage_depth,
                       0 AS event_group,
                       0::bigint AS event_order,
                       0 AS event_side,
                       'user'::text AS role,
                       input AS content
                FROM lineage
                WHERE input <> ''

                UNION ALL

                SELECT lineage.lineage_depth,
                       1,
                       request.sequence,
                       0,
                       'assistant',
                       request.content
                FROM lineage
                JOIN agent_interactions AS interaction
                  ON interaction.run_id = lineage.id
                JOIN messages AS request
                  ON request.id = interaction.request_message_id
                WHERE request.role = 'assistant'
                  AND request.status IN ('completed', 'interrupted')
                  AND request.fact_extraction_status <> 'waiting_consumption'
                  AND request.content <> ''

                UNION ALL

                SELECT lineage.lineage_depth,
                       1,
                       response.sequence,
                       1,
                       'user',
                       response.content
                FROM lineage
                JOIN agent_interactions AS interaction
                  ON interaction.run_id = lineage.id
                JOIN messages AS response
                  ON response.id = interaction.response_message_id
                JOIN agent_runs AS current ON current.id = $1
                WHERE response.role = 'user'
                  AND response.status = 'initialized'
                  AND response.id <> current.source_message_id
                  AND response.content <> ''

                UNION ALL

                SELECT lineage_depth,
                       2,
                       0::bigint,
                       0,
                       'assistant',
                       output
                FROM lineage
                WHERE status IN ('completed', 'failed', 'cancelled')
                  AND output IS NOT NULL
                  AND output <> ''
            ),
            bounded AS (
                SELECT lineage_depth, event_group, event_order, event_side,
                       role, content
                FROM events
                ORDER BY lineage_depth,
                         event_group DESC,
                         event_order DESC,
                         event_side DESC
                LIMIT $2
            )
            SELECT role, content
            FROM bounded
            ORDER BY lineage_depth DESC,
                     event_group,
                     event_order,
                     event_side
            """,
            run_id,
            effective_limit,
        )
        return tuple(
            (cast(str, row["role"]), cast(str, row["content"])) for row in rows
        )

    async def load_user_message_content(self, message_id: UUID) -> str | None:
        value = await self._pool.fetchval(
            """
            SELECT content
            FROM messages
            WHERE id = $1
              AND role = 'user'
              AND status = 'initialized'
            """,
            message_id,
        )
        return value if isinstance(value, str) else None

    async def update_active_run(
        self,
        run_id: UUID,
        status: AgentRunStatus | str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun | None:
        next_status = AgentRunStatus(status)
        if next_status not in {AgentRunStatus.RUNNING, AgentRunStatus.WAITING_USER}:
            raise ValueError("status must be running or waiting_user")
        _validate_incoming_run_provider(provider_data)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                current_row = await connection.fetchrow(
                    f"SELECT {_RUN_COLUMNS} FROM agent_runs WHERE id = $1 FOR UPDATE",
                    run_id,
                )
                if current_row is None or current_row["status"] not in (
                    _MUTABLE_ACTIVE_STATUS_VALUES
                ):
                    return None
                provider_json = _merged_run_provider_json(
                    current_row["provider_data"], provider_data
                )
                row = await connection.fetchrow(
                    f"""
                    UPDATE agent_runs
                    SET status = $2,
                        provider_data = $3::jsonb,
                        started_at = COALESCE(
                            started_at, date_trunc('second', now())
                        ),
                        updated_at = date_trunc('second', now())
                    WHERE id = $1
                      AND status IN (
                          'dispatching', 'running', 'waiting_user', 'unknown'
                      )
                    RETURNING {_RUN_COLUMNS}
                    """,
                    run_id,
                    next_status.value,
                    provider_json,
                )
                return None if row is None else _run_from_record(row)

    async def mark_run_unknown(
        self,
        run_id: UUID,
        error_code: str,
        error_message: str,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun | None:
        _bounded("error_code", error_code, _MAX_EXTERNAL_ID_CHARS)
        _bounded("error_message", error_message, _MAX_ERROR_CHARS, allow_empty=True)
        _validate_incoming_run_provider(provider_data)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                current_row = await connection.fetchrow(
                    f"SELECT {_RUN_COLUMNS} FROM agent_runs WHERE id = $1 FOR UPDATE",
                    run_id,
                )
                if current_row is None or current_row["status"] not in (
                    _MUTABLE_ACTIVE_STATUS_VALUES
                ):
                    return None
                provider_json = _merged_run_provider_json(
                    current_row["provider_data"], provider_data
                )
                row = await connection.fetchrow(
                    f"""
                    UPDATE agent_runs
                    SET status = 'unknown',
                        error_code = $2,
                        error_message = $3,
                        provider_data = $4::jsonb,
                        updated_at = date_trunc('second', now())
                    WHERE id = $1
                      AND status IN (
                          'dispatching', 'running', 'waiting_user', 'unknown'
                      )
                    RETURNING {_RUN_COLUMNS}
                    """,
                    run_id,
                    error_code,
                    error_message,
                    provider_json,
                )
                return None if row is None else _run_from_record(row)

    async def finish_run(
        self,
        run_id: UUID,
        status: AgentRunStatus | str,
        output: str | None,
        error_code: str | None,
        error_message: str | None,
        provider_data: Mapping[str, object] | None,
    ) -> AgentRun:
        terminal = AgentRunStatus(status)
        if terminal not in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }:
            raise ValueError("status must be completed, failed, or cancelled")
        if output is not None:
            _bounded("output", output, _MAX_MATERIALIZED_CHARS, allow_empty=True)
        if error_code is not None:
            _bounded("error_code", error_code, _MAX_EXTERNAL_ID_CHARS)
        if error_message is not None:
            _bounded("error_message", error_message, _MAX_ERROR_CHARS, allow_empty=True)
        _validate_incoming_run_provider(provider_data)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                current_row = await connection.fetchrow(
                    f"SELECT {_RUN_COLUMNS} FROM agent_runs WHERE id = $1 FOR UPDATE",
                    run_id,
                )
                current = _required_run(current_row)
                if current.status.is_terminal:
                    if current.status is terminal:
                        return current
                    raise AgentRunConflictError(
                        "run already has a different terminal status"
                    )
                if current.status.value not in _MUTABLE_ACTIVE_STATUS_VALUES:
                    raise AgentRunConflictError("run is not active")
                provider_json = _merged_run_provider_json(
                    current.provider_data, provider_data
                )
                row = await connection.fetchrow(
                    f"""
                    UPDATE agent_runs
                    SET status = $2,
                        output = $3,
                        error_code = $4,
                        error_message = $5,
                        provider_data = $6::jsonb,
                        finished_at = date_trunc('second', now()),
                        updated_at = date_trunc('second', now())
                    WHERE id = $1
                      AND status IN (
                          'dispatching', 'running', 'waiting_user', 'unknown'
                      )
                    RETURNING {_RUN_COLUMNS}
                    """,
                    run_id,
                    terminal.value,
                    output,
                    error_code,
                    error_message,
                    provider_json,
                )
                await connection.execute(
                    """
                    UPDATE agent_interactions
                    SET closed_at = date_trunc('second', now())
                    WHERE run_id = $1
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                      AND NOT (
                          kind = 'approval'
                          AND response_message_id IS NOT NULL
                          AND provider_data #>> '{approval_response,state}'
                              IN ('dispatching', 'unknown')
                      )
                    """,
                    run_id,
                )
                return _required_run(row)

    async def open_approval_interaction(
        self,
        run_id: UUID,
        provider_data: Mapping[str, object] | None,
    ) -> AgentInteraction:
        provider_json = _interaction_provider_json(provider_data)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                run = await connection.fetchrow(
                    f"SELECT {_RUN_COLUMNS} FROM agent_runs WHERE id = $1 FOR UPDATE",
                    run_id,
                )
                current = _required_run(run)
                existing = await connection.fetchrow(
                    f"""
                    SELECT {_INTERACTION_COLUMNS}
                    FROM agent_interactions
                    WHERE run_id = $1
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                    FOR UPDATE
                    """,
                    run_id,
                )
                if existing is not None:
                    interaction = _interaction_from_record(existing)
                    if interaction.kind is AgentInteractionKind.APPROVAL:
                        return interaction
                    raise AgentInteractionConflictError("run already has an open input")
                if current.status.value not in _MUTABLE_ACTIVE_STATUS_VALUES:
                    raise AgentRunConflictError("run is not active")
                row = await connection.fetchrow(
                    f"""
                    INSERT INTO agent_interactions (
                        id, run_id, fish_id, session_id, kind, provider_data
                    )
                    VALUES ($1, $2, $3, $4, 'approval', $5::jsonb)
                    RETURNING {_INTERACTION_COLUMNS}
                    """,
                    uuid4(),
                    run_id,
                    current.fish_id,
                    current.session_id,
                    provider_json,
                )
                await connection.execute(
                    """
                    UPDATE agent_runs
                    SET status = 'waiting_user',
                        updated_at = date_trunc('second', now())
                    WHERE id = $1
                      AND status IN (
                          'dispatching', 'running', 'waiting_user', 'unknown'
                      )
                    """,
                    run_id,
                )
                return _required_interaction(row)

    async def complete_run_requiring_input(
        self,
        run_id: UUID,
        output: str,
        provider_data: Mapping[str, object] | None,
    ) -> tuple[AgentRun, AgentInteraction]:
        _bounded("output", output, _MAX_MATERIALIZED_CHARS)
        _validate_incoming_run_provider(provider_data)
        interaction_provider_json = _interaction_provider_json(provider_data)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                run = await connection.fetchrow(
                    f"SELECT {_RUN_COLUMNS} FROM agent_runs WHERE id = $1 FOR UPDATE",
                    run_id,
                )
                current = _required_run(run)
                existing_row = await connection.fetchrow(
                    f"""
                    SELECT {_INTERACTION_COLUMNS}
                    FROM agent_interactions
                    WHERE run_id = $1
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                    FOR UPDATE
                    """,
                    run_id,
                )
                existing = (
                    None
                    if existing_row is None
                    else _interaction_from_record(existing_row)
                )
                if current.status.is_terminal:
                    if (
                        current.status is AgentRunStatus.COMPLETED
                        and current.output == output
                    ):
                        replay_input = existing
                        if replay_input is None:
                            replay_row = await connection.fetchrow(
                                f"""
                                SELECT {_INTERACTION_COLUMNS}
                                FROM agent_interactions
                                WHERE run_id = $1
                                  AND kind = 'input'
                                ORDER BY created_at, id
                                LIMIT 1
                                FOR UPDATE
                                """,
                                run_id,
                            )
                            replay_input = (
                                None
                                if replay_row is None
                                else _interaction_from_record(replay_row)
                            )
                        if (
                            replay_input is not None
                            and replay_input.kind is AgentInteractionKind.INPUT
                        ):
                            return current, replay_input
                    raise AgentRunConflictError(
                        "run already has a different terminal input outcome"
                    )
                if current.status.value not in _MUTABLE_ACTIVE_STATUS_VALUES:
                    raise AgentRunConflictError("run is not active")
                if existing is not None:
                    if existing.kind is not AgentInteractionKind.APPROVAL:
                        raise AgentInteractionConflictError(
                            "run already has an open input"
                        )
                    existing_provider = _json_mapping(existing.provider_data) or {}
                    raw_response = existing_provider.get("approval_response")
                    response_state = (
                        raw_response.get("state")
                        if isinstance(raw_response, Mapping)
                        else None
                    )
                    if existing.response_message_id is not None and response_state in {
                        "dispatching",
                        "unknown",
                    }:
                        await connection.execute(
                            """
                            UPDATE agent_interactions
                            SET resolved_at = date_trunc('second', now()),
                                provider_data = jsonb_set(
                                    provider_data,
                                    '{approval_response,state}',
                                    '"accepted"'::jsonb,
                                    false
                                )
                            WHERE id = $1
                            """,
                            existing.id,
                        )
                    else:
                        await connection.execute(
                            """
                            UPDATE agent_interactions
                            SET closed_at = date_trunc('second', now())
                            WHERE id = $1
                            """,
                            existing.id,
                        )
                run_provider_json = _merged_run_provider_json(
                    current.provider_data, provider_data
                )
                run_row = await connection.fetchrow(
                    f"""
                    UPDATE agent_runs
                    SET status = 'completed',
                        output = $2,
                        provider_data = $3::jsonb,
                        finished_at = date_trunc('second', now()),
                        updated_at = date_trunc('second', now())
                    WHERE id = $1
                      AND status IN (
                          'dispatching', 'running', 'waiting_user', 'unknown'
                      )
                    RETURNING {_RUN_COLUMNS}
                    """,
                    run_id,
                    output,
                    run_provider_json,
                )
                interaction_row = await connection.fetchrow(
                    f"""
                    INSERT INTO agent_interactions (
                        id, run_id, fish_id, session_id, kind, provider_data
                    )
                    VALUES ($1, $2, $3, $4, 'input', $5::jsonb)
                    RETURNING {_INTERACTION_COLUMNS}
                    """,
                    uuid4(),
                    run_id,
                    current.fish_id,
                    current.session_id,
                    interaction_provider_json,
                )
                return _required_run(run_row), _required_interaction(interaction_row)

    async def insert_and_bind_approval_response(
        self,
        interaction_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        decision: Mapping[str, object] | str,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
        adopt_existing: bool = False,
    ) -> AgentReplyMutation:
        _bounded("content", content, _MAX_MATERIALIZED_CHARS)
        response = _approval_response(decision)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                locked = await self._lock_interaction_reply_subject(
                    connection,
                    interaction_id,
                    fish_id,
                    session_id,
                    AgentInteractionKind.APPROVAL,
                )
                if locked is None:
                    raise AgentInteractionConflictError("approval is not replyable")
                _task, target_enabled, _run, interaction, _request = locked
                existing = await self._load_existing_reply_message(
                    connection,
                    message_id,
                    fish_id,
                    session_id,
                    content,
                )
                reply_context = AgentReplyContext(
                    AgentMessageSubjectKind.INTERACTION,
                    interaction_id,
                )
                if existing is not None:
                    handled = await self._authoritative_reply_handled(
                        connection,
                        existing.id,
                        reply_context,
                    )
                    if handled or not adopt_existing:
                        return AgentReplyMutation(
                            existing,
                            inserted=False,
                            replayed=True,
                            handled=handled,
                            interaction=interaction if handled else None,
                        )
                    await self._require_unassociated_direct_reply_message(
                        connection, existing.id
                    )
                elif adopt_existing:
                    raise AgentReplyConflictError(
                        "committed approval reply Message does not exist"
                    )
                if interaction.response_message_id is not None:
                    raise AgentInteractionConflictError("approval is not replyable")
                if (
                    interaction.resolved_at is not None
                    or interaction.closed_at is not None
                ):
                    raise AgentInteractionConflictError("approval is not replyable")
                if response["choice"] != "deny" and not target_enabled:
                    raise AgentInteractionConflictError(
                        "Agent task target is unavailable"
                    )
                if (
                    _recovered_without_details(interaction.provider_data)
                    and response["choice"] != "deny"
                ):
                    raise AgentInteractionConflictError(
                        "recovered approval without details can only be denied"
                    )
                _require_allowed_decision(interaction.provider_data, response["choice"])
                merged_provider = dict(interaction.provider_data or {})
                merged_provider["approval_response"] = response
                merged_provider_json = _provider_json(
                    merged_provider,
                    reserve_bytes=_RECOVERY_METADATA_RESERVE_BYTES,
                )
                inserted = existing is None
                if existing is None:
                    message = await self._messages.insert_user_message_on_connection(
                        connection,
                        fish_id,
                        session_id,
                        content,
                        message_id=message_id,
                        created_at=created_at,
                    )
                    await self._require_unassociated_direct_reply_message(
                        connection, message.id
                    )
                else:
                    message = existing
                row = await connection.fetchrow(
                    f"""
                    UPDATE agent_interactions
                    SET response_message_id = $2,
                        provider_data = $3::jsonb
                    WHERE id = $1
                      AND kind = 'approval'
                      AND response_message_id IS NULL
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                    RETURNING {_INTERACTION_COLUMNS}
                    """,
                    interaction_id,
                    message.id,
                    merged_provider_json,
                )
                if row is None:
                    raise AgentInteractionConflictError("approval response lost CAS")
                return AgentReplyMutation(
                    message,
                    inserted=inserted,
                    replayed=False,
                    handled=True,
                    interaction=_interaction_from_record(row),
                )

    async def claim_pending_approval_response(
        self,
    ) -> AgentInteractionWorkItem | None:
        return await self._claim_interaction_dispatch("queued", "dispatching")

    async def mark_unidentified_approval_dispatches_unknown(self) -> int:
        value = await self._pool.fetchval(
            """
            WITH updated AS (
                UPDATE agent_interactions
                SET provider_data = jsonb_set(
                    provider_data,
                    '{approval_response,state}',
                    '"unknown"'::jsonb,
                    false
                )
                WHERE resolved_at IS NULL
                  AND closed_at IS NULL
                  AND provider_data #>> '{approval_response,state}' = 'dispatching'
                RETURNING 1
            )
            SELECT count(*) FROM updated
            """
        )
        return int(value or 0)

    async def mark_approval_dispatch_unknown(
        self,
        interaction_id: UUID,
        error_code: str,
        error_message: str,
    ) -> AgentInteraction | None:
        _bounded("error_code", error_code, _MAX_EXTERNAL_ID_CHARS)
        _bounded("error_message", error_message, _MAX_ERROR_CHARS, allow_empty=True)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                current = await connection.fetchrow(
                    f"""
                    SELECT {_INTERACTION_COLUMNS}
                    FROM agent_interactions
                    WHERE id = $1
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                      AND provider_data #>> '{{approval_response,state}}' =
                          'dispatching'
                    FOR UPDATE
                    """,
                    interaction_id,
                )
                if current is None:
                    return None
                provider = dict(_json_mapping(current["provider_data"]) or {})
                approval = dict(
                    cast(
                        Mapping[str, object],
                        provider.get("approval_response", {}),
                    )
                )
                approval.update(
                    {
                        "state": "unknown",
                        "error_code": error_code,
                        "error_message": error_message,
                    }
                )
                provider["approval_response"] = approval
                provider_json = _provider_json(provider)
                row = await connection.fetchrow(
                    f"""
                    UPDATE agent_interactions
                    SET provider_data = $2::jsonb
                    WHERE id = $1
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                    RETURNING {_INTERACTION_COLUMNS}
                    """,
                    interaction_id,
                    provider_json,
                )
                return None if row is None else _interaction_from_record(row)

    async def resolve_approval(
        self,
        interaction_id: UUID,
    ) -> AgentInteraction | None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                run_id = await connection.fetchval(
                    """
                    SELECT run_id
                    FROM agent_interactions
                    WHERE id = $1
                    """,
                    interaction_id,
                )
                if not isinstance(run_id, UUID):
                    return None
                run = await connection.fetchrow(
                    f"""
                    SELECT {_RUN_COLUMNS}
                    FROM agent_runs
                    WHERE id = $1
                    FOR UPDATE
                    """,
                    run_id,
                )
                if run is None:
                    return None
                row = await connection.fetchrow(
                    f"""
                    UPDATE agent_interactions
                    SET resolved_at = date_trunc('second', now()),
                        provider_data = jsonb_set(
                            provider_data,
                            '{{approval_response,state}}',
                            '"accepted"'::jsonb,
                            false
                        )
                    WHERE id = $1
                      AND kind = 'approval'
                      AND response_message_id IS NOT NULL
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                      AND provider_data #>> '{{approval_response,state}}'
                          IN ('dispatching', 'unknown')
                    RETURNING {_INTERACTION_COLUMNS}
                    """,
                    interaction_id,
                )
                if row is None:
                    return None
                return _interaction_from_record(row)

    async def insert_and_resolve_input(
        self,
        interaction_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
        confirmation_message_id: UUID | None = None,
        adopt_existing: bool = False,
    ) -> AgentReplyMutation:
        _bounded("content", content, _MAX_MATERIALIZED_CHARS)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                locked = await self._lock_interaction_reply_subject(
                    connection,
                    interaction_id,
                    fish_id,
                    session_id,
                    AgentInteractionKind.INPUT,
                )
                if locked is None:
                    raise AgentInteractionConflictError("input is not replyable")
                task, target_enabled, run, interaction, _request = locked
                existing = await self._load_existing_reply_message(
                    connection,
                    message_id,
                    fish_id,
                    session_id,
                    content,
                )
                reply_context = AgentReplyContext(
                    AgentMessageSubjectKind.INTERACTION,
                    interaction_id,
                )
                if existing is not None:
                    handled = await self._authoritative_reply_handled(
                        connection,
                        existing.id,
                        reply_context,
                    )
                    if not handled and not adopt_existing:
                        return AgentReplyMutation(
                            existing,
                            inserted=False,
                            replayed=True,
                            handled=False,
                        )
                    if handled:
                        if (
                            interaction.response_message_id != existing.id
                            or interaction.next_run_id is None
                            or interaction.resolved_at is None
                        ):
                            raise AgentInteractionConflictError(
                                "input direct reply proof is incomplete"
                            )
                        next_run_row = await connection.fetchrow(
                            f"SELECT {_RUN_COLUMNS} FROM agent_runs WHERE id = $1",
                            interaction.next_run_id,
                        )
                        return AgentReplyMutation(
                            existing,
                            inserted=False,
                            replayed=True,
                            handled=True,
                            interaction=interaction,
                            run=_required_run(next_run_row),
                        )
                    await self._require_unassociated_direct_reply_message(
                        connection, existing.id
                    )
                elif adopt_existing:
                    raise AgentReplyConflictError(
                        "committed input reply Message does not exist"
                    )
                if interaction.response_message_id is not None:
                    raise AgentInteractionConflictError("input is not replyable")
                if (
                    interaction.resolved_at is not None
                    or interaction.closed_at is not None
                ):
                    raise AgentInteractionConflictError("input is not replyable")
                if not target_enabled:
                    raise AgentRunConflictError("Agent task target is unavailable")
                task_id = cast(UUID, task["id"])
                if await self._has_active_run(connection, task_id):
                    raise AgentRunConflictError("Agent task already has an active run")
                inserted = existing is None
                if existing is None:
                    message = await self._messages.insert_user_message_on_connection(
                        connection,
                        fish_id,
                        session_id,
                        content,
                        message_id=message_id,
                        created_at=created_at,
                    )
                    await self._require_unassociated_direct_reply_message(
                        connection, message.id
                    )
                else:
                    message = existing
                if confirmation_message_id is not None:
                    await self._lock_run_messages(
                        connection,
                        fish_id,
                        message.id,
                        confirmation_message_id,
                        expected_session_id=cast(UUID, task["session_id"]),
                    )
                occupied = await connection.fetchrow(
                    """
                    UPDATE agent_interactions
                    SET response_message_id = $2
                    WHERE id = $1
                      AND kind = 'input'
                      AND response_message_id IS NULL
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                    RETURNING id
                    """,
                    interaction_id,
                    message.id,
                )
                if occupied is None:
                    raise AgentInteractionConflictError("input response lost CAS")
                next_run = await self._insert_run(
                    connection,
                    task_id=task_id,
                    target_id=cast(UUID, task["target_id"]),
                    fish_id=cast(UUID, task["fish_id"]),
                    session_id=cast(UUID, task["session_id"]),
                    previous_run_id=run.id,
                    source_message_id=message.id,
                    confirmation_message_id=confirmation_message_id,
                    input=content,
                )
                interaction_row = await connection.fetchrow(
                    f"""
                    UPDATE agent_interactions
                    SET next_run_id = $2,
                        resolved_at = date_trunc('second', now())
                    WHERE id = $1
                      AND response_message_id = $3
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                    RETURNING {_INTERACTION_COLUMNS}
                    """,
                    interaction_id,
                    next_run.id,
                    message.id,
                )
                if interaction_row is None:
                    raise AgentInteractionConflictError("input resolution lost CAS")
                return AgentReplyMutation(
                    message,
                    inserted=inserted,
                    replayed=False,
                    handled=True,
                    interaction=_interaction_from_record(interaction_row),
                    run=next_run,
                )

    async def insert_and_create_result_followup(
        self,
        run_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        action: str,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> AgentReplyMutation:
        _bounded("content", content, _MAX_MATERIALIZED_CHARS)
        if action not in {"continue", "retry", "changes_requested"}:
            raise ValueError("action is not a result follow-up")
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                locked = await self._lock_result_reply_subject(
                    connection,
                    run_id,
                    fish_id,
                    session_id,
                )
                if locked is None:
                    raise AgentRunConflictError("run result is not followable")
                task, target_enabled, run, _result, child = locked
                existing = await self._load_existing_reply_message(
                    connection,
                    message_id,
                    fish_id,
                    session_id,
                    content,
                )
                reply_context = AgentReplyContext(
                    AgentMessageSubjectKind.RUN_RESULT,
                    run_id,
                )
                if existing is not None:
                    handled = await self._authoritative_reply_handled(
                        connection,
                        existing.id,
                        reply_context,
                    )
                    if not handled:
                        return AgentReplyMutation(
                            existing,
                            inserted=False,
                            replayed=True,
                            handled=False,
                        )
                    if (
                        child is None
                        or child.source_message_id != existing.id
                        or child.confirmation_message_id is not None
                    ):
                        raise AgentRunConflictError(
                            "result direct reply proof is incomplete"
                        )
                    return AgentReplyMutation(
                        existing,
                        inserted=False,
                        replayed=True,
                        handled=True,
                        run=child,
                    )
                if child is not None:
                    raise AgentRunConflictError("run result is not followable")
                if not target_enabled:
                    raise AgentRunConflictError("Agent task target is unavailable")
                task_id = cast(UUID, task["id"])
                if await self._has_active_run(connection, task_id):
                    raise AgentRunConflictError("Agent task already has an active run")
                message = await self._messages.insert_user_message_on_connection(
                    connection,
                    fish_id,
                    session_id,
                    content,
                    message_id=message_id,
                    created_at=created_at,
                )
                await self._require_unassociated_direct_reply_message(
                    connection, message.id
                )
                next_run = await self._insert_run(
                    connection,
                    task_id=task_id,
                    target_id=cast(UUID, task["target_id"]),
                    fish_id=cast(UUID, task["fish_id"]),
                    session_id=cast(UUID, task["session_id"]),
                    previous_run_id=run_id,
                    source_message_id=message.id,
                    confirmation_message_id=None,
                    input=content,
                )
                return AgentReplyMutation(
                    message,
                    inserted=True,
                    replayed=False,
                    handled=True,
                    run=next_run,
                )

    async def insert_validated_ordinary_reply(
        self,
        context: AgentReplyContext,
        fish_id: UUID,
        session_id: UUID,
        content: str,
        *,
        message_id: UUID | None = None,
        created_at: datetime | None = None,
    ) -> AgentReplyMutation:
        """Revalidate one exact subject and insert an ordinary User Message."""

        _bounded("content", content, _MAX_MATERIALIZED_CHARS)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                if context.kind is AgentMessageSubjectKind.INTERACTION:
                    locked = await self._lock_interaction_reply_subject(
                        connection,
                        context.id,
                        fish_id,
                        session_id,
                        None,
                    )
                    if locked is None:
                        raise AgentInteractionConflictError(
                            "interaction is not replyable"
                        )
                    _task, _enabled, _run, interaction, _request = locked
                    existing = await self._load_existing_reply_message(
                        connection,
                        message_id,
                        fish_id,
                        session_id,
                        content,
                    )
                    if existing is not None:
                        handled = await self._authoritative_reply_handled(
                            connection,
                            existing.id,
                            context,
                        )
                        return AgentReplyMutation(
                            existing,
                            inserted=False,
                            replayed=True,
                            handled=handled,
                        )
                    if (
                        interaction.response_message_id is not None
                        or interaction.resolved_at is not None
                        or interaction.closed_at is not None
                    ):
                        raise AgentInteractionConflictError(
                            "interaction is not replyable"
                        )
                elif context.kind is AgentMessageSubjectKind.RUN_RESULT:
                    locked_result = await self._lock_result_reply_subject(
                        connection,
                        context.id,
                        fish_id,
                        session_id,
                    )
                    if locked_result is None:
                        raise AgentRunConflictError("run result is not replyable")
                    existing = await self._load_existing_reply_message(
                        connection,
                        message_id,
                        fish_id,
                        session_id,
                        content,
                    )
                    if existing is not None:
                        handled = await self._authoritative_reply_handled(
                            connection,
                            existing.id,
                            context,
                        )
                        return AgentReplyMutation(
                            existing,
                            inserted=False,
                            replayed=True,
                            handled=handled,
                        )
                    _task, _enabled, _run, _result, child = locked_result
                    if child is not None:
                        raise AgentRunConflictError("run result is not replyable")
                else:
                    raise AgentReplyConflictError("reply subject kind is unsupported")
                message = await self._messages.insert_user_message_on_connection(
                    connection,
                    fish_id,
                    session_id,
                    content,
                    message_id=message_id,
                    created_at=created_at,
                )
                return AgentReplyMutation(
                    message,
                    inserted=True,
                    replayed=False,
                    handled=False,
                )

    async def abandon_unknown_and_create_retry(
        self,
        run_id: UUID,
        source_message_id: UUID,
        confirmation_message_id: UUID | None,
        input: str,
    ) -> AgentRun | None:
        _bounded("input", input, _MAX_MATERIALIZED_CHARS)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                task_id = await connection.fetchval(
                    "SELECT task_id FROM agent_runs WHERE id = $1",
                    run_id,
                )
                if not isinstance(task_id, UUID):
                    return None
                task = await self._lock_task_with_enabled_target(connection, task_id)
                if task is None:
                    return None
                row = await connection.fetchrow(
                    """
                    SELECT run.task_id, run.target_id
                    FROM agent_runs AS run
                    WHERE run.id = $1
                      AND run.task_id = $2
                      AND run.status = 'unknown'
                    FOR UPDATE OF run
                    """,
                    run_id,
                    task_id,
                )
                if row is None:
                    return None
                if row["target_id"] != task["target_id"]:
                    return None
                await self._lock_run_messages(
                    connection,
                    cast(UUID, task["fish_id"]),
                    source_message_id,
                    confirmation_message_id,
                    expected_session_id=cast(UUID, task["session_id"]),
                )
                abandoned = await connection.fetchrow(
                    """
                    UPDATE agent_runs
                    SET status = 'abandoned',
                        finished_at = date_trunc('second', now()),
                        updated_at = date_trunc('second', now())
                    WHERE id = $1 AND status = 'unknown'
                    RETURNING id
                    """,
                    run_id,
                )
                if abandoned is None:
                    return None
                return await self._insert_run(
                    connection,
                    task_id=cast(UUID, row["task_id"]),
                    target_id=cast(UUID, row["target_id"]),
                    fish_id=cast(UUID, task["fish_id"]),
                    session_id=cast(UUID, task["session_id"]),
                    previous_run_id=run_id,
                    source_message_id=source_message_id,
                    confirmation_message_id=confirmation_message_id,
                    input=input,
                )

    async def queue_running_steer(
        self,
        run_id: UUID,
        source_message_id: UUID,
    ) -> AgentRun | None:
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                task_id = await connection.fetchval(
                    "SELECT task_id FROM agent_runs WHERE id = $1",
                    run_id,
                )
                if not isinstance(task_id, UUID):
                    return None
                task = await self._lock_task_with_enabled_target(connection, task_id)
                if task is None:
                    return None
                current = await connection.fetchrow(
                    f"""
                    SELECT {_QUALIFIED_RUN_COLUMNS}
                    FROM agent_runs AS run
                    JOIN messages AS source ON source.id = $2
                    JOIN messages AS run_source ON run_source.id = run.source_message_id
                    WHERE run.id = $1
                      AND run.task_id = $3
                      AND run.status = 'running'
                      AND run.target_id = $4
                      AND source.fish_id = $5
                      AND source.role = 'user'
                      AND source.session_id = $6
                      AND run_source.session_id = $6
                    FOR UPDATE OF run
                    """,
                    run_id,
                    source_message_id,
                    task_id,
                    task["target_id"],
                    task["fish_id"],
                    task["session_id"],
                )
                if current is None:
                    return None
                merged_provider = dict(_json_mapping(current["provider_data"]) or {})
                existing_steer = merged_provider.get("steer")
                existing = (
                    cast(Mapping[str, object], existing_steer)
                    if isinstance(existing_steer, Mapping)
                    else {}
                )
                existing_state = existing.get("state")
                same_source = existing.get("source_message_id") == str(
                    source_message_id
                )
                current_run = _run_from_record(current)
                if existing_state == "unknown":
                    return current_run if same_source else None
                if existing_state in {"queued", "dispatching"}:
                    return current_run if same_source else None
                if existing_state == "accepted" and same_source:
                    return current_run
                if existing_state not in {None, "accepted"}:
                    return None
                merged_provider["steer"] = {
                    "source_message_id": str(source_message_id),
                    "state": "queued",
                }
                provider_json = _provider_json(
                    merged_provider,
                    reserve_bytes=_RECOVERY_METADATA_RESERVE_BYTES,
                )
                row = await connection.fetchrow(
                    f"""
                    UPDATE agent_runs
                    SET provider_data = $2::jsonb,
                        updated_at = date_trunc('second', now())
                    WHERE id = $1
                      AND status = 'running'
                    RETURNING {_RUN_COLUMNS}
                    """,
                    run_id,
                    provider_json,
                )
                return None if row is None else _run_from_record(row)

    async def claim_pending_steer(self) -> AgentRunWorkItem | None:
        row = await self._pool.fetchrow(
            f"""
            WITH candidate AS (
                SELECT id
                FROM agent_runs
                WHERE status = 'running'
                  AND provider_data #>> '{{steer,state}}' = 'queued'
                ORDER BY updated_at, id
                LIMIT 1
            ),
            claimed AS (
                UPDATE agent_runs AS run
                SET provider_data = jsonb_set(
                        run.provider_data,
                        '{{steer,state}}',
                        '"dispatching"'::jsonb,
                        false
                    ),
                    updated_at = date_trunc('second', now())
                FROM candidate
                WHERE run.id = candidate.id
                  AND run.status = 'running'
                  AND run.provider_data #>> '{{steer,state}}' = 'queued'
                RETURNING run.*
            )
            SELECT {_WORK_COLUMNS}
            FROM claimed AS run
            JOIN agent_tasks AS task ON task.id = run.task_id
            JOIN agent_targets AS target ON target.id = run.target_id
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            """
        )
        return None if row is None else _work_item_from_record(row)

    async def mark_unidentified_steer_dispatches_unknown(self) -> int:
        value = await self._pool.fetchval(
            """
            WITH updated AS (
                UPDATE agent_runs
                SET provider_data = jsonb_set(
                        provider_data,
                        '{steer,state}',
                        '"unknown"'::jsonb,
                        false
                    ),
                    updated_at = date_trunc('second', now())
                WHERE status = 'running'
                  AND provider_data #>> '{steer,state}' = 'dispatching'
                RETURNING 1
            )
            SELECT count(*) FROM updated
            """
        )
        return int(value or 0)

    async def record_steer_accepted(self, run_id: UUID) -> AgentRun | None:
        return await self._update_steer_state(run_id, "dispatching", "accepted")

    async def mark_steer_dispatch_unknown(
        self,
        run_id: UUID,
        error_code: str,
        error_message: str,
    ) -> AgentRun | None:
        _bounded("error_code", error_code, _MAX_EXTERNAL_ID_CHARS)
        _bounded("error_message", error_message, _MAX_ERROR_CHARS, allow_empty=True)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                current = await connection.fetchrow(
                    f"""
                    SELECT {_RUN_COLUMNS}
                    FROM agent_runs
                    WHERE id = $1
                      AND status = 'running'
                      AND provider_data #>> '{{steer,state}}' = 'dispatching'
                    FOR UPDATE
                    """,
                    run_id,
                )
                if current is None:
                    return None
                provider = dict(_json_mapping(current["provider_data"]) or {})
                steer = dict(cast(Mapping[str, object], provider.get("steer", {})))
                steer.update(
                    {
                        "state": "unknown",
                        "error_code": error_code,
                        "error_message": error_message,
                    }
                )
                provider["steer"] = steer
                provider_json = _provider_json(provider)
                row = await connection.fetchrow(
                    f"""
                    UPDATE agent_runs
                    SET provider_data = $2::jsonb,
                        updated_at = date_trunc('second', now())
                    WHERE id = $1 AND status = 'running'
                    RETURNING {_RUN_COLUMNS}
                    """,
                    run_id,
                    provider_json,
                )
                return None if row is None else _run_from_record(row)

    async def close_open_interaction(
        self,
        run_id: UUID,
    ) -> AgentInteraction | None:
        row = await self._pool.fetchrow(
            f"""
            UPDATE agent_interactions
            SET closed_at = date_trunc('second', now())
            WHERE run_id = $1
              AND resolved_at IS NULL
              AND closed_at IS NULL
            RETURNING {_INTERACTION_COLUMNS}
            """,
            run_id,
        )
        return None if row is None else _interaction_from_record(row)

    async def materialize_interaction_request(
        self,
        interaction_id: UUID,
        content: str,
        *,
        emotion: str = "neutral",
    ) -> Message | None:
        _bounded("content", content, _MAX_MATERIALIZED_CHARS)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT interaction.request_message_id,
                           task.fish_id,
                           task.session_id,
                           source.sequence
                    FROM agent_interactions AS interaction
                    JOIN agent_runs AS run ON run.id = interaction.run_id
                    JOIN agent_tasks AS task ON task.id = run.task_id
                    JOIN messages AS source ON source.id = run.source_message_id
                    WHERE interaction.id = $1
                      AND interaction.resolved_at IS NULL
                      AND interaction.closed_at IS NULL
                    FOR UPDATE OF interaction
                    """,
                    interaction_id,
                )
                if row is None or row["request_message_id"] is not None:
                    return None
                message = await self._messages.insert_background_message_on_connection(
                    connection,
                    cast(UUID, row["fish_id"]),
                    cast(UUID, row["session_id"]),
                    cast(int, row["sequence"]),
                    content,
                )
                await self._background_messages.insert_agent_delivery_on_connection(
                    connection,
                    message.id,
                    emotion=emotion,
                )
                await self._lock_unbound_assistant(connection, message.id)
                updated = await connection.fetchrow(
                    """
                    UPDATE agent_interactions
                    SET request_message_id = $2
                    WHERE id = $1
                      AND request_message_id IS NULL
                      AND resolved_at IS NULL
                      AND closed_at IS NULL
                    RETURNING id
                    """,
                    interaction_id,
                    message.id,
                )
                if updated is None:
                    raise AgentMessageSubjectConflictError(
                        "interaction request binding lost CAS"
                    )
                return message

    async def materialize_run_result(
        self,
        run_id: UUID,
        content: str,
        *,
        emotion: str = "neutral",
    ) -> Message | None:
        _bounded("content", content, _MAX_MATERIALIZED_CHARS)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    SELECT run.result_message_id,
                           run.status,
                           task.fish_id,
                           task.session_id,
                           source.sequence
                    FROM agent_runs AS run
                    JOIN agent_tasks AS task ON task.id = run.task_id
                    JOIN messages AS source ON source.id = run.source_message_id
                    WHERE run.id = $1
                      AND run.status IN ('completed', 'failed', 'cancelled')
                      AND NOT EXISTS (
                          SELECT 1
                          FROM agent_interactions AS interaction
                          WHERE interaction.run_id = run.id
                            AND interaction.kind = 'input'
                      )
                    FOR UPDATE OF run
                    """,
                    run_id,
                )
                if row is None or row["result_message_id"] is not None:
                    return None
                message = await self._messages.insert_background_message_on_connection(
                    connection,
                    cast(UUID, row["fish_id"]),
                    cast(UUID, row["session_id"]),
                    cast(int, row["sequence"]),
                    content,
                )
                await self._background_messages.insert_agent_delivery_on_connection(
                    connection,
                    message.id,
                    emotion=emotion,
                )
                await self._lock_unbound_assistant(connection, message.id)
                updated = await connection.fetchrow(
                    """
                    UPDATE agent_runs
                    SET result_message_id = $2,
                        updated_at = date_trunc('second', now())
                    WHERE id = $1
                      AND result_message_id IS NULL
                      AND status IN ('completed', 'failed', 'cancelled')
                    RETURNING id
                    """,
                    run_id,
                    message.id,
                )
                if updated is None:
                    raise AgentMessageSubjectConflictError(
                        "run result binding lost CAS"
                    )
                return message

    async def finalize_agent_message_consumption(
        self,
        message_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        consumed_offset: int,
        interrupted: bool,
    ) -> Message | None:
        subject = await self._pool.fetchrow(
            """
            SELECT message.fish_id,
                   message.session_id,
                   message.role,
                   ((result_run.id IS NOT NULL)::int +
                    (interaction.id IS NOT NULL)::int) AS subject_count
            FROM messages AS message
            LEFT JOIN agent_runs AS result_run
              ON result_run.result_message_id = message.id
            LEFT JOIN agent_interactions AS interaction
              ON interaction.request_message_id = message.id
            WHERE message.id = $1
            """,
            message_id,
        )
        if (
            subject is None
            or subject["fish_id"] != fish_id
            or subject["session_id"] != session_id
        ):
            return None
        if subject["role"] != "assistant" or subject["subject_count"] != 1:
            raise AgentMessageSubjectConflictError(
                "message is not bound to exactly one Agent subject"
            )
        return await self._messages.finalize_agent_message_consumption(
            message_id,
            fish_id,
            consumed_offset,
            interrupted,
        )

    async def list_agent_messages_for_delivery(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int,
        include_incomplete: bool,
        limit: int,
        *,
        incomplete_after_sequence: int = 0,
    ) -> Sequence[AgentMessageEnvelope]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if incomplete_after_sequence < 0:
            raise ValueError("incomplete_after_sequence must be non-negative")
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = await self._pool.fetch(
            """
            SELECT message.id AS message_id,
                   message.sequence,
                   message.session_id,
                   message.content,
                   message.consumed_offset,
                   result_run.id AS result_run_id,
                   interaction.id AS interaction_id,
                   interaction.kind AS interaction_kind
            FROM messages AS message
            LEFT JOIN agent_runs AS result_run
              ON result_run.result_message_id = message.id
            LEFT JOIN agent_interactions AS interaction
              ON interaction.request_message_id = message.id
            WHERE message.fish_id = $1
              AND message.session_id = $2
              AND message.role = 'assistant'
              AND ((result_run.id IS NOT NULL)::int +
                   (interaction.id IS NOT NULL)::int) = 1
              AND char_length(message.content) > 0
              AND (
                  message.sequence > $3
                  OR (
                      $4
                      AND message.sequence > $5
                      AND message.sequence <= $3
                      AND (
                          message.fact_extraction_status = 'waiting_consumption'
                          OR (
                              message.consumed_offset <> -1
                              AND message.consumed_offset
                                  < char_length(message.content)
                          )
                      )
                  )
              )
            ORDER BY CASE
                         WHEN $4
                          AND message.sequence > $5
                          AND message.sequence <= $3
                          AND (
                              message.fact_extraction_status = 'waiting_consumption'
                              OR (
                                  message.consumed_offset <> -1
                                  AND message.consumed_offset
                                      < char_length(message.content)
                              )
                          )
                         THEN 0
                         ELSE 1
                     END,
                     message.sequence
            LIMIT $6
            """,
            fish_id,
            session_id,
            after_sequence,
            include_incomplete,
            incomplete_after_sequence,
            limit,
        )
        return tuple(_agent_envelope_from_record(row) for row in rows)

    async def list_incomplete_agent_messages_for_delivery(
        self,
        fish_id: UUID,
        session_id: UUID,
        after_sequence: int,
        through_sequence: int,
        limit: int,
    ) -> Sequence[AgentMessageEnvelope]:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if through_sequence < 0:
            raise ValueError("through_sequence must be non-negative")
        if after_sequence > through_sequence:
            raise ValueError("after_sequence must not exceed through_sequence")
        if limit < 1:
            raise ValueError("limit must be positive")
        rows = await self._pool.fetch(
            """
            SELECT message.id AS message_id,
                   message.sequence,
                   message.session_id,
                   message.content,
                   message.consumed_offset,
                   result_run.id AS result_run_id,
                   interaction.id AS interaction_id,
                   interaction.kind AS interaction_kind
            FROM messages AS message
            LEFT JOIN agent_runs AS result_run
              ON result_run.result_message_id = message.id
            LEFT JOIN agent_interactions AS interaction
              ON interaction.request_message_id = message.id
            WHERE message.fish_id = $1
              AND message.session_id = $2
              AND message.role = 'assistant'
              AND ((result_run.id IS NOT NULL)::int +
                   (interaction.id IS NOT NULL)::int) = 1
              AND char_length(message.content) > 0
              AND message.sequence > $3
              AND message.sequence <= $4
              AND (
                  message.fact_extraction_status = 'waiting_consumption'
                  OR (
                      message.consumed_offset <> -1
                      AND message.consumed_offset
                          < char_length(message.content)
                  )
              )
            ORDER BY message.sequence
            LIMIT $5
            """,
            fish_id,
            session_id,
            after_sequence,
            through_sequence,
            limit,
        )
        return tuple(_agent_envelope_from_record(row) for row in rows)

    async def _claim_interaction_dispatch(
        self,
        from_state: str,
        to_state: str,
    ) -> AgentInteractionWorkItem | None:
        row = await self._pool.fetchrow(
            f"""
            WITH candidate AS (
                SELECT id
                FROM agent_interactions
                WHERE kind = 'approval'
                  AND response_message_id IS NOT NULL
                  AND resolved_at IS NULL
                  AND closed_at IS NULL
                  AND provider_data #>> '{{approval_response,state}}' = $1
                ORDER BY created_at, id
                LIMIT 1
            ),
            claimed_interaction AS (
                UPDATE agent_interactions AS interaction
                SET provider_data = jsonb_set(
                        interaction.provider_data,
                        '{{approval_response,state}}',
                        to_jsonb($2::text),
                        false
                    )
                FROM candidate
                WHERE interaction.id = candidate.id
                  AND interaction.provider_data #>> '{{approval_response,state}}' = $1
                RETURNING interaction.*
            )
            SELECT
                interaction.id AS interaction_id,
                interaction.run_id AS interaction_run_id,
                interaction.fish_id AS interaction_fish_id,
                interaction.session_id AS interaction_session_id,
                interaction.kind AS interaction_kind,
                interaction.request_message_id AS interaction_request_message_id,
                interaction.response_message_id AS interaction_response_message_id,
                interaction.next_run_id AS interaction_next_run_id,
                interaction.provider_data AS interaction_provider_data,
                interaction.created_at AS interaction_created_at,
                interaction.resolved_at AS interaction_resolved_at,
                interaction.closed_at AS interaction_closed_at,
                {_WORK_COLUMNS}
            FROM claimed_interaction AS interaction
            JOIN agent_runs AS run ON run.id = interaction.run_id
            JOIN agent_tasks AS task ON task.id = run.task_id
            JOIN agent_targets AS target ON target.id = run.target_id
            JOIN agent_endpoints AS endpoint ON endpoint.id = target.endpoint_id
            """,
            from_state,
            to_state,
        )
        return None if row is None else _interaction_work_item_from_record(row)

    async def _update_steer_state(
        self,
        run_id: UUID,
        from_state: str,
        to_state: str,
    ) -> AgentRun | None:
        row = await self._pool.fetchrow(
            f"""
            UPDATE agent_runs
            SET provider_data = jsonb_set(
                    provider_data,
                    '{{steer,state}}',
                    to_jsonb($3::text),
                    false
                ),
                updated_at = date_trunc('second', now())
            WHERE id = $1
              AND status = 'running'
              AND provider_data #>> '{{steer,state}}' = $2
            RETURNING {_RUN_COLUMNS}
            """,
            run_id,
            from_state,
            to_state,
        )
        return None if row is None else _run_from_record(row)

    async def _lock_task_with_enabled_target(
        self,
        connection: asyncpg.Connection,
        task_id: UUID,
    ) -> Mapping[str, object] | None:
        """Lock Task then Target so prior-task work cannot race disable."""

        locked = await self._lock_task_and_target(connection, task_id)
        if locked is None or not locked[1]:
            return None
        return locked[0]

    async def _lock_task_and_target(
        self,
        connection: asyncpg.Connection,
        task_id: UUID,
    ) -> tuple[Mapping[str, object], bool] | None:
        """Lock Task then Target and report the current enabled state."""

        task = await connection.fetchrow(
            """
            SELECT id, fish_id, session_id, target_id
            FROM agent_tasks
            WHERE id = $1
            FOR UPDATE
            """,
            task_id,
        )
        if task is None:
            return None
        target = await connection.fetchrow(
            """
            SELECT id, enabled
            FROM agent_targets
            WHERE id = $1
            FOR SHARE
            """,
            task["target_id"],
        )
        if target is None:
            return None
        return cast(Mapping[str, object], task), cast(bool, target["enabled"])

    async def _lock_interaction_reply_subject(
        self,
        connection: asyncpg.Connection,
        interaction_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        expected_kind: AgentInteractionKind | None,
    ) -> (
        tuple[
            Mapping[str, object],
            bool,
            AgentRun,
            AgentInteraction,
            Mapping[str, object],
        ]
        | None
    ):
        identity = await connection.fetchrow(
            """
            SELECT interaction.run_id, run.task_id
            FROM agent_interactions AS interaction
            JOIN agent_runs AS run ON run.id = interaction.run_id
            WHERE interaction.id = $1
            """,
            interaction_id,
        )
        if identity is None:
            return None
        task_id = cast(UUID, identity["task_id"])
        locked_task = await self._lock_task_and_target(connection, task_id)
        if locked_task is None:
            return None
        task, target_enabled = locked_task
        if task["fish_id"] != fish_id or task["session_id"] != session_id:
            return None
        run_row = await connection.fetchrow(
            f"""
            SELECT {_RUN_COLUMNS}
            FROM agent_runs
            WHERE id = $1
              AND task_id = $2
              AND target_id = $3
            FOR UPDATE
            """,
            identity["run_id"],
            task_id,
            task["target_id"],
        )
        if run_row is None:
            return None
        interaction_row = await connection.fetchrow(
            f"""
            SELECT {_INTERACTION_COLUMNS}
            FROM agent_interactions
            WHERE id = $1 AND run_id = $2
            FOR UPDATE
            """,
            interaction_id,
            identity["run_id"],
        )
        if interaction_row is None:
            return None
        interaction = _interaction_from_record(interaction_row)
        if expected_kind is not None and interaction.kind is not expected_kind:
            return None
        if interaction.request_message_id is None:
            return None
        request = await connection.fetchrow(
            """
            SELECT id, fish_id, session_id, role, content, status,
                   consumed_offset, fact_extraction_status
            FROM messages
            WHERE id = $1
            FOR UPDATE
            """,
            interaction.request_message_id,
        )
        if request is None:
            return None
        _require_locked_reply_source(request, fish_id, session_id)
        return (
            task,
            target_enabled,
            _run_from_record(run_row),
            interaction,
            cast(Mapping[str, object], request),
        )

    async def _lock_result_reply_subject(
        self,
        connection: asyncpg.Connection,
        run_id: UUID,
        fish_id: UUID,
        session_id: UUID,
    ) -> (
        tuple[
            Mapping[str, object],
            bool,
            AgentRun,
            Mapping[str, object],
            AgentRun | None,
        ]
        | None
    ):
        task_id = await connection.fetchval(
            "SELECT task_id FROM agent_runs WHERE id = $1",
            run_id,
        )
        if not isinstance(task_id, UUID):
            return None
        locked_task = await self._lock_task_and_target(connection, task_id)
        if locked_task is None:
            return None
        task, target_enabled = locked_task
        if task["fish_id"] != fish_id or task["session_id"] != session_id:
            return None
        run_row = await connection.fetchrow(
            f"""
            SELECT {_RUN_COLUMNS}
            FROM agent_runs
            WHERE id = $1
              AND task_id = $2
              AND target_id = $3
            FOR UPDATE
            """,
            run_id,
            task_id,
            task["target_id"],
        )
        if run_row is None:
            return None
        run = _run_from_record(run_row)
        if (
            run.status
            not in {
                AgentRunStatus.COMPLETED,
                AgentRunStatus.FAILED,
                AgentRunStatus.CANCELLED,
            }
            or run.result_message_id is None
        ):
            return None
        child_row = await connection.fetchrow(
            f"""
            SELECT {_RUN_COLUMNS}
            FROM agent_runs
            WHERE previous_run_id = $1
            FOR UPDATE
            """,
            run_id,
        )
        result = await connection.fetchrow(
            """
            SELECT id, fish_id, session_id, role, content, status,
                   consumed_offset, fact_extraction_status
            FROM messages
            WHERE id = $1
            FOR UPDATE
            """,
            run.result_message_id,
        )
        if result is None:
            return None
        _require_locked_reply_source(result, fish_id, session_id)
        return (
            task,
            target_enabled,
            run,
            cast(Mapping[str, object], result),
            None if child_row is None else _run_from_record(child_row),
        )

    async def _load_existing_reply_message(
        self,
        connection: asyncpg.Connection,
        message_id: UUID | None,
        fish_id: UUID,
        session_id: UUID,
        content: str,
    ) -> Message | None:
        if message_id is None:
            return None
        row = await connection.fetchrow(
            """
            SELECT id, fish_id, session_id, sequence, role, content, status,
                   created_at, context_through_sequence, consumed_offset,
                   finished_at, history_token_count
            FROM messages
            WHERE id = $1
            FOR UPDATE
            """,
            message_id,
        )
        if row is None:
            return None
        message = _message_from_reply_record(row)
        _require_reply_message_identity(message, fish_id, session_id, content)
        return message

    async def _load_direct_reply_context(
        self,
        executor: asyncpg.Pool | asyncpg.Connection,
        message_id: UUID,
    ) -> AgentReplyContext | None:
        rows = await executor.fetch(
            """
            WITH interaction_direct AS (
                SELECT 'interaction'::text AS subject_kind,
                       interaction.id AS subject_id,
                       interaction.kind AS interaction_kind,
                       interaction.run_id,
                       interaction.next_run_id,
                       interaction.resolved_at,
                       interaction.closed_at
                FROM agent_interactions AS interaction
                WHERE interaction.response_message_id = $1
            ),
            result_direct AS (
                SELECT 'run_result'::text AS subject_kind,
                       child.previous_run_id AS subject_id
                FROM agent_runs AS child
                WHERE child.source_message_id = $1
                  AND child.confirmation_message_id IS NULL
                  AND child.previous_run_id IS NOT NULL
                  AND NOT EXISTS (
                      SELECT 1
                      FROM interaction_direct
                      WHERE interaction_direct.interaction_kind = 'input'
                        AND interaction_direct.run_id = child.previous_run_id
                        AND interaction_direct.next_run_id = child.id
                        AND interaction_direct.resolved_at IS NOT NULL
                        AND interaction_direct.closed_at IS NULL
                  )
            )
            SELECT subject_kind, subject_id FROM interaction_direct
            UNION ALL
            SELECT subject_kind, subject_id FROM result_direct
            """,
            message_id,
        )
        contexts = {
            AgentReplyContext(
                AgentMessageSubjectKind(cast(str, row["subject_kind"])),
                cast(UUID, row["subject_id"]),
            )
            for row in rows
        }
        if len(contexts) > 1:
            raise AgentReplyConflictError(
                "reply Message has ambiguous direct Agent associations"
            )
        return next(iter(contexts), None)

    async def _authoritative_reply_handled(
        self,
        executor: asyncpg.Pool | asyncpg.Connection,
        message_id: UUID,
        context: AgentReplyContext,
    ) -> bool:
        direct = await self._load_direct_reply_context(executor, message_id)
        if direct is None:
            return False
        if direct != context:
            raise AgentReplyConflictError(
                "reply Message is directly associated with another Agent subject"
            )
        return True

    async def _require_unassociated_direct_reply_message(
        self,
        connection: asyncpg.Connection,
        message_id: UUID,
    ) -> None:
        if await self._load_direct_reply_context(connection, message_id) is not None:
            raise AgentReplyConflictError(
                "reply Message is already associated with another Agent subject"
            )

    async def _insert_run(
        self,
        connection: asyncpg.Connection,
        *,
        task_id: UUID,
        target_id: UUID,
        fish_id: UUID,
        session_id: UUID,
        previous_run_id: UUID,
        source_message_id: UUID,
        confirmation_message_id: UUID | None,
        input: str,
    ) -> AgentRun:
        row = await connection.fetchrow(
            f"""
            INSERT INTO agent_runs (
                id,
                task_id,
                target_id,
                fish_id,
                session_id,
                previous_run_id,
                source_message_id,
                confirmation_message_id,
                input,
                status
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, 'queued')
            RETURNING {_RUN_COLUMNS}
            """,
            uuid4(),
            task_id,
            target_id,
            fish_id,
            session_id,
            previous_run_id,
            source_message_id,
            confirmation_message_id,
            input,
        )
        await connection.execute(
            """
            UPDATE agent_tasks
            SET updated_at = date_trunc('second', now())
            WHERE id = $1
            """,
            task_id,
        )
        return _required_run(row)

    async def _has_active_run(
        self,
        connection: asyncpg.Connection,
        task_id: UUID,
    ) -> bool:
        return bool(
            await connection.fetchval(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM agent_runs
                    WHERE task_id = $1
                      AND status IN (
                          'queued', 'dispatching', 'running', 'waiting_user', 'unknown'
                      )
                )
                """,
                task_id,
            )
        )

    async def _lock_run_messages(
        self,
        connection: asyncpg.Connection,
        fish_id: UUID,
        source_message_id: UUID,
        confirmation_message_id: UUID | None,
        *,
        expected_session_id: UUID | None = None,
    ) -> UUID:
        confirmation: Mapping[str, object] | None = None
        if confirmation_message_id is not None:
            confirmation = await self._lock_unbound_assistant(
                connection,
                confirmation_message_id,
            )
        source = await connection.fetchrow(
            """
            SELECT id, fish_id, session_id, role
            FROM messages
            WHERE id = $1
            FOR UPDATE
            """,
            source_message_id,
        )
        if (
            source is None
            or source["fish_id"] != fish_id
            or source["role"] != Role.USER
        ):
            raise AgentRunConflictError(
                "source Message is not a User Message for the Fish"
            )
        session_id = cast(UUID, source["session_id"])
        if expected_session_id is not None and session_id != expected_session_id:
            raise AgentRunConflictError(
                "source Message does not match Agent task Session"
            )
        if confirmation is not None and (
            confirmation["fish_id"] != fish_id
            or confirmation["session_id"] != session_id
            or confirmation["role"] != Role.ASSISTANT
        ):
            raise AgentMessageSubjectConflictError(
                "confirmation Message does not match source Fish and Session"
            )
        return session_id

    async def _lock_unbound_assistant(
        self,
        connection: asyncpg.Connection,
        message_id: UUID,
    ) -> Mapping[str, object]:
        message = await connection.fetchrow(
            """
            SELECT id, fish_id, session_id, role
            FROM messages
            WHERE id = $1
            FOR UPDATE
            """,
            message_id,
        )
        if message is None or message["role"] != Role.ASSISTANT:
            raise AgentMessageSubjectConflictError("subject Message is not Assistant")
        bound = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1 FROM agent_runs
                WHERE confirmation_message_id = $1 OR result_message_id = $1
                UNION ALL
                SELECT 1 FROM agent_interactions WHERE request_message_id = $1
            )
            """,
            message_id,
        )
        if bound:
            raise AgentMessageSubjectConflictError(
                "Assistant Message already has a subject"
            )
        return cast(Mapping[str, object], message)


def _require_reply_source(
    record: Mapping[str, object],
    fish_id: UUID,
    session_id: UUID,
) -> None:
    if (
        record["task_fish_id"] != fish_id
        or record["source_message_id"] is None
        or record["source_fish_id"] != fish_id
        or record["source_session_id"] != session_id
        or record["source_role"] != Role.ASSISTANT
        or record["source_status"]
        not in {MessageStatus.COMPLETED, MessageStatus.INTERRUPTED}
        or record["source_fact_status"] == "waiting_consumption"
    ):
        raise AgentReplyConflictError("reply subject is not replyable")
    length = record["source_content_length"]
    offset = record["source_consumed_offset"]
    if (
        not isinstance(length, int)
        or length > _MAX_MATERIALIZED_CHARS
        or not isinstance(offset, int)
        or (offset != -1 and offset < length)
    ):
        raise AgentReplyConflictError("reply subject is not replyable")


def _require_locked_reply_source(
    record: Mapping[str, object],
    fish_id: UUID,
    session_id: UUID,
) -> None:
    content = record["content"]
    offset = record["consumed_offset"]
    if (
        record["fish_id"] != fish_id
        or record["session_id"] != session_id
        or record["role"] != Role.ASSISTANT
        or record["status"] not in {MessageStatus.COMPLETED, MessageStatus.INTERRUPTED}
        or record["fact_extraction_status"] == "waiting_consumption"
        or not isinstance(content, str)
        or len(content) > _MAX_MATERIALIZED_CHARS
        or not isinstance(offset, int)
        or (offset != -1 and offset < len(content))
    ):
        raise AgentReplyConflictError("reply subject is not replyable")


def _existing_reply_message(
    record: Mapping[str, object],
    fish_id: UUID,
    session_id: UUID,
    *,
    message_id: UUID | None,
    content: str | None,
) -> Message | None:
    if record["existing_id"] is None:
        return None
    if message_id is None or content is None:
        raise AgentReplyConflictError("reply replay identity is incomplete")
    message = Message(
        id=cast(UUID, record["existing_id"]),
        fish_id=cast(UUID, record["existing_fish_id"]),
        session_id=cast(UUID, record["existing_session_id"]),
        sequence=cast(int, record["existing_sequence"]),
        role=Role(cast(str, record["existing_role"])),
        content=cast(str, record["existing_content"]),
        status=MessageStatus(cast(str, record["existing_status"])),
        created_at=cast(datetime, record["existing_created_at"]),
        context_through_sequence=cast(int | None, record["existing_context_sequence"]),
        consumed_offset=cast(int | None, record["existing_consumed_offset"]),
        finished_at=cast(datetime | None, record["existing_finished_at"]),
        history_token_count=cast(int, record["existing_history_tokens"]),
    )
    if message.id != message_id:
        raise AgentReplyConflictError("reply replay identity changed")
    _require_reply_message_identity(message, fish_id, session_id, content)
    return message


def _message_from_reply_record(record: Mapping[str, object]) -> Message:
    return Message(
        id=cast(UUID, record["id"]),
        fish_id=cast(UUID, record["fish_id"]),
        session_id=cast(UUID, record["session_id"]),
        sequence=cast(int, record["sequence"]),
        role=Role(cast(str, record["role"])),
        content=cast(str, record["content"]),
        status=MessageStatus(cast(str, record["status"])),
        created_at=cast(datetime, record["created_at"]),
        context_through_sequence=cast(int | None, record["context_through_sequence"]),
        consumed_offset=cast(int | None, record["consumed_offset"]),
        finished_at=cast(datetime | None, record["finished_at"]),
        history_token_count=cast(int, record["history_token_count"]),
    )


def _require_reply_message_identity(
    message: Message,
    fish_id: UUID,
    session_id: UUID,
    content: str,
) -> None:
    if (
        message.fish_id != fish_id
        or message.session_id != session_id
        or message.role is not Role.USER
        or message.status is not MessageStatus.INITIALIZED
        or message.content != content
    ):
        raise MessageIdentityConflictError(str(message.id))


def _allowed_decisions(provider_data: object) -> tuple[str, ...]:
    provider = _json_mapping(provider_data) or {}
    raw = provider.get("allowed_decisions")
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise AgentReplyConflictError("approval decisions are malformed")
    if len(raw) > 16:
        raise AgentReplyConflictError("approval decisions exceed the reply bound")
    decisions: list[str] = []
    for value in raw:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > _MAX_EXTERNAL_ID_CHARS
        ):
            raise AgentReplyConflictError("approval decisions are malformed")
        decisions.append(value)
    return tuple(decisions)


def _recovered_without_details(provider_data: object) -> bool:
    provider = _json_mapping(provider_data) or {}
    return provider.get("recoveredWithoutDetails") is True


def _bounded(
    name: str,
    value: str,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must not be empty")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")


def _bounded_integer(
    name: str,
    value: int,
    *,
    minimum: int,
    maximum: int,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")


def _validated_connector_type_filter(
    values: Collection[str] | None,
) -> list[str] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError("supported connector types must be a collection of slugs")
    normalized: set[str] = set()
    for value in values:
        try:
            normalized.add(str(ConnectorTypeSlug(value)))
        except (TypeError, ValueError) as error:
            raise ValueError(
                "supported connector types contain an invalid slug"
            ) from error
    return sorted(normalized)


def _provider_json(
    value: Mapping[str, object] | None,
    *,
    reserve_bytes: int = 0,
) -> str | None:
    return canonical_json_object(
        value,
        label="provider_data",
        maximum_bytes=_MAX_PROVIDER_BYTES,
        reserve_bytes=reserve_bytes,
    )


def _merged_run_provider_json(
    current_value: object,
    incoming_value: Mapping[str, object] | None,
) -> str | None:
    current = dict(_json_mapping(current_value) or {})
    incoming = dict(incoming_value or {})
    try:
        _provider_json(incoming)
    except ValueError as error:
        if "exceeds" not in str(error):
            raise

    local_steer = current.pop("steer", None)
    incoming.pop("steer", None)
    merged = {**current, **incoming}
    if local_steer is not None:
        merged["steer"] = local_steer
    try:
        return _provider_json(merged)
    except ValueError as error:
        if "exceeds" not in str(error):
            raise

    selected: dict[str, object] = {}
    if local_steer is not None:
        selected["steer"] = local_steer
        _provider_json(selected)

    incoming_priority = ("pending_steer", "output_truncated", "provider_status")
    incoming_keys = [
        *[key for key in incoming_priority if key in incoming],
        *sorted(key for key in incoming if key not in incoming_priority),
    ]
    for key in incoming_keys:
        _add_provider_field_if_bounded(selected, key, incoming[key])
    for key in sorted(current):
        if key not in selected and key not in incoming:
            _add_provider_field_if_bounded(selected, key, current[key])
    return _provider_json(selected)


def _validate_incoming_run_provider(
    value: Mapping[str, object] | None,
) -> None:
    try:
        _provider_json(value)
    except ValueError as error:
        if "exceeds" in str(error):
            return
        raise


def _add_provider_field_if_bounded(
    selected: dict[str, object],
    key: str,
    value: object,
) -> None:
    candidate = {**selected, key: value}
    try:
        _provider_json(candidate)
    except ValueError as error:
        if "exceeds" in str(error):
            return
        raise
    selected[key] = value


def _interaction_provider_json(value: Mapping[str, object] | None) -> str | None:
    _reject_interaction_text_fields(value)
    return _provider_json(value)


def _reject_interaction_text_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and key.casefold() in {
                "prompt",
                "response",
                "options",
            }:
                raise ValueError(
                    "provider_data must not duplicate prompt, response, or options text"
                )
            _reject_interaction_text_fields(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_interaction_text_fields(child)


def _json_mapping(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    if isinstance(value, str):
        decoded = json.loads(value)
        return cast(Mapping[str, object], decoded)
    return cast(Mapping[str, object], value)


def _approval_response(
    decision: Mapping[str, object] | str,
) -> Mapping[str, object]:
    if isinstance(decision, str):
        choice = decision
        resolve_all = False
    elif isinstance(decision, Mapping):
        choice = decision.get("choice")
        resolve_all = decision.get("resolve_all", False)
        if not isinstance(choice, str) or not isinstance(resolve_all, bool):
            raise ValueError(
                "decision must contain string choice and boolean resolve_all"
            )
    else:
        raise ValueError("decision must be a choice or JSON object")
    _bounded("decision choice", choice, _MAX_EXTERNAL_ID_CHARS)
    return {"choice": choice, "resolve_all": resolve_all, "state": "queued"}


def _require_allowed_decision(provider_data: object, choice: object) -> None:
    provider = _json_mapping(provider_data) or {}
    allowed = provider.get("allowed_decisions")
    if (
        not isinstance(allowed, Sequence)
        or isinstance(allowed, (str, bytes, bytearray))
        or choice not in allowed
    ):
        raise AgentInteractionConflictError(
            "decision choice is not allowed by the interaction"
        )


def _required_task(record: Mapping[str, object] | None) -> AgentTask:
    if record is None:
        raise RuntimeError("database did not return Agent task")
    return _task_from_record(record)


def _task_from_record(record: Mapping[str, object], prefix: str = "") -> AgentTask:
    return AgentTask(
        id=cast(UUID, record[f"{prefix}id"]),
        fish_id=cast(UUID, record[f"{prefix}fish_id"]),
        session_id=cast(UUID, record[f"{prefix}session_id"]),
        target_id=cast(UUID, record[f"{prefix}target_id"]),
        external_session_id=cast(str, record[f"{prefix}external_session_id"]),
        title=cast(str, record[f"{prefix}title"]),
        created_at=cast(datetime, record[f"{prefix}created_at"]),
        updated_at=cast(datetime, record[f"{prefix}updated_at"]),
    )


def _required_run(record: Mapping[str, object] | None) -> AgentRun:
    if record is None:
        raise RuntimeError("database did not return Agent run")
    return _run_from_record(record)


def _run_from_record(record: Mapping[str, object], prefix: str = "") -> AgentRun:
    return AgentRun(
        id=cast(UUID, record[f"{prefix}id"]),
        task_id=cast(UUID, record[f"{prefix}task_id"]),
        target_id=cast(UUID, record[f"{prefix}target_id"]),
        fish_id=cast(UUID, record[f"{prefix}fish_id"]),
        session_id=cast(UUID, record[f"{prefix}session_id"]),
        previous_run_id=cast(UUID | None, record[f"{prefix}previous_run_id"]),
        source_message_id=cast(UUID | None, record[f"{prefix}source_message_id"]),
        confirmation_message_id=cast(
            UUID | None, record[f"{prefix}confirmation_message_id"]
        ),
        input=cast(str, record[f"{prefix}input"]),
        external_run_id=cast(str | None, record[f"{prefix}external_run_id"]),
        history_source=AgentHistorySource(cast(str, record[f"{prefix}history_source"])),
        status=AgentRunStatus(cast(str, record[f"{prefix}status"])),
        output=cast(str | None, record[f"{prefix}output"]),
        result_message_id=cast(UUID | None, record[f"{prefix}result_message_id"]),
        error_code=cast(str | None, record[f"{prefix}error_code"]),
        error_message=cast(str | None, record[f"{prefix}error_message"]),
        provider_data=_json_mapping(record[f"{prefix}provider_data"]),
        created_at=cast(datetime, record[f"{prefix}created_at"]),
        started_at=cast(datetime | None, record[f"{prefix}started_at"]),
        updated_at=cast(datetime, record[f"{prefix}updated_at"]),
        finished_at=cast(datetime | None, record[f"{prefix}finished_at"]),
        origin=cast(str, record.get(f"{prefix}origin", "user")),
        origin_context=_json_mapping(record.get(f"{prefix}origin_context")),
    )


def _required_interaction(
    record: Mapping[str, object] | None,
) -> AgentInteraction:
    if record is None:
        raise RuntimeError("database did not return Agent interaction")
    return _interaction_from_record(record)


def _interaction_from_record(
    record: Mapping[str, object],
    prefix: str = "",
) -> AgentInteraction:
    return AgentInteraction(
        id=cast(UUID, record[f"{prefix}id"]),
        run_id=cast(UUID, record[f"{prefix}run_id"]),
        fish_id=cast(UUID, record[f"{prefix}fish_id"]),
        session_id=cast(UUID, record[f"{prefix}session_id"]),
        kind=AgentInteractionKind(cast(str, record[f"{prefix}kind"])),
        request_message_id=cast(UUID | None, record[f"{prefix}request_message_id"]),
        response_message_id=cast(UUID | None, record[f"{prefix}response_message_id"]),
        next_run_id=cast(UUID | None, record[f"{prefix}next_run_id"]),
        provider_data=_json_mapping(record[f"{prefix}provider_data"]),
        created_at=cast(datetime, record[f"{prefix}created_at"]),
        resolved_at=cast(datetime | None, record[f"{prefix}resolved_at"]),
        closed_at=cast(datetime | None, record[f"{prefix}closed_at"]),
    )


def _work_item_from_record(record: Mapping[str, object]) -> AgentRunWorkItem:
    return AgentRunWorkItem(
        run=_run_from_record(record, "run_"),
        task=_task_from_record(record, "task_"),
        target=_target_from_record(record),
        endpoint=_endpoint_from_record(record),
    )


def _candidate_from_record(record: Mapping[str, object]) -> AgentTaskCandidate:
    work = _work_item_from_record(record)
    interaction = (
        None
        if record["interaction_id"] is None
        else _interaction_from_record(record, "interaction_")
    )
    return AgentTaskCandidate(
        task=work.task,
        run=work.run,
        interaction=interaction,
        target=work.target,
        endpoint=work.endpoint,
        interaction_request_content=(
            None
            if interaction is None
            else cast(str, record["interaction_request_content"])
        ),
        interaction_request_session_id=(
            None
            if interaction is None
            else cast(UUID, record["interaction_request_session_id"])
        ),
    )


def _interaction_work_item_from_record(
    record: Mapping[str, object],
) -> AgentInteractionWorkItem:
    work = _work_item_from_record(record)
    return AgentInteractionWorkItem(
        interaction=_interaction_from_record(record, "interaction_"),
        run=work.run,
        task=work.task,
        target=work.target,
        endpoint=work.endpoint,
    )


def _target_from_record(record: Mapping[str, object]) -> AgentTarget:
    return AgentTarget(
        id=cast(UUID, record["target_id"]),
        endpoint_id=cast(UUID, record["target_endpoint_id"]),
        external_key=cast(str, record["target_external_key"]),
        route_prefix=cast(str, record["target_route_prefix"]),
        credential_ref=cast(str | None, record["target_credential_ref"]),
        connector_config=_json_mapping(record["target_connector_config"]) or {},
        display_name=cast(str, record["target_display_name"]),
        enabled=cast(bool, record["target_enabled"]),
        created_at=cast(datetime, record["target_created_at"]),
        updated_at=cast(datetime, record["target_updated_at"]),
    )


def _endpoint_from_record(record: Mapping[str, object]) -> AgentEndpoint:
    return AgentEndpoint(
        id=cast(UUID, record["endpoint_id"]),
        connector_type=ConnectorTypeSlug(cast(str, record["endpoint_connector_type"])),
        base_url=cast(str, record["endpoint_base_url"]),
        created_at=cast(datetime, record["endpoint_created_at"]),
        updated_at=cast(datetime, record["endpoint_updated_at"]),
    )


def _agent_envelope_from_record(record: Mapping[str, object]) -> AgentMessageEnvelope:
    interaction_id = cast(UUID | None, record["interaction_id"])
    if interaction_id is not None:
        subject = AgentMessageSubject(
            AgentMessageSubjectKind.INTERACTION, interaction_id
        )
        policy = (
            AgentActivationPolicy.PLAYBACK_COMPLETE
            if record["interaction_kind"] == AgentInteractionKind.APPROVAL
            else AgentActivationPolicy.PLAYBACK_START
        )
    else:
        subject = AgentMessageSubject(
            AgentMessageSubjectKind.RUN_RESULT,
            cast(UUID, record["result_run_id"]),
        )
        policy = AgentActivationPolicy.PLAYBACK_START
    return AgentMessageEnvelope(
        message_id=cast(UUID, record["message_id"]),
        sequence=cast(int, record["sequence"]),
        session_id=cast(UUID, record["session_id"]),
        subject=subject,
        activation_policy=policy,
        content=cast(str, record["content"]),
        consumed_offset=cast(int, record["consumed_offset"]),
    )
