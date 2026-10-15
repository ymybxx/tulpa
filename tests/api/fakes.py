from datetime import UTC, datetime
from uuid import UUID

from fish_memory.answer.runtime import FishRuntime
from fish_memory.answer.voice_drafts import VoiceDraftLease
from fish_memory.db.messages import MessageIdentityConflictError
from fish_memory.db.sessions import SessionNotFoundError
from fish_memory.facts.models import BackgroundTaskProgress
from fish_memory.models import (
    FishSummary,
    Message,
    MessageStatus,
    Role,
    Session,
    SessionSummary,
)
from fish_memory.performance.models import PerformanceStep
from fish_memory.performance.tracker import MessagePerformanceTracker

DEFAULT_FISH_ID = UUID("00000000-0000-0000-0000-000000000001")
DEFAULT_SESSION_ID = UUID("00000000-0000-0000-0000-000000000101")
NOW = datetime(2026, 7, 20, 8, 30, tzinfo=UTC)


class FakeSessionStore:
    def __init__(self, fish_id: UUID = DEFAULT_FISH_ID) -> None:
        self.fish_id = fish_id
        self.sessions: list[Session] = [
            Session(
                id=DEFAULT_SESSION_ID,
                fish_id=fish_id,
                created_at=NOW,
                last_message_at=NOW,
            )
        ]

    async def create(self, fish_id: UUID) -> Session:
        if fish_id != self.fish_id:
            raise SessionNotFoundError(str(fish_id))
        session = Session(
            id=UUID(int=101 + len(self.sessions)),
            fish_id=fish_id,
            created_at=NOW,
            last_message_at=None,
        )
        self.sessions.insert(0, session)
        return session

    async def get_latest(self, fish_id: UUID) -> Session | None:
        if fish_id != self.fish_id or not self.sessions:
            return None
        return self.sessions[0]

    async def list(self, fish_id: UUID) -> list[SessionSummary]:
        if fish_id != self.fish_id:
            return []
        return [
            SessionSummary(
                id=session.id,
                fish_id=session.fish_id,
                created_at=session.created_at,
                last_message_at=session.last_message_at,
            )
            for session in self.sessions
        ]

    async def require_owned(self, fish_id: UUID, session_id: UUID) -> Session:
        for session in self.sessions:
            if session.fish_id == fish_id and session.id == session_id:
                return session
        raise SessionNotFoundError(str(session_id))


class FakeApiStore:
    def __init__(self) -> None:
        self.contents: list[tuple[UUID, str]] = []
        self.events: list[str] = []
        self.error: Exception | None = None
        self.fishes: list[FishSummary] = []
        self.messages: list[Message] = []
        self.message_limits: list[tuple[UUID, int]] = []
        self.message_sessions: list[UUID] = []
        self.user_messages_by_id: dict[UUID, Message] = {}
        self.created_ats: list[datetime | None] = []
        self.timeline_calls: list[tuple[UUID, int, int | None, int | None]] = []
        self.searches: list[tuple[UUID, str, int]] = []
        self.origins: dict[UUID, object] = {}
        self.input_modes_by_id: dict[UUID, str] = {}
        self.unsaid_by_id: dict[UUID, str] = {}

    async def insert_user_message(
        self,
        fish_id: UUID,
        content: str,
        *,
        session_id: UUID | None = None,
        created_at: datetime | None = None,
        message_id: UUID | None = None,
    ) -> Message:
        self.events.append("persist")
        if self.error is not None:
            raise self.error
        effective_message_id = message_id or UUID(
            "00000000-0000-0000-0000-000000000010"
        )
        effective_session_id = session_id or DEFAULT_SESSION_ID
        existing = self.user_messages_by_id.get(effective_message_id)
        if existing is not None:
            if (
                existing.fish_id != fish_id
                or existing.session_id != effective_session_id
                or existing.content != content
            ):
                raise MessageIdentityConflictError(str(effective_message_id))
            return existing
        self.contents.append((fish_id, content))
        self.created_ats.append(created_at)
        message = Message(
            id=effective_message_id,
            fish_id=fish_id,
            session_id=effective_session_id,
            sequence=7,
            role=Role.USER,
            content=content,
            status=MessageStatus.INITIALIZED,
            created_at=created_at or NOW,
        )
        self.user_messages_by_id[effective_message_id] = message
        return message

    async def list_fishes(self) -> list[FishSummary]:
        return self.fishes

    async def list_messages(
        self,
        fish_id: UUID,
        session_id: UUID,
        *,
        limit: int,
    ) -> list[Message]:
        assert session_id == DEFAULT_SESSION_ID
        self.message_limits.append((fish_id, limit))
        self.message_sessions.append(session_id)
        return self.messages

    async def list_timeline_page(
        self,
        fish_id: UUID,
        *,
        limit: int,
        before: int | None = None,
        after: int | None = None,
    ):
        from fish_memory.db.messages import MessagePage

        self.timeline_calls.append((fish_id, limit, before, after))
        items = self.messages[-limit:]
        return MessagePage(
            items=items,
            has_before=len(self.messages) > limit,
            has_after=False,
            min_sequence=items[0].sequence if items else None,
            max_sequence=items[-1].sequence if items else None,
        )

    async def search_messages(
        self, fish_id: UUID, query: str, *, limit: int
    ) -> list[Message]:
        self.searches.append((fish_id, query, limit))
        return [message for message in self.messages if query in message.content]

    async def record_input_mode(self, message_id: UUID, mode: str) -> None:
        self.input_modes_by_id.setdefault(message_id, mode)

    async def input_modes(self, fish_id: UUID, message_ids):
        return {
            key: value
            for key, value in self.input_modes_by_id.items()
            if key in message_ids
        }

    async def unsaid_texts(self, fish_id: UUID, message_ids):
        return {
            key: value for key, value in self.unsaid_by_id.items() if key in message_ids
        }

    async def message_origins(self, fish_id: UUID, message_ids):
        return {key: value for key, value in self.origins.items() if key in message_ids}


class FakeFactStore:
    def __init__(self, progress: BackgroundTaskProgress) -> None:
        self.progress = progress
        self.progress_calls: list[tuple[UUID, UUID]] = []

    async def get_progress(
        self,
        fish_id: UUID,
        session_id: UUID,
    ) -> BackgroundTaskProgress:
        self.progress_calls.append((fish_id, session_id))
        return self.progress


class FakePerformanceStore:
    def __init__(self) -> None:
        self.steps: list[PerformanceStep] = []
        self.calls: list[tuple[UUID, UUID]] = []
        self.tts_calls: list[tuple[UUID, UUID, dict[str, object]]] = []
        self.pet_voice_calls: list[tuple[UUID, UUID, dict[str, object]]] = []
        self.pet_voice_result = True

    async def list_message_steps(
        self,
        fish_id: UUID,
        message_id: UUID,
    ) -> list[PerformanceStep]:
        self.calls.append((fish_id, message_id))
        return self.steps

    async def replace_tts_metrics(
        self,
        fish_id: UUID,
        message_id: UUID,
        metrics: dict[str, object],
    ) -> bool:
        self.tts_calls.append((fish_id, message_id, metrics))
        return True

    async def replace_pet_voice_metrics(
        self,
        fish_id: UUID,
        message_id: UUID,
        metrics: dict[str, object],
    ) -> bool:
        self.pet_voice_calls.append((fish_id, message_id, metrics))
        return self.pet_voice_result


class FakeApiCoordinator:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.notifications: list[
            tuple[UUID, UUID, int, UUID, MessagePerformanceTracker]
        ] = []
        self.source_messages: list[Message] = []
        self.voice_notifications: list[
            tuple[
                UUID,
                UUID,
                int,
                UUID,
                UUID,
                MessagePerformanceTracker,
                VoiceDraftLease | None,
            ]
        ] = []
        self.runtime = FishRuntime()
        self.progress_result = 0
        self.progress_error: Exception | None = None
        self.progress_calls: list[tuple[UUID, UUID, int]] = []
        self.interrupt_calls: list[tuple[UUID, UUID, int]] = []
        self.delete_result = True
        self.delete_calls: list[UUID] = []

    def notify_user_message(
        self,
        fish_id: UUID,
        session_id: UUID,
        sequence: int,
        message_id: UUID,
        tracker: MessagePerformanceTracker,
        *,
        source_message: Message | None = None,
    ) -> None:
        if source_message is not None:
            assert source_message.id == message_id
            self.source_messages.append(source_message)
        self.events.append("notify")
        self.notifications.append((fish_id, session_id, sequence, message_id, tracker))

    def notify_voice_turn(
        self,
        fish_id: UUID,
        session_id: UUID,
        sequence: int,
        message_id: UUID,
        answer_message_id: UUID,
        tracker: MessagePerformanceTracker,
        *,
        source_message: Message,
        draft_lease: VoiceDraftLease | None,
    ) -> None:
        assert source_message.id == message_id
        self.events.append("notify_voice")
        self.source_messages.append(source_message)
        self.voice_notifications.append(
            (
                fish_id,
                session_id,
                sequence,
                message_id,
                answer_message_id,
                tracker,
                draft_lease,
            )
        )

    def get_runtime(self, fish_id: UUID, session_id: UUID) -> FishRuntime:
        del fish_id, session_id
        return self.runtime

    async def update_progress(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> int:
        self.progress_calls.append((fish_id, message_id, consumed_offset))
        if self.progress_error is not None:
            raise self.progress_error
        return self.progress_result

    async def interrupt_answer(
        self,
        fish_id: UUID,
        message_id: UUID,
        consumed_offset: int,
    ) -> None:
        self.interrupt_calls.append((fish_id, message_id, consumed_offset))
        if self.progress_error is not None:
            raise self.progress_error

    async def delete_fish(self, fish_id: UUID) -> bool:
        self.delete_calls.append(fish_id)
        return self.delete_result
