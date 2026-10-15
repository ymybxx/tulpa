import asyncio
from dataclasses import dataclass, field
from uuid import UUID

from fish_memory.models import Message, MessageStatus
from fish_memory.performance.tracker import MessagePerformanceTracker
from fish_memory.pet.types import PetEmotion


@dataclass(slots=True)
class FishRuntime:
    latest_user_message: Message | None = None
    answer_message_id: UUID | None = None
    answer_sequence: int | None = None
    context_through_sequence: int | None = None
    content: str = ""
    speech_content: str = ""
    speech_tags: bool = False
    flush_offset: int = 0
    consumed_offset: int = -1
    playback_owner: str = "web"
    status: MessageStatus | None = None
    # See Message.failure.
    failure: str | None = None
    end_conversation: bool = False
    pet_emotion: PetEmotion = PetEmotion.NEUTRAL
    pet_metadata_revision: int = 0
    source_message_id: UUID | None = None
    performance_tracker: MessagePerformanceTracker | None = None
    llm_task: asyncio.Task[None] | None = None
    generation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    condition: asyncio.Condition = field(default_factory=asyncio.Condition)
