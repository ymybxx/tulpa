from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from .client import Pet
from .review import ReviewProfile

Data = TypeVar("Data", bound=BaseModel)


class Activity(BaseModel):
    """What this running plugin asks the pet to do; not its permanent identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=100)
    role: str = Field(min_length=1, max_length=100)
    audience: str = Field(min_length=1, max_length=200)
    objective: str = Field(min_length=1, max_length=600)


@dataclass(frozen=True)
class Event(Generic[Data]):
    type: str
    data: type[Data]
    description: str
    speak: bool = True
    include_in_context: bool = True
    # Accepted for older plugins; the host now always owns speech context.
    context_fields: tuple[str, ...] | None = None
    mergeable: bool = False

    def definition(self) -> dict[str, Any]:
        if type(self.speak) is not bool:
            raise ValueError("speak must be false or true")
        if type(self.mergeable) is not bool:
            raise ValueError("mergeable must be false or true")
        schema = self.data.model_json_schema()
        schema["description"] = self.description
        result = {"type": self.type, "schema": schema, "speak": self.speak}
        if not self.include_in_context:
            result["include_in_context"] = False
        if self.mergeable:
            result["mergeable"] = True
        return result


class EmptyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class Input(Generic[Data]):
    """An App form delivered to the running plugin, before its event processing."""

    type: str
    data: type[Data]
    title: str
    description: str = ""
    submit_label: str = "发送"

    def definition(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "schema": self.data.model_json_schema(),
            "title": self.title,
            "description": self.description,
            "submit_label": self.submit_label,
        }


class Plugin:
    """Declare metadata without I/O. Put platform connections inside run()."""

    Config: type[BaseModel] = EmptyConfig
    events: tuple[Event[Any], ...] = ()
    inputs: tuple[Input[Any], ...] = ()
    activity: Activity | None = None
    review: ReviewProfile | None = None

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = self.Config.model_validate(config)

    @classmethod
    def describe(cls) -> dict[str, Any]:
        result = {
            "protocol": 1,
            "config_schema": cls.Config.model_json_schema(),
            "events": [event.definition() for event in cls.events],
        }
        if cls.review is not None:
            result["review"] = cls.review.model_dump(mode="json")
        if cls.activity is not None:
            result["activity"] = cls.activity.model_dump(mode="json")
        if cls.inputs:
            result["inputs"] = [value.definition() for value in cls.inputs]
        return result

    async def input(
        self, pet: Pet, type: str, data: BaseModel, *, request_id: str
    ) -> dict[str, Any]:
        """Return an admission receipt; it does not mean the pet has spoken."""
        raise ValueError("Unsupported plugin input")

    async def run(self, pet: Pet) -> None:
        """Cancellation means stop; use async with/finally to close resources."""
        raise NotImplementedError
