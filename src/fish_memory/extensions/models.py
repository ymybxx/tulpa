from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from string import Template
from typing import Any, Literal
from uuid import UUID, uuid4

from fish_plugin_sdk import Activity
from jsonschema import Draft202012Validator
from jsonschema.protocols import Validator
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 65_536
MAX_DEFINITIONS = 32


class ExtensionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class EventDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    type: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,95}$", strict=True)
    schema_: dict[str, Any] = Field(alias="schema")
    speak: bool = Field(default=True, strict=True)
    mergeable: bool = Field(default=False, strict=True)
    include_in_context: bool = Field(default=True, strict=True)
    context_fields: (
        list[Literal["memory", "history", "perception", "events"]] | None
    ) = None

    @field_validator("speak", mode="before")
    @classmethod
    def validate_speak(cls, value: object) -> object:
        if type(value) is not bool:
            raise ValueError("speak must be false or true")
        return value

    @field_validator("schema_")
    @classmethod
    def validate_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("event schema must describe an object")
        if len(json.dumps(value, ensure_ascii=False)) > 16_000:
            raise ValueError("event schema is too large")

        def check_refs(item: Any) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    if key in {"$ref", "$dynamicRef"} and (
                        not isinstance(child, str) or not child.startswith("#")
                    ):
                        raise ValueError("external schema references are not allowed")
                    check_refs(child)
            elif isinstance(item, list):
                for child in item:
                    check_refs(child)

        check_refs(value)
        Draft202012Validator.check_schema(value)
        template = value.get("x-fish-history-template")
        if template is not None:
            if not isinstance(template, str) or not template.strip():
                raise ValueError("history template must be non-empty text")
            parsed = Template(template)
            if not parsed.is_valid():
                raise ValueError("invalid history template")
            fields = value.get("properties", {})
            for name in parsed.get_identifiers():
                field = fields.get(name)
                if not isinstance(field, dict) or field.get("x-fish-context") in {
                    "metadata",
                    "identity",
                }:
                    raise ValueError("history template must use visible content fields")
        return value

    @property
    def revision(self) -> str:
        data = json.dumps(self.model_dump(by_alias=True), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


class CreateInstance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instance_id: UUID = Field(default_factory=uuid4)
    plugin_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,95}$", strict=True)
    fish_id: UUID
    session_id: UUID
    autonomous: bool = False


class InputDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    type: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,95}$", strict=True)
    title: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=600)
    submit_label: str = Field(default="发送", min_length=1, max_length=40)
    schema_: dict[str, Any] = Field(alias="schema")

    @field_validator("schema_")
    @classmethod
    def validate_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        return EventDefinition.validate_schema(value)


class PushEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: UUID
    type: str = Field(max_length=96, strict=True)
    data: dict[str, Any]


@dataclass(slots=True)
class Instance:
    id: UUID
    plugin_id: str
    fish_id: UUID
    session_id: UUID
    autonomous: bool
    token: str = field(repr=False)
    generation: UUID = field(default_factory=uuid4)
    connected: bool = False
    stopped: bool = False
    definitions: dict[str, EventDefinition] = field(default_factory=dict)
    validators: dict[str, Validator] = field(default_factory=dict)
    activity: Activity | None = None
    inputs: dict[str, InputDefinition] = field(default_factory=dict)


@dataclass(slots=True)
class PluginExpressionIntent:
    id: UUID
    fish_id: UUID
    session_id: UUID
    instance_id: UUID
    event_id: UUID
    observation_id: UUID
    state: str = "pending"
    text: str = ""
    merged_events: tuple[tuple[UUID, UUID], ...] = ()

    @property
    def event_references(self) -> tuple[tuple[UUID, UUID], ...]:
        return ((self.event_id, self.observation_id), *self.merged_events)
