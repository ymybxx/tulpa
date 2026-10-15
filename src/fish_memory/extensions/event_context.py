"""Project plugin event data for the pet while retaining full diagnostic records."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from string import Template
from typing import Any

from fish_memory.clock import format_prompt_timestamp_text
from fish_memory.extensions.models import EventDefinition


def event_data_view(
    data: dict[str, Any], definition: EventDefinition
) -> dict[str, Any]:
    properties = definition.schema_.get("properties", {})
    result: dict[str, Any] = {}
    for key, value in data.items():
        field = properties.get(key, {})
        visibility = field.get("x-fish-context") if isinstance(field, dict) else None
        if visibility == "metadata":
            continue
        if visibility == "identity" and isinstance(value, str):
            # Stable across current/history/reasoning views, with no source prefix
            # exposed to the character. Raw platform identities remain in storage.
            scope = field.get("x-fish-identity-scope") or definition.type
            digest = hashlib.sha256(f"{scope}:{key}:{value}".encode()).hexdigest()
            value = "speaker-" + digest[:24]
        result[key] = _field_context(value, field)
    return result


def _field_context(value: Any, schema: Any) -> Any:
    if not isinstance(schema, dict):
        return deepcopy(value)
    if schema.get("format") == "date-time" and isinstance(value, str):
        return format_prompt_timestamp_text(value)
    if isinstance(value, dict):
        fields = schema.get("properties", {})
        return {k: _field_context(v, fields.get(k, {})) for k, v in value.items()}
    if isinstance(value, list):
        return [_field_context(v, schema.get("items", {})) for v in value]
    return deepcopy(value)


def event_schema_view(definition: EventDefinition) -> dict[str, Any]:
    schema = deepcopy(definition.schema_)
    schema.pop("x-fish-history-template", None)
    properties = schema.get("properties", {})
    hidden = set()
    for key, field in list(properties.items()):
        if not isinstance(field, dict):
            continue
        visibility = field.pop("x-fish-context", None)
        field.pop("x-fish-identity-scope", None)
        if visibility == "metadata":
            hidden.add(key)
            del properties[key]
    if "required" in schema:
        schema["required"] = [key for key in schema["required"] if key not in hidden]
    return {"type": definition.type, "schema": schema}


def event_history_text(data: dict[str, Any], definition: EventDefinition) -> str | None:
    """Render an optional plugin-owned narrative without another model call."""
    template = definition.schema_.get("x-fish-history-template")
    if template is None:
        return None
    try:
        # Substitute once: braces and dollar signs in audience text stay literal.
        return Template(template).substitute(event_data_view(data, definition))
    except (KeyError, ValueError):
        return None


def event_history_view(
    events: list[dict[str, Any]], definitions: Mapping[str, EventDefinition]
) -> list[dict[str, Any]]:
    result = []
    for event in events:
        definition = definitions.get(event.get("type", ""))
        if not event.get("include_in_context", True) or (
            definition is not None and not definition.include_in_context
        ):
            continue
        projected = dict(event)
        if isinstance(projected.get("received_at"), str):
            projected["received_at"] = format_prompt_timestamp_text(
                projected["received_at"]
            )
        if definition is not None and isinstance(event.get("data"), dict):
            projected["data"] = event_data_view(event["data"], definition)
        result.append(projected)
    return result
