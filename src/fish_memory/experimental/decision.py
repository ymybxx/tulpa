"""Which decision model a desktop configuration uses, read from its files.

Mirrors desktop/pet/src/shared/decision-model.ts for the facial test, which
reads config.json and secrets.json itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from fish_memory.experimental.config import JEV_ENDPOINT

JEV_DEFAULT_MODEL = "jev-1.13.0"
BAILIAN_DECISION_MODEL = "decision-model-preview"
BAILIAN_DECISION_REGION = "cn-beijing"
_WORKSPACE = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}$", re.IGNORECASE)


@dataclass(frozen=True)
class DecisionConnection:
    provider: str
    endpoint: str
    model: str
    api_key: str = field(repr=False)


def bailian_decision_endpoint(workspace_id: str) -> str:
    return (
        f"https://{workspace_id}.{BAILIAN_DECISION_REGION}.maas.aliyuncs.com"
        "/compatible-mode/v1/systemone"
    )


def is_bailian_url(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return (
        host == "dashscope.aliyuncs.com"
        or host.endswith(".dashscope.aliyuncs.com")
        or host.endswith(".maas.aliyuncs.com")
    )


def decision_connection(
    configuration: dict[str, Any], secrets: dict[str, Any]
) -> DecisionConnection | None:
    """The decision model to call, or None when none is connected."""
    companion = configuration.get("experimentalCompanion") or {}
    models = configuration.get("models") or {}
    keys = secrets.get("modelApiKeys") or {}

    def key(model_id: Any) -> str:
        value = keys.get(model_id)
        return value.strip() if isinstance(value, str) else ""

    jev_key = secrets.get("jevApiKey")
    jev_key = jev_key.strip() if isinstance(jev_key, str) else ""
    workspace = companion.get("bailianWorkspaceId")
    workspace = (
        workspace.strip()
        if isinstance(workspace, str) and _WORKSPACE.match(workspace.strip())
        else ""
    )
    bailian_key = next(
        (
            key(model.get("id"))
            for model in models.get("text") or []
            if is_bailian_url(str(model.get("baseUrl", ""))) and key(model.get("id"))
        ),
        "",
    ) or next(
        (
            key(model.get("id"))
            for model in models.get("voice") or []
            if model.get("provider") == "dashscope" and key(model.get("id"))
        ),
        "",
    )
    provider = companion.get("decisionProvider")
    if provider not in ("bailian", "jev"):
        provider = (
            "bailian" if workspace and bailian_key else "jev" if jev_key else "bailian"
        )
    if provider == "jev":
        return (
            DecisionConnection(
                "jev",
                JEV_ENDPOINT,
                companion.get("model") or JEV_DEFAULT_MODEL,
                jev_key,
            )
            if jev_key
            else None
        )
    if not (workspace and bailian_key):
        return None
    return DecisionConnection(
        "bailian",
        bailian_decision_endpoint(workspace),
        BAILIAN_DECISION_MODEL,
        bailian_key,
    )
