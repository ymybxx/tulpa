from collections.abc import Sequence

_MECHANICAL_ROLE_PREFIXES = ("该用户", "该助手", "用户", "助手")
_GENERIC_ROLE_NAMES = frozenset(
    {
        "用户",
        "助手",
        "该用户",
        "该助手",
        "user",
        "assistant",
    }
)


def build_fact_embedding_text(
    *,
    text: str,
    what: str | None,
    where: Sequence[str],
    who: Sequence[str],
    entities: Sequence[str],
) -> str:
    """Build a semantic embedding input without repeated role boilerplate.

    ``text`` remains the standalone, source-aware fact shown to the LLM and
    callers. Embeddings use the structured semantics instead, because a
    repeated ``该用户``/``该助手`` prefix is metadata rather than the topic
    being searched.
    """

    parts: list[str] = []
    _append_semantic_part(parts, what or text)
    for value in (*where, *who, *entities):
        if value.strip() in _GENERIC_ROLE_NAMES:
            continue
        _append_semantic_part(parts, value)
    return " ".join(parts)


def _append_semantic_part(parts: list[str], value: str) -> None:
    normalized = _strip_mechanical_role_prefix(value)
    if not normalized:
        return
    if any(normalized in existing or existing in normalized for existing in parts):
        return
    parts.append(normalized)


def _strip_mechanical_role_prefix(value: str) -> str:
    normalized = value.strip()
    for prefix in _MECHANICAL_ROLE_PREFIXES:
        if normalized.startswith(prefix) and len(normalized) > len(prefix):
            remainder = normalized[len(prefix) :].lstrip(" ：:，,")
            if remainder:
                return remainder
    return normalized
