"""Views of published context, without recapturing or resummarizing anything."""

from typing import Any


def perception_view(value: Any, *, recent_chars: int = 6000) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"available": False, "recent": "", "earlier": ""}
    # Published perception is chronological, with the newest raw audio last.
    # Keep all content for generation, but put its recent end in a separate field.
    if "recent" in value:
        return dict(value)
    text = str(value.get("text") or "")
    cut = max(0, len(text) - recent_chars)
    if cut:
        boundary = text.find("\n\n", cut)
        if boundary != -1:
            cut = boundary + 2
    return {
        "available": value.get("available", False),
        "earlier": text[:cut],
        "recent": text[cut:],
    }


def perception_evidence(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    view = perception_view(value, recent_chars=2800)
    recent, earlier = str(view.get("recent", "")), str(view.get("earlier", ""))
    # The reviewer gets the same most recent material, never just the oldest
    # 1200 characters of a one-hour recap. Mark omission explicitly.
    return {
        "available": view.get("available", False),
        "text": recent[-2800:],
        "earlier_excerpt": earlier[-400:],
        "omitted_chars": max(0, len(recent) - 2800) + max(0, len(earlier) - 400),
    }


def conversation_progress(history: list[dict[str, Any]]) -> dict[str, Any]:
    latest = next(
        (
            i
            for i in range(len(history) - 1, -1, -1)
            if history[i].get("role") == "user"
        ),
        None,
    )
    if latest is None:
        return {"latest_user_message_id": None, "assistant_responses_since_user": 0}
    return {
        "latest_user_message_id": history[latest].get("id"),
        "assistant_responses_since_user": sum(
            m.get("role") == "assistant" for m in history[latest + 1 :]
        ),
    }
