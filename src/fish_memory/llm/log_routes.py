from datetime import UTC, datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import AwareDatetime, BaseModel, Field

router = APIRouter(
    prefix="/internal/llm-logs", tags=["llm-logs"], include_in_schema=False
)


class DesktopCallLog(BaseModel):
    id: UUID
    call_type: str = Field(pattern=r"^[a-z][a-z0-9_.-]{0,119}$")
    model: str = Field(max_length=300)
    request_url: str = Field(max_length=2000)
    status: Literal["completed", "failed", "cancelled"]
    started_at: AwareDatetime
    ended_at: AwareDatetime
    duration_ms: int = Field(ge=0)
    http_status: int | None = None
    error: str | None = Field(None, max_length=2000)
    request: Any
    response: Any


@router.post("")
async def record_desktop_call(value: DesktopCallLog, request: Request):
    from urllib.parse import urlsplit, urlunsplit

    url = urlsplit(value.request_url)
    safe_url = urlunsplit((url.scheme, url.netloc.rsplit("@", 1)[-1], url.path, "", ""))
    record = {
        **value.model_dump(),
        "id": str(value.id),
        "request_url": safe_url,
        "source": "desktop",
    }
    await request.app.state.llm_log_store.save(record)
    return {"recorded": True}


@router.get("")
async def list_logs(
    request: Request,
    call_type: str | None = Query(None, max_length=120),
    status: Literal["running", "completed", "failed", "cancelled"] | None = None,
    before: int | None = Query(None, ge=1),
    limit: int = Query(40, ge=1, le=100),
):
    result = await request.app.state.llm_log_store.list_calls(
        call_type=call_type, status=status, before=before, limit=limit
    )
    return {**result, "dropped_records": request.app.state.llm_call_journal.dropped}


@router.get("/usage")
async def usage(
    request: Request,
    day_start: Annotated[AwareDatetime, Query(alias="dayStart")],
):
    """Token use by call type and purpose, for the user's own day."""
    now = datetime.now(UTC)
    if day_start > now:
        raise HTTPException(422, "dayStart is in the future")
    rows = await request.app.state.llm_log_store.usage(day_start, now)
    return {"items": rows}


@router.get("/types")
async def list_types(request: Request):
    return {"types": await request.app.state.llm_log_store.types()}


@router.get("/{identifier}")
async def log_detail(identifier: UUID, request: Request):
    value = await request.app.state.llm_log_store.get(identifier)
    if value is None:
        raise HTTPException(404, "调用记录不存在")
    return value
