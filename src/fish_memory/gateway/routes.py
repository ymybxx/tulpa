from typing import Any

from fastapi import APIRouter, HTTPException, Request

from fish_memory.gateway.delivery import GatewayEvent

router = APIRouter(prefix="/internal/gateway", tags=["gateway"])


@router.post("/events", status_code=202, include_in_schema=False)
async def receive_event(event: GatewayEvent, request: Request) -> dict[str, Any]:
    deliver = getattr(request.app.state, "gateway_delivery", None)
    if deliver is None:
        raise HTTPException(503, "TA 还没准备好")
    identifier = await deliver(event)
    if identifier is None:
        raise HTTPException(409, "还没有可以转告的角色")
    return {"accepted": True, "id": str(identifier)}
