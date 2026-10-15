from __future__ import annotations

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from fish_memory.auth.credentials import EnvironmentApiKeyRotationError
from fish_memory.auth.service import ApiAuthService

auth_router = APIRouter(prefix="/v1/auth", tags=["authentication"])


class UnlockRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    api_key: str = Field(min_length=1, validation_alias="apiKey")


def _service(request: Request) -> ApiAuthService:
    return request.app.state.auth_service


@auth_router.get("/status")
async def auth_status(request: Request) -> dict[str, bool]:
    principal = _service(request).authenticate_headers(list(request.scope["headers"]))
    return {
        "authenticated": principal is not None,
        "fileBacked": _service(request).is_file_backed,
    }


@auth_router.post("/unlock", status_code=status.HTTP_204_NO_CONTENT)
async def unlock(request: Request, payload: UnlockRequest) -> Response:
    service = _service(request)
    if not service.verify_api_key(payload.api_key):
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
            content={
                "error": {
                    "code": "unauthorized",
                    "message": "invalid API key",
                }
            },
        )
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    _set_session_cookie(request, response, service.create_browser_session())
    return response


@auth_router.post("/rotate")
async def rotate_api_key(request: Request) -> JSONResponse:
    try:
        credential = _service(request).rotate()
    except EnvironmentApiKeyRotationError as error:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error": {
                    "code": "environment_key_not_rotatable",
                    "message": str(error),
                }
            },
        )
    response = JSONResponse(content={"apiKey": credential.api_key})
    _set_session_cookie(request, response, credential.browser_session)
    return response


def _set_session_cookie(request: Request, response: Response, session: str) -> None:
    response.set_cookie(
        key=_service(request).cookie_name,
        value=session,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        path="/",
    )
