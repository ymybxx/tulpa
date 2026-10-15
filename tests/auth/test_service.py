from pathlib import Path

from fish_memory.auth.credentials import ApiKeyCredentialStore
from fish_memory.auth.service import ApiAuthService


def _service(key_file: Path) -> ApiAuthService:
    return ApiAuthService(ApiKeyCredentialStore(api_key=None, key_file=key_file))


def _cookie_headers(
    service: ApiAuthService,
    session: str,
) -> list[tuple[bytes, bytes]]:
    return [(b"cookie", f"{service.cookie_name}={session}".encode())]


def test_browser_session_survives_service_restart(tmp_path: Path) -> None:
    key_file = tmp_path / "api-key"
    first = _service(key_file)
    session = first.create_browser_session()
    restarted = _service(key_file)

    principal = restarted.authenticate_headers(_cookie_headers(restarted, session))

    assert principal is not None
    assert principal.kind == "session"


def test_browser_session_is_invalid_after_api_key_rotation(tmp_path: Path) -> None:
    service = _service(tmp_path / "api-key")
    session = service.create_browser_session()

    service.rotate()

    assert service.authenticate_headers(_cookie_headers(service, session)) is None


def test_browser_session_rejects_modified_signature(tmp_path: Path) -> None:
    service = _service(tmp_path / "api-key")
    session = service.create_browser_session()
    replacement = "a" if session[-1] != "a" else "b"
    modified = f"{session[:-1]}{replacement}"

    assert service.authenticate_headers(_cookie_headers(service, modified)) is None
