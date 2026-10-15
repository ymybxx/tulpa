from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from http.cookies import SimpleCookie

from fish_memory.auth.credentials import ApiKeyCredentialStore


@dataclass(frozen=True, slots=True)
class AuthPrincipal:
    kind: str
    value: str


@dataclass(frozen=True, slots=True)
class RotatedCredential:
    api_key: str
    browser_session: str


class ApiAuthService:
    cookie_name = "fish_memory_session"

    def __init__(self, credentials: ApiKeyCredentialStore) -> None:
        self.credentials = credentials
        self._known_key = credentials.current_key()

    @property
    def is_file_backed(self) -> bool:
        return self.credentials.is_file_backed

    def authenticate_headers(
        self,
        headers: list[tuple[bytes, bytes]],
    ) -> AuthPrincipal | None:
        self._refresh_credentials()
        authorization: str | None = None
        cookie_header: str | None = None
        for raw_name, raw_value in headers:
            name = raw_name.decode("latin-1").lower()
            value = raw_value.decode("latin-1")
            if name == "authorization":
                authorization = value
            elif name == "cookie":
                cookie_header = value

        if authorization is not None:
            scheme, separator, token = authorization.partition(" ")
            if (
                separator
                and scheme.lower() == "bearer"
                and hmac.compare_digest(token, self._known_key)
            ):
                return AuthPrincipal(kind="bearer", value=token)

        session = _session_from_cookie(cookie_header, self.cookie_name)
        if session is not None and self._verify_browser_session(session):
            return AuthPrincipal(kind="session", value=session)
        return None

    def verify_api_key(self, candidate: str) -> bool:
        self._refresh_credentials()
        return hmac.compare_digest(candidate, self._known_key)

    def create_browser_session(self) -> str:
        nonce = secrets.token_urlsafe(32)
        return f"{nonce}.{self._browser_session_signature(nonce)}"

    def rotate(self) -> RotatedCredential:
        replacement = self.credentials.rotate()
        self._known_key = replacement
        return RotatedCredential(
            api_key=replacement,
            browser_session=self.create_browser_session(),
        )

    def _refresh_credentials(self) -> None:
        current = self.credentials.current_key()
        if hmac.compare_digest(current, self._known_key):
            return
        self._known_key = current

    def _verify_browser_session(self, session: str) -> bool:
        nonce, separator, signature = session.partition(".")
        if not separator or not nonce or not signature:
            return False
        expected = self._browser_session_signature(nonce)
        return hmac.compare_digest(signature, expected)

    def _browser_session_signature(self, nonce: str) -> str:
        digest = hmac.new(
            self._known_key.encode(),
            f"fish-memory-browser-session:v1:{nonce}".encode(),
            hashlib.sha256,
        ).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _session_from_cookie(
    cookie_header: str | None,
    cookie_name: str,
) -> str | None:
    if cookie_header is None:
        return None
    cookies = SimpleCookie()
    try:
        cookies.load(cookie_header)
    except Exception:
        return None
    morsel = cookies.get(cookie_name)
    return morsel.value if morsel is not None else None
