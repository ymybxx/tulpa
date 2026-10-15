"""Bounded local IPC for reusing Companion's authoritative SenseVoice model."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import hmac
import json
import logging
import os
import secrets
import stat
import struct
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from fish_memory.companion.asr import AsrUpdate, validated_token_alignment
from fish_memory.runtime_platform import private_file_permissions

logger = logging.getLogger(__name__)

SHARED_ASR_SOCKET_ENV = "FISH_MEMORY_SHARED_ASR_SOCKET"
MAX_PCM_BYTES = 320_000
_MAX_RESPONSE_BYTES = 16_384
_HEADER = struct.Struct("!4sI")
_MAGIC = b"FMA1"
_READ_TIMEOUT_SECONDS = 2.0
_BUSY_WAIT_SECONDS = 5.0
_BUSY_RETRY_SECONDS = 0.05
_QOS_CLASS_UTILITY = 0x11
_AUTH_MAGIC = b"FMA2"
_TOKEN_BYTES = 32
_ENDPOINT_MAX_BYTES = 512
Transport = Literal["unix", "tcp"]


class SharedSnapshotRecognizer(Protocol):
    @property
    def model(self) -> str: ...

    def try_recognize_snapshot_update(self, pcm: bytes) -> AsrUpdate | None: ...

    def buffered_pcm_bytes(self) -> int: ...


@dataclass(frozen=True, slots=True)
class SharedAsrResponse:
    status: str
    text: str = ""
    event: str = ""
    tokens: tuple[str, ...] = ()
    timestamps: tuple[float, ...] = ()


def configured_shared_asr_socket() -> Path | None:
    value = os.getenv(SHARED_ASR_SOCKET_ENV, "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if sys.platform != "win32" and len(os.fsencode(path)) > 100:
        raise ValueError("shared ASR socket path is too long")
    return path


class SharedAsrClient:
    def __init__(self, path: Path, *, transport: Transport | None = None) -> None:
        self._path = path
        self._transport = transport or ("tcp" if sys.platform == "win32" else "unix")

    async def recognize(self, pcm: bytes) -> SharedAsrResponse:
        if type(pcm) is not bytes:
            raise TypeError("shared ASR PCM must be immutable bytes")
        if not pcm or len(pcm) > MAX_PCM_BYTES or len(pcm) % 2:
            raise ValueError("shared ASR PCM length is invalid")
        token = None
        if self._transport == "tcp":
            port, token = _read_endpoint(self._path)
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port), _READ_TIMEOUT_SECONDS
            )
        else:
            reader, writer = await asyncio.open_unix_connection(str(self._path))
        try:
            if token is not None:
                await _authenticate_server(reader, writer, token)
            writer.write(_HEADER.pack(_MAGIC, len(pcm)))
            writer.write(pcm)
            await writer.drain()
            size_raw = await asyncio.wait_for(
                reader.readexactly(4),
                timeout=_READ_TIMEOUT_SECONDS + _BUSY_WAIT_SECONDS,
            )
            size = struct.unpack("!I", size_raw)[0]
            if not 0 < size <= _MAX_RESPONSE_BYTES:
                raise ValueError("shared ASR response length is invalid")
            payload = await asyncio.wait_for(
                reader.readexactly(size), timeout=_READ_TIMEOUT_SECONDS
            )
            return _parse_response(payload)
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()


class SharedAsrServer:
    def __init__(self, path: Path, *, transport: Transport | None = None) -> None:
        self._path = path
        self._transport = transport or ("tcp" if sys.platform == "win32" else "unix")
        self._token: bytes | None = None
        self._recognizer: SharedSnapshotRecognizer | None = None
        self._server: asyncio.AbstractServer | None = None
        self._request_lock = asyncio.Lock()
        self._writers: set[asyncio.StreamWriter] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="environment-shared-asr",
            initializer=_set_utility_qos,
        )
        self._socket_identity: tuple[int, int] | None = None
        self._closed = False

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("shared ASR server is closed")
        if self._server is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._transport == "tcp":
            await self._start_tcp()
            return
        try:
            existing = self._path.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISSOCK(existing.st_mode):
                raise RuntimeError("shared ASR path is occupied by a non-socket")
            self._path.unlink()
        server = await asyncio.start_unix_server(self._handle, path=str(self._path))
        self._server = server
        try:
            os.chmod(self._path, 0o600)
            socket_stat = self._path.stat()
            self._socket_identity = (socket_stat.st_dev, socket_stat.st_ino)
        except BaseException:
            server.close()
            await server.wait_closed()
            self._server = None
            self._path.unlink(missing_ok=True)
            raise

    async def _start_tcp(self) -> None:
        if self._path.exists() or self._path.is_symlink():
            # Never overwrite an unrelated file, or steal a running owner's
            # endpoint. A stale valid endpoint can be replaced after a crash.
            port, _ = _read_endpoint(self._path)
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection("127.0.0.1", port), 0.5
                )
            except (OSError, TimeoutError):
                pass
            else:
                writer.close()
                await writer.wait_closed()
                raise RuntimeError("shared ASR endpoint is already in use")
        self._token = secrets.token_bytes(_TOKEN_BYTES)
        server = await asyncio.start_server(self._handle, host="127.0.0.1", port=0)
        self._server = server
        port = server.sockets[0].getsockname()[1]
        publication = asyncio.create_task(
            asyncio.to_thread(_write_endpoint, self._path, port, self._token)
        )
        try:
            # File/ACL work cannot be cancelled once its worker thread starts.
            # Wait for publication before removing our endpoint on shutdown.
            await asyncio.shield(publication)
        except BaseException:
            server.close()
            await server.wait_closed()
            await asyncio.gather(publication, return_exceptions=True)
            self._server = None
            self._unlink_owned_socket()
            raise

    def bind(self, recognizer: SharedSnapshotRecognizer | None) -> None:
        self._recognizer = recognizer

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._recognizer = None
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        writers = tuple(self._writers)
        for writer in writers:
            writer.close()
        if writers:
            await asyncio.gather(
                *(writer.wait_closed() for writer in writers),
                return_exceptions=True,
            )
        await asyncio.to_thread(
            self._executor.shutdown,
            wait=True,
            cancel_futures=True,
        )
        self._unlink_owned_socket()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._writers.add(writer)
        try:
            if self._token is not None:
                await _authenticate_client(reader, writer, self._token)
            header = await asyncio.wait_for(
                reader.readexactly(_HEADER.size), timeout=_READ_TIMEOUT_SECONDS
            )
            magic, size = _HEADER.unpack(header)
            if magic != _MAGIC or not 0 < size <= MAX_PCM_BYTES or size % 2:
                await _write_response(writer, SharedAsrResponse("invalid"))
                return
            pcm = await asyncio.wait_for(
                reader.readexactly(size), timeout=_READ_TIMEOUT_SECONDS
            )
            response = await self._recognize_connected(reader, pcm)
            if response is not None:
                await _write_response(writer, response)
        except (asyncio.IncompleteReadError, TimeoutError):
            with suppress(Exception):
                await _write_response(writer, SharedAsrResponse("invalid"))
        except ConnectionError:
            # A retired window closes its client socket to cancel work.
            pass
        except Exception:
            logger.exception("Shared environment ASR request failed")
            with suppress(Exception):
                await _write_response(writer, SharedAsrResponse("error"))
        finally:
            self._writers.discard(writer)
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    async def _recognize_connected(
        self, reader: asyncio.StreamReader, pcm: bytes
    ) -> SharedAsrResponse | None:
        """Retire queued work and busy retries as soon as its client leaves.

        Native inference already in progress cannot be interrupted, but its
        result is discarded and no further inference is queued for this request.
        A request contains exactly one PCM payload, so any following byte is an
        invalid request rather than another audio segment.
        """

        if reader.at_eof():
            return None

        async def recognize() -> SharedAsrResponse | None:
            async with self._request_lock:
                if reader.at_eof():
                    return None
                return await self._recognize(pcm)

        recognition = asyncio.create_task(recognize())
        disconnected = asyncio.create_task(reader.read(1))
        try:
            completed, _ = await asyncio.wait(
                (recognition, disconnected), return_when=asyncio.FIRST_COMPLETED
            )
            if disconnected in completed:
                return SharedAsrResponse("invalid") if disconnected.result() else None
            return recognition.result()
        finally:
            for task in (recognition, disconnected):
                if not task.done():
                    task.cancel()
            await asyncio.gather(recognition, disconnected, return_exceptions=True)

    async def _recognize(self, pcm: bytes) -> SharedAsrResponse:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _BUSY_WAIT_SECONDS
        while not self._closed:
            recognizer = self._recognizer
            if recognizer is None:
                return SharedAsrResponse("unavailable")
            # Draft/final user speech owns the recognizer. In particular, do
            # not start a 10-second media decode while an utterance is still
            # accumulating but has not reached its authoritative decode yet.
            if recognizer.buffered_pcm_bytes() > 0:
                result = None
            else:
                result = await loop.run_in_executor(
                    self._executor,
                    recognizer.try_recognize_snapshot_update,
                    pcm,
                )
            if result is not None:
                tokens, timestamps = validated_token_alignment(
                    getattr(result, "tokens", ()), getattr(result, "timestamps", ())
                )
                # A token sequence describes the entire decode. Truncating
                # text without a proven token-to-text mapping invalidates it.
                if len(result.text) > 4000:
                    tokens, timestamps = (), ()
                return SharedAsrResponse(
                    "ok",
                    text=result.text[:4000],
                    event=result.event[:100],
                    tokens=tokens,
                    timestamps=timestamps,
                )
            if loop.time() >= deadline:
                return SharedAsrResponse("busy")
            await asyncio.sleep(_BUSY_RETRY_SECONDS)
        return SharedAsrResponse("unavailable")

    def _unlink_owned_socket(self) -> None:
        if self._transport == "tcp":
            try:
                _, token = _read_endpoint(self._path)
                if self._token is not None and hmac.compare_digest(token, self._token):
                    self._path.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass
            self._token = None
            return
        identity = self._socket_identity
        self._socket_identity = None
        if identity is None:
            return
        try:
            current = self._path.stat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) == identity:
            self._path.unlink()


def _write_endpoint(path: Path, port: int, token: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".asr-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            private_file_permissions(temporary, output.fileno())
            json.dump({"version": 1, "port": port, "token": token.hex()}, output)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_endpoint(path: Path) -> tuple[int, bytes]:
    """Read only bounded local metadata; the hostname is never file-controlled."""
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("shared ASR endpoint must be a regular file")
    with path.open("rb") as source:
        raw = source.read(_ENDPOINT_MAX_BYTES + 1)
    if len(raw) > _ENDPOINT_MAX_BYTES:
        raise ValueError("shared ASR endpoint is too large")
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {"version", "port", "token"}:
            raise ValueError
        port, token = value["port"], value["token"]
        if (
            type(value["version"]) is not int
            or value["version"] != 1
            or type(port) is not int
            or not 0 < port < 65536
            or not isinstance(token, str)
            or len(token) != _TOKEN_BYTES * 2
        ):
            raise ValueError
        secret = bytes.fromhex(token)
        if len(secret) != _TOKEN_BYTES:
            raise ValueError
    except (ValueError, UnicodeDecodeError, TypeError) as error:
        raise ValueError("shared ASR endpoint is invalid") from error
    return port, secret


def _proof(token: bytes, role: bytes, nonce: bytes) -> bytes:
    return hmac.digest(token, role + nonce, hashlib.sha256)


async def _authenticate_server(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, token: bytes
) -> None:
    # Mutual challenge/response keeps PCM and the secret away from a different
    # process that happened to reuse an old endpoint's port after a crash.
    nonce = secrets.token_bytes(_TOKEN_BYTES)
    writer.write(_AUTH_MAGIC + nonce)
    await writer.drain()
    response = await asyncio.wait_for(reader.readexactly(64), _READ_TIMEOUT_SECONDS)
    both = nonce + response[:32]
    if not hmac.compare_digest(response[32:], _proof(token, b"server", both)):
        raise ConnectionError("shared ASR server authentication failed")
    writer.write(_proof(token, b"client", both))
    await writer.drain()


async def _authenticate_client(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, token: bytes
) -> None:
    request = await asyncio.wait_for(reader.readexactly(36), _READ_TIMEOUT_SECONDS)
    if request[:4] != _AUTH_MAGIC:
        raise ConnectionError("shared ASR client authentication failed")
    nonce = secrets.token_bytes(_TOKEN_BYTES)
    both = request[4:] + nonce
    writer.write(nonce + _proof(token, b"server", both))
    await writer.drain()
    proof = await asyncio.wait_for(reader.readexactly(32), _READ_TIMEOUT_SECONDS)
    if not hmac.compare_digest(proof, _proof(token, b"client", both)):
        raise ConnectionError("shared ASR client authentication failed")


async def _write_response(
    writer: asyncio.StreamWriter, response: SharedAsrResponse
) -> None:
    payload = _encode_response(response)
    writer.write(struct.pack("!I", len(payload)))
    writer.write(payload)
    await writer.drain()


def _encode_response(response: SharedAsrResponse) -> bytes:
    text = response.text[:4000]
    value: dict[str, object] = {
        "status": response.status,
        "text": text,
        "event": response.event[:100],
    }
    tokens, timestamps = validated_token_alignment(response.tokens, response.timestamps)
    if tokens and text == response.text and response.status == "ok":
        value.update(tokens=tokens, timestamps=timestamps)

    def encode() -> bytes:
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()

    payload = encode()
    if len(payload) <= _MAX_RESPONSE_BYTES:
        return payload
    # Text takes precedence over optional alignment, including for an older
    # peer whose text-only response cap is still 16 KiB.
    value.pop("tokens", None)
    value.pop("timestamps", None)
    payload = encode()
    if len(payload) <= _MAX_RESPONSE_BYTES:
        return payload
    # Escaping controls and four-byte Unicode can overflow the byte cap even
    # below the character cap. Never retain alignment for this shortened text.
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        value["text"] = text[:middle]
        if len(encode()) <= _MAX_RESPONSE_BYTES:
            low = middle
        else:
            high = middle - 1
    value["text"] = text[:low]
    return encode()


def _parse_response(payload: bytes) -> SharedAsrResponse:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("shared ASR response is not JSON") from error
    required = {"status", "text", "event"}
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or not set(value) <= required | {"tokens", "timestamps"}
    ):
        raise ValueError("shared ASR response has invalid fields")
    status, text, event = value["status"], value["text"], value["event"]
    if not isinstance(status, str) or status not in {
        "ok",
        "busy",
        "unavailable",
        "invalid",
        "error",
    }:
        raise ValueError("shared ASR response status is invalid")
    if not isinstance(text, str) or not isinstance(event, str):
        raise ValueError("shared ASR response values are invalid")
    if len(text) > 4000 or len(event) > 100:
        raise ValueError("shared ASR response text is too long")
    tokens, timestamps = validated_token_alignment(
        value.get("tokens", ()), value.get("timestamps", ())
    )
    if status != "ok":
        tokens, timestamps = (), ()
    return SharedAsrResponse(status, text, event, tokens, timestamps)


def _set_utility_qos() -> None:
    if sys.platform != "darwin":
        return
    try:
        function = ctypes.CDLL(None).pthread_set_qos_class_self_np
        function.argtypes = [ctypes.c_uint, ctypes.c_int]
        function.restype = ctypes.c_int
        function(_QOS_CLASS_UTILITY, 0)
    except (AttributeError, OSError):
        logger.debug("Unable to lower shared ASR worker QoS", exc_info=True)
