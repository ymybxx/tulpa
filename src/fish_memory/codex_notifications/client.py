"""Small read-only client for the official App Server JSON-RPC protocol.

No authentication tokens, resume, turn execution or approval APIs are exposed.
The existing installed Codex runtime owns access to its own persisted history.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

READ_METHODS = frozenset({"thread/list", "thread/read", "thread/turns/list"})


def find_codex() -> str | None:
    candidates = [
        Path("/Applications/ChatGPT.app/Contents/Resources/codex"),
        Path("/Applications/Codex.app/Contents/Resources/codex"),
        Path.home() / ".local/bin/codex",
        Path.home() / ".npm-global/bin/codex",
    ]
    located = shutil.which("codex")
    if located:
        candidates.append(Path(located))
    return next(
        (str(p) for p in candidates if p.is_file() and os.access(p, os.X_OK)), None
    )


class CodexReadClient:
    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable
        self.process: asyncio.subprocess.Process | None = None
        self._sequence = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return
        executable = self.executable or find_codex()
        if not executable:
            raise RuntimeError("未找到本机 Codex 运行时，请先安装或打开 Codex Desktop")
        allowed = {
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "SHELL",
            "TMPDIR",
            "TEMP",
            "TMP",
            "LANG",
            "SYSTEMROOT",
            "SystemRoot",
            "APPDATA",
            "LOCALAPPDATA",
            "USERPROFILE",
            "CODEX_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_CACHE_HOME",
        }
        env = {
            k: v for k, v in os.environ.items() if k in allowed or k.startswith("LC_")
        }
        self.process = await asyncio.create_subprocess_exec(
            executable,
            "app-server",
            "--stdio",
            cwd=Path.home(),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=8 * 1024 * 1024,
        )
        try:
            await self._request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "fish_completion_observer",
                        "version": "0.1.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            )
            assert self.process.stdin is not None
            self.process.stdin.write(b'{"method":"initialized","params":{}}\n')
            await self.process.stdin.drain()
        except BaseException:
            await self.close()
            raise

    async def read(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method not in READ_METHODS:
            raise ValueError("completion observer only permits read methods")
        async with self._lock:
            await self.start()
            try:
                return await self._request(method, params)
            except (TimeoutError, EOFError, BrokenPipeError):
                await self.close()
                raise

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert self.process is not None and self.process.stdin and self.process.stdout
        self._sequence += 1
        request_id = self._sequence
        self.process.stdin.write(
            (
                json.dumps({"id": request_id, "method": method, "params": params})
                + "\n"
            ).encode()
        )
        await self.process.stdin.drain()
        async with asyncio.timeout(15):
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    raise EOFError("Codex 只读连接已结束")
                value = json.loads(line)
                if value.get("id") != request_id or "method" in value:
                    continue
                if "error" in value:
                    raise RuntimeError(
                        f"Codex read API error: {value['error'].get('code')}"
                    )
                result = value.get("result")
                if not isinstance(result, dict):
                    raise ValueError("invalid Codex response")
                return result

    async def close(self) -> None:
        process, self.process = self.process, None
        if process is None or process.returncode is not None:
            return
        if process.stdin:
            process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 3)
        except TimeoutError:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 2)
            except TimeoutError:
                process.kill()
                await process.wait()
