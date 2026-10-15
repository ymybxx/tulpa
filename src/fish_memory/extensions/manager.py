"""Local plugin lifecycle. This module is never imported into a plugin process."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import signal
import sys
import tomllib
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fish_plugin_sdk import Activity
from fish_plugin_sdk.manifest import Manifest, extract_bundle, target
from fish_plugin_sdk.review import ReviewProfile
from jsonschema import Draft202012Validator

from fish_memory.extensions.models import (
    CreateInstance,
    EventDefinition,
    InputDefinition,
)
from fish_memory.runtime_platform import private_file_permissions


@dataclass
class InstalledPlugin:
    id: UUID
    manifest: Manifest
    path: Path
    python: Path
    development: bool
    description: dict[str, Any]
    fish_id: UUID | None = None
    session_id: UUID | None = None
    config: dict[str, Any] = field(default_factory=dict, repr=False)
    process: asyncio.subprocess.Process | None = field(default=None, repr=False)
    tasks: list[asyncio.Task[Any]] = field(default_factory=list, repr=False)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=100))
    status: str = "stopped"
    redactions: list[str] = field(default_factory=list, repr=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "manifest": self.manifest.model_dump(),
            "development": self.development,
            "path": str(self.path),
            "description": self.description,
            "status": self.status,
            "fish_id": str(self.fish_id) if self.fish_id else None,
            "session_id": str(self.session_id) if self.session_id else None,
            "logs": list(self.logs),
        }


class PluginManager:
    def __init__(
        self,
        service: Any,
        sessions: Any,
        root: Path,
        assets: Path | None = None,
        reviews: Any = None,
    ):
        self.service, self.sessions, self.root, self.assets = (
            service,
            sessions,
            root,
            assets,
        )
        self.items: dict[UUID, InstalledPlugin] = {}
        self.reviews = reviews
        self._mutation = asyncio.Lock()
        self._dev_token: str | None = None
        self._dev_ids: set[UUID] = set()
        self._closed = False
        self._restore()

    def _restore(self) -> None:
        index = self.root / "installed.json"
        if not index.exists():
            return
        for row in json.loads(index.read_text(encoding="utf-8")):
            item = InstalledPlugin(
                UUID(row["id"]),
                Manifest.model_validate(row["manifest"]),
                Path(row["path"]),
                Path(row["python"]),
                row["development"],
                row["description"],
            )
            self.items[item.id] = item

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        records = [{**i.public(), "python": str(i.python)} for i in self.items.values()]
        for row in records:
            row.pop("logs", None)
            row.pop("fish_id", None)
            row.pop("session_id", None)
        path = self.root / "installed.json.tmp"
        path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        path.replace(self.root / "installed.json")

    def _tools(self) -> tuple[str, str]:
        if self.assets is not None:
            uv = self.assets / ("uv.exe" if os.name == "nt" else "uv")
            python = (
                self.assets
                / "python"
                / ("python.exe" if os.name == "nt" else "bin/python3")
            )
            if not uv.is_file() or not python.is_file():
                raise ValueError("插件运行时资源不完整，请重新安装应用")
            return str(uv), str(python)
        uv = shutil.which("uv")
        if uv is None or getattr(sys, "frozen", False):
            raise ValueError("插件运行时不可用")
        return uv, sys.executable

    def _sdk(self) -> str:
        if self.assets is not None:
            wheel = next(self.assets.glob("fish_plugin_sdk-*.whl"), None)
            if wheel is None:
                raise ValueError("插件 SDK 安装资源缺失")
            return str(wheel)
        return str(Path(__file__).resolve().parents[3] / "sdk/python")

    @staticmethod
    def _environment(data: Path, source: Path | None = None) -> dict[str, str]:
        allowed = {
            "HOME",
            "USERPROFILE",
            "SystemRoot",
            "WINDIR",
            "TEMP",
            "TMP",
            "TMPDIR",
            "LANG",
            "LC_ALL",
            "PATH",
        }
        allowed_upper = {key.upper() for key in allowed}
        env = {k: v for k, v in os.environ.items() if k.upper() in allowed_upper}
        env.update(
            PYTHONUNBUFFERED="1",
            PYTHONNOUSERSITE="1",
            PYTHONUTF8="1",
            FISH_PLUGIN_DATA_DIR=str(data),
        )
        if source is not None:
            env["PYTHONPATH"] = str(source / "src")
        return env

    async def _command(
        self,
        arguments: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        deadline_seconds: float = 180,
    ) -> str:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **({"start_new_session": True} if os.name != "nt" else {}),
        )
        try:
            async with asyncio.timeout(deadline_seconds):
                stdout, stderr = await process.communicate()
            if process.returncode:
                raise ValueError(
                    stderr.decode(errors="replace")[-4000:] or "Plugin command failed"
                )
            if len(stdout) > 65_536:
                raise ValueError("Plugin declaration output is too large")
            return stdout.decode()
        finally:
            if process.returncode is None:
                await self._kill(process)

    async def load(self, path: Path, *, development: bool) -> InstalledPlugin:
        async with self._mutation:
            if self._closed:
                raise ValueError("App is stopping")
            path = await asyncio.to_thread(
                lambda: path.expanduser().resolve(strict=True)
            )
            uv, base_python = self._tools()
            if development:
                manifest = Manifest.read(path)
                existing = next(
                    (
                        i
                        for i in self.items.values()
                        if i.path == path and i.development
                    ),
                    None,
                )
                if existing is not None and existing.process is not None:
                    raise ValueError("请先停止此插件，再重新加载依赖")
                digest = hashlib.sha256(
                    str(path).encode() + (path / "pyproject.toml").read_bytes()
                ).hexdigest()[:16]
                environment = self.root / "environments" / f"dev-{digest}"
            else:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                environment = self.root / "environments" / digest
                source = self.root / "packages" / digest
                bundle = await asyncio.to_thread(extract_bundle, path, source, target())
                manifest, path = bundle.manifest, source
                existing = next(
                    (
                        i
                        for i in self.items.values()
                        if i.manifest.id == manifest.id and not i.development
                    ),
                    None,
                )
            python = environment / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            marker = environment / ".fish-ready"
            if not marker.exists():
                environment.parent.mkdir(parents=True, exist_ok=True)
                await self._command(
                    [uv, "venv", "--python", base_python, str(environment), "--clear"]
                )
                if development:
                    await self._command(
                        [
                            uv,
                            "pip",
                            "install",
                            "--python",
                            str(python),
                            "--link-mode",
                            "copy",
                            self._sdk(),
                        ]
                    )
                    project = tomllib.loads(
                        (path / "pyproject.toml").read_text(encoding="utf-8")
                    )["project"]
                    dependencies = [
                        d
                        for d in project.get("dependencies", [])
                        if not d.lower().startswith("fish-plugin-sdk")
                    ]
                    if dependencies:
                        await self._command(
                            [
                                uv,
                                "pip",
                                "install",
                                "--python",
                                str(python),
                                "--link-mode",
                                "copy",
                                "--only-binary",
                                ":all:",
                                *dependencies,
                            ]
                        )
                else:
                    wheels = [str(w) for w in (path / "wheels").glob("*.whl")]
                    if not wheels:
                        raise ValueError("Plugin bundle contains no wheels")
                    await self._command(
                        [
                            uv,
                            "pip",
                            "install",
                            "--python",
                            str(python),
                            "--no-index",
                            "--no-deps",
                            "--link-mode",
                            "copy",
                            *wheels,
                        ]
                    )
                marker.touch()
            description = await self._describe(manifest, path, python, development)
            if existing is not None:
                # Validate the new environment before touching the running version.
                await self._stop(existing)
            item = InstalledPlugin(
                existing.id if existing else uuid4(),
                manifest,
                path,
                python,
                development,
                description,
            )
            if existing:
                item.fish_id, item.session_id, item.config = (
                    existing.fish_id,
                    existing.session_id,
                    existing.config,
                )
            self.items[item.id] = item
            self._save()
            return item

    async def _describe(
        self, manifest: Manifest, path: Path, python: Path, development: bool
    ) -> dict[str, Any]:
        raw = await self._command(
            [
                str(python),
                "-m",
                "fish_plugin_sdk.runner",
                manifest.entrypoint,
                "--describe",
            ],
            cwd=path,
            env=self._environment(self.root, path if development else None),
            deadline_seconds=15,
        )
        value = json.loads(raw)
        events = [EventDefinition.model_validate(e) for e in value["events"]]
        if not 0 < len(events) <= 32 or len({e.type for e in events}) != len(events):
            raise ValueError("插件需要声明 1 至 32 个不同事件")
        Draft202012Validator.check_schema(value["config_schema"])
        inputs = [InputDefinition.model_validate(v) for v in value.get("inputs", [])]
        if len(inputs) > 8 or len({v.type for v in inputs}) != len(inputs):
            raise ValueError("插件最多声明 8 个不同输入")
        if inputs:
            value["inputs"] = [v.model_dump(by_alias=True) for v in inputs]
        if value.get("activity") is not None:
            value["activity"] = Activity.model_validate(value["activity"]).model_dump()
        if value.get("review") is not None:
            value["review"] = ReviewProfile.model_validate(value["review"]).model_dump(
                mode="json"
            )
        return value

    async def start(
        self,
        identifier: UUID,
        config: dict[str, Any],
        url: str,
    ) -> InstalledPlugin:
        async with self._mutation:
            if self._closed:
                raise ValueError("App is stopping")
            fish_id, session_id = await self.service.current_session()
            item = self.items[identifier]
            await self._stop(item)
            await self.sessions.require_owned(fish_id, session_id)
            if item.development:
                item.description = await self._describe(
                    item.manifest, item.path, item.python, True
                )
            error = next(
                iter(
                    Draft202012Validator(item.description["config_schema"]).iter_errors(
                        config
                    )
                ),
                None,
            )
            if error:
                # Schema errors can contain secrets: expose only the field path.
                raise ValueError(
                    "插件配置不合法："
                    + ".".join(map(str, error.path))
                    + " ("
                    + str(error.validator)
                    + ")"
                )
            instance, token = await self.service.create(
                CreateInstance(
                    instance_id=identifier,
                    plugin_id=item.manifest.id,
                    fish_id=fish_id,
                    session_id=session_id,
                    autonomous=item.manifest.autonomous,
                )
            )
            item.fish_id, item.session_id, item.config = (
                instance.fish_id,
                instance.session_id,
                dict(config),
            )
            item.redactions = [
                token,
                *(
                    str(v)
                    for v in config.values()
                    if isinstance(v, str) and len(v) >= 8
                ),
            ]
            data = self.root / "data" / str(identifier)
            data.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                item.process = await asyncio.create_subprocess_exec(
                    str(item.python),
                    "-m",
                    "fish_plugin_sdk.runner",
                    item.manifest.entrypoint,
                    cwd=data,
                    env=self._environment(
                        data, item.path if item.development else None
                    ),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    **({"start_new_session": True} if os.name != "nt" else {}),
                )
                startup = {
                    "url": url.replace("http://", "ws://").replace("https://", "wss://")
                    + "/v1/extensions/ws?protocol=1",
                    "token": token,
                    "config": config,
                    "parent_pid": os.getpid(),
                }
                assert item.process.stdin is not None
                assert item.process.stdout is not None
                assert item.process.stderr is not None
                item.process.stdin.write((json.dumps(startup) + "\n").encode())
                await item.process.stdin.drain()
                item.process.stdin.close()
                item.status = "starting"
                item.tasks = [
                    asyncio.create_task(self._logs(item, item.process.stdout)),
                    asyncio.create_task(self._logs(item, item.process.stderr)),
                    asyncio.create_task(self._monitor(item, item.process)),
                ]
                if self.reviews is not None:
                    try:
                        await self.reviews.start(
                            instance,
                            item.manifest.version,
                            item.description.get("review"),
                        )
                    except Exception:
                        item.logs.append("运行评估暂不可用，插件继续运行。")
            except BaseException:
                await self.service.stop(identifier)
                raise
            self._save()
            return item

    async def _logs(self, item: InstalledPlugin, reader: asyncio.StreamReader) -> None:
        while line := await reader.readline():
            text = line.decode(errors="replace")[:2000].strip()
            for secret in item.redactions:
                text = text.replace(secret, "[redacted]")
            item.logs.append(text)

    async def _monitor(
        self, item: InstalledPlugin, process: asyncio.subprocess.Process
    ) -> None:
        code = await process.wait()
        if item.process is process:
            item.status = "stopped" if code == 0 else "failed"
            item.logs.append(f"进程已退出（{code}）")
            await self.service.stop(item.id)
            if self.reviews is not None:
                await self.reviews.stop(item.id)

    async def _stop(self, item: InstalledPlugin) -> None:
        if self.reviews is not None:
            await self.reviews.stop(item.id)
        await self.service.stop(item.id)
        process, item.process = item.process, None
        if process is not None:
            await self._kill(process)
        for task in item.tasks:
            task.cancel()
        await asyncio.gather(*item.tasks, return_exceptions=True)
        item.tasks.clear()
        item.status = "stopped"

    async def stop(self, identifier: UUID) -> None:
        async with self._mutation:
            await self._stop(self.items[identifier])

    @staticmethod
    async def _kill(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        if os.name == "nt":
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            async with asyncio.timeout(3):
                await process.wait()
        except TimeoutError:
            if os.name != "nt":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            await process.wait()

    def listing(self) -> dict[str, Any]:
        rows = []
        for item in self.items.values():
            row = item.public()
            instance = self.service.instances.get(item.id)
            if instance and instance.connected:
                row["status"] = "running"
                row["state"] = self.service.state(instance)
                row["fish_id"] = str(instance.fish_id)
                row["session_id"] = str(instance.session_id)
            if self.reviews is not None:
                row["review"] = self.reviews.state(item.id)
            rows.append(row)
        return {
            "plugins": rows,
            "development": self._dev_token is not None,
            "consumers": [
                {"fish_id": str(fish), "session_id": str(session)}
                for fish, session in self.service._consumers
            ],
        }

    async def development(self, enabled: bool, url: str) -> None:
        discovery = self.root.parent / "plugin-dev.json"
        self._dev_token = secrets.token_urlsafe(32) if enabled else None
        discovery.unlink(missing_ok=True)
        if enabled:
            discovery.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(discovery, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    private_file_permissions(discovery, stream.fileno())
                    json.dump({"url": url, "token": self._dev_token}, stream)
            except BaseException:
                self._dev_token = None
                discovery.unlink(missing_ok=True)
                raise
        else:
            for identifier in self._dev_ids:
                await self.stop(identifier)
            self._dev_ids.clear()

    def authenticate_dev(self, scope: dict[str, Any]) -> bool:
        if (
            not self._dev_token
            or scope.get("type") != "http"
            or scope.get("method") != "POST"
        ):
            return False
        if scope.get("path") not in {
            "/internal/extensions/dev/load",
            "/internal/extensions/dev/restart",
            "/internal/extensions/dev/stop",
        }:
            return False
        headers = dict(scope.get("headers", []))
        return b"origin" not in headers and secrets.compare_digest(
            headers.get(b"authorization", b""), ("Bearer " + self._dev_token).encode()
        )

    async def dev_command(
        self, command: str, params: dict[str, Any], url: str
    ) -> dict[str, Any]:
        if self._dev_token is None:
            raise ValueError("请先在 App 开启开发模式")
        if command == "load":
            item = await self.load(Path(params["path"]), development=True)
            self._dev_ids.add(item.id)
            return (await self.start(item.id, item.config, url)).public()
        identifier = UUID(params["id"])
        if identifier not in self._dev_ids:
            raise ValueError("此实例不属于当前开发连接")
        if command == "stop":
            await self.stop(identifier)
            return {"stopped": True}
        item = self.items[identifier]
        return (await self.start(item.id, item.config, url)).public()

    async def close(self) -> None:
        self._closed = True
        self._dev_token = None
        (self.root.parent / "plugin-dev.json").unlink(missing_ok=True)
        for item in self.items.values():
            await self._stop(item)
