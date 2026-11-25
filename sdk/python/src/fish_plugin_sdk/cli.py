from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

from .manifest import Bundle, Manifest, target


def app_file() -> Path:
    if os.environ.get("FISH_PLUGIN_APP_FILE"):
        return Path(os.environ["FISH_PLUGIN_APP_FILE"])
    if sys.platform == "darwin":
        base = Path.home() / "Library/Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ["APPDATA"])
    else:
        base = Path.home() / ".config"
    candidates = [
        base / name / "plugin-dev.json"
        for name in ("fish-memory-desktop", "Fish Memory")
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def app_request(discovery: dict, method: str, data: dict | None = None) -> dict:
    request = urllib.request.Request(
        discovery["url"] + "/internal/extensions/dev/" + method,
        data=json.dumps(data or {}).encode(),
        method="POST",
        headers={
            "Authorization": "Bearer " + discovery["token"],
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def init(root: Path) -> None:
    if root.exists() and any(root.iterdir()):
        raise ValueError("Choose an empty directory")
    root.mkdir(parents=True, exist_ok=True)
    name = root.name.replace("-", "_").replace(" ", "_").lower()
    manifest = Manifest(
        id=name, name=name, version="0.1.0", entrypoint=f"{name}:MyPlugin"
    )
    (root / "fish-plugin.json").write_text(manifest.model_dump_json(indent=2))
    (root / "pyproject.toml").write_text(f'''[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["fish-plugin-sdk==0.1.1"]

[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{name}"]
''')
    module = root / "src" / name
    module.mkdir(parents=True)
    (module / "__init__.py").write_text("""import asyncio
from pydantic import BaseModel, ConfigDict, Field
from fish_plugin_sdk import Event, Pet, Plugin

class Notice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(max_length=500, description="这次发生的事情")

NOTICE = Event("example.notice", Notice,
               "回应这个事情，结合自己的性格表达感受。", speak="auto")

class MyPlugin(Plugin):
    events = (NOTICE,)

    async def run(self, pet: Pet) -> None:
        # Use App's event tester; connect your real event source here.
        # await pet.push(NOTICE, Notice(text="窗外下雨了"))
        await asyncio.Event().wait()
""")
    (root / "examples.json").write_text(
        json.dumps(
            {"example.notice": {"text": "窗外下雨了"}}, ensure_ascii=False, indent=2
        )
    )
    (root / "README.md").write_text(
        "在 Fish App 插件面板启用开发模式、选择宠物和会话，运行 `fish-plugin dev .`。\n"
        "修改 src 下的插件，在本体事件测试中输入 examples.json 的数据。\n"
        "`fish-plugin check .` 检查声明；`fish-plugin build .` 构建本机产物。\n"
    )
    (root / ".gitignore").write_text(".venv/\n__pycache__/\ndist/\n*.egg-info/\n")
    print(f"Created {root}; enable development in the App, then fish-plugin dev {root}")


def check(root: Path) -> dict:
    manifest = Manifest.read(root)
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fish_plugin_sdk.runner",
            manifest.entrypoint,
            "--describe",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode:
        raise ValueError(
            "Plugin import failed. Install declared dependencies first.\n"
            + result.stderr[-3000:]
        )
    description = json.loads(result.stdout)
    events = description.get("events", [])
    if not 0 < len(events) <= 32 or len({e["type"] for e in events}) != len(events):
        raise ValueError("Declare 1 to 32 unique event types")
    for event in events:
        if (
            event["speak"] not in (False, True, "auto")
            or event["schema"].get("type") != "object"
        ):
            raise ValueError("Invalid event definition: " + event["type"])
    return {
        "manifest": manifest.model_dump(),
        "declarations": description,
        "target": target(),
        "checks": {
            "manifest": "passed",
            "import": "passed",
            "declarations": "passed",
            "live_app": "not_run",
            "other_platforms": "not_run",
        },
    }


def sign_wheel_binaries(wheel: Path, identity: str) -> None:
    """Code-sign the native extensions inside a wheel and keep its RECORD valid.

    Apple notarization inspects binaries nested in archives, so a plugin shipped
    inside a notarized app needs its wheels' extensions signed too.
    """
    with zipfile.ZipFile(wheel) as source:
        infos = source.infolist()
        native = [i.filename for i in infos if i.filename.endswith((".so", ".dylib"))]
        if not native:
            return
        data = {i.filename: source.read(i) for i in infos}
    with tempfile.TemporaryDirectory(prefix="fish-sign-") as temp:
        for name in native:
            binary = Path(temp) / name
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(data[name])
            subprocess.run(
                [
                    "codesign",
                    "--force",
                    "--sign",
                    identity,
                    "--timestamp",
                    "--options",
                    "runtime",
                    str(binary),
                ],
                check=True,
            )
            data[name] = binary.read_bytes()
    record = next(n for n in data if n.endswith(".dist-info/RECORD"))
    rows = list(csv.reader(io.StringIO(data[record].decode())))
    for row in rows:
        if row and row[0] in native:
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(data[row[0]]).digest()
            ).rstrip(b"=")
            row[1:3] = ["sha256=" + digest.decode(), str(len(data[row[0]]))]
    text = io.StringIO()
    csv.writer(text, lineterminator="\n").writerows(rows)
    data[record] = text.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as output:
        for info in infos:
            output.writestr(info, data[info.filename], zipfile.ZIP_DEFLATED)


def build(
    root: Path, sdk_wheel: Path | None = None, codesign_identity: str | None = None
) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise ValueError("Install fish-plugin-sdk[tools] to use build")
    report = check(root)
    manifest = Manifest.model_validate(report["manifest"])
    with tempfile.TemporaryDirectory(prefix="fish-build-") as temp:
        work = Path(temp)
        wheels = work / "wheels"
        wheels.mkdir()
        subprocess.run(
            [uv, "build", "--wheel", str(root), "--out-dir", str(wheels)], check=True
        )
        own = next(wheels.glob("*.whl"))
        if sdk_wheel:
            shutil.copy2(sdk_wheel, wheels)
        # Release installation uses these exact wheels offline, never a new resolution.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "download",
                "--only-binary=:all:",
                "--find-links",
                str(wheels),
                "--dest",
                str(wheels),
                str(own),
            ],
            check=True,
        )
        if codesign_identity:
            for wheel in wheels.glob("*.whl"):
                sign_wheel_binaries(wheel, codesign_identity)
        (work / "check-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        files = {
            str(p.relative_to(work)).replace(os.sep, "/"): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in work.rglob("*")
            if p.is_file()
        }
        bundle = Bundle(manifest=manifest, target=target(), files=files)
        destination = root / "dist"
        destination.mkdir(exist_ok=True)
        archive = (
            destination / f"{manifest.id}-{manifest.version}-{target()}.fish-plugin"
        )
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
            output.writestr("bundle.json", bundle.model_dump_json(indent=2))
            for path in files:
                output.write(work / path, path)
        return archive


def dev(root: Path, discovery_path: Path) -> None:
    if not discovery_path.exists():
        raise ValueError(
            f"Enable development in Fish App first. Discovery file: {discovery_path}"
        )
    discovery = json.loads(discovery_path.read_text())
    result = app_request(discovery, "load", {"path": str(root)})
    identifier = result["id"]
    print(f"Connected to real App. Instance {identifier}. Ctrl+C stops this plugin.")

    def fingerprint() -> tuple:
        files = [*root.glob("*.toml"), *root.glob("*.json"), *root.glob("src/**/*.py")]
        return tuple(
            sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        )

    previous = fingerprint()
    try:
        while True:
            time.sleep(0.75)
            current = fingerprint()
            if current != previous:
                previous = current
                app_request(discovery, "restart", {"id": identifier})
                print("Plugin reloaded; App and voice runtime remain running.")
    except KeyboardInterrupt:
        app_request(discovery, "stop", {"id": identifier})


def main() -> None:
    parser = argparse.ArgumentParser(prog="fish-plugin")
    parser.add_argument("command", choices=["init", "dev", "check", "build"])
    parser.add_argument("path", nargs="?", default=".")
    parser.add_argument("--app-file", type=Path, default=None)
    parser.add_argument(
        "--sdk-wheel",
        type=Path,
        default=None,
        help="Unpublished SDK wheel for first-party development",
    )
    parser.add_argument(
        "--codesign-identity",
        default=None,
        help="macOS signing identity for native extensions in bundled wheels",
    )
    args = parser.parse_args()
    root = Path(args.path).resolve()
    try:
        if args.command == "init":
            init(root)
        elif args.command == "check":
            print(json.dumps(check(root), ensure_ascii=False, indent=2))
        elif args.command == "build":
            print(build(root, args.sdk_wheel, args.codesign_identity))
        else:
            dev(root, args.app_file or app_file())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
