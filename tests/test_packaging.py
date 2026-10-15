import json
import runpy
import sys
import tomllib
from functools import partial
from pathlib import Path
from types import ModuleType
from urllib.parse import urlparse

from packaging.requirements import Requirement

PROJECT_ROOT = Path(__file__).parents[1]


def test_native_audio_dependencies_are_split_by_runtime_role() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    base = "\n".join(project["dependencies"])
    voice = "\n".join(project["optional-dependencies"]["voice"])

    for dependency in (
        "aec-audio-processing",
        "sentencepiece",
        "sounddevice",
        "webrtcvad-wheels",
    ):
        assert dependency not in base
        assert dependency in voice

    for dependency in ("sherpa-onnx", "sherpa-onnx-core"):
        assert dependency not in base
        assert dependency in voice

    assert "websockets" in base
    assert "pypinyin" not in base
    assert "pypinyin" in voice


def test_livekit_vad_dependencies_are_pinned_and_desktop_scoped() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    dependencies = {
        requirement.name: requirement
        for value in project["optional-dependencies"]["voice"]
        if (requirement := Requirement(value)).name
        in {"livekit-agents", "livekit-plugins-silero", "av"}
    }

    assert project["requires-python"] == ">=3.11"
    assert set(dependencies) == {"livekit-agents", "livekit-plugins-silero", "av"}
    assert str(dependencies["livekit-agents"].specifier) == "==1.7.1"
    assert str(dependencies["livekit-plugins-silero"].specifier) == "==1.7.1"
    assert str(dependencies["av"].specifier) == "==14.2.0"
    for requirement in dependencies.values():
        assert requirement.marker is not None
        environment = {
            "sys_platform": "darwin",
            "platform_machine": "arm64",
            "python_version": "3.11",
        }
        assert requirement.marker.evaluate(environment)
        assert not requirement.marker.evaluate({**environment, "sys_platform": "linux"})
        assert not requirement.marker.evaluate(
            {**environment, "platform_machine": "x86_64"}
        )
        assert not requirement.marker.evaluate(
            {**environment, "python_version": "3.15"}
        )


def test_livekit_vad_hook_collects_only_required_plugin_assets(monkeypatch) -> None:
    calls = []
    hooks = ModuleType("PyInstaller.utils.hooks")

    def collect(kind, package, **kwargs):
        calls.append((kind, package, kwargs))
        return [(f"source/{package}/{kind}", f"destination/{package}")]

    monkeypatch.setattr(
        hooks, "copy_metadata", partial(collect, "metadata"), raising=False
    )
    monkeypatch.setattr(
        hooks, "collect_data_files", partial(collect, "data"), raising=False
    )
    monkeypatch.setattr(
        hooks, "collect_dynamic_libs", partial(collect, "binary"), raising=False
    )
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    result = runpy.run_path(
        str(PROJECT_ROOT / "desktop/pet/packaging/hooks/hook-livekit.plugins.silero.py")
    )

    assert ("metadata", "livekit-plugins-silero", {"recursive": True}) in calls
    assert (
        "data",
        "livekit.plugins.silero",
        {"includes": ["resources/silero_vad.onnx"]},
    ) in calls
    assert {package for kind, package, _ in calls if kind == "binary"} == {
        "livekit.rtc",
        "livekit.local_inference",
        "av",
    }
    assert set(result["hiddenimports"]) == {
        "livekit.plugins.silero.resources",
        "livekit.rtc.resources",
        "livekit.local_inference._native",
    }
    assert not any(package == "livekit.plugins" for _, package, _ in calls)


def test_desktop_runtime_includes_local_vad_hook_and_license_notices() -> None:
    packaging_root = PROJECT_ROOT / "desktop/pet/packaging"
    spec = (packaging_root / "fish-memory-runtime.spec").read_text(encoding="utf-8")
    assert '"livekit.plugins.silero"' in spec
    assert '"licenses/livekit-vad"' in spec
    assert 'collect_all("livekit' not in spec
    model_license = (
        packaging_root / "licenses/livekit-vad/Silero-VAD-MIT-LICENSE.txt"
    ).read_text(encoding="utf-8")
    assert "MIT License" in model_license
    assert "Copyright (c) 2020-present Silero Team" in model_license
    notices = (
        packaging_root / "licenses/livekit-vad/THIRD_PARTY_NOTICES.md"
    ).read_text(encoding="utf-8")
    assert "GPL version 3 or later" in notices


def test_desktop_compatibility_overrides_pin_upstream_wheels_with_hashes() -> None:
    path = PROJECT_ROOT / "desktop/pet/packaging/runtime-macos-12.3.txt"
    requirements = [
        Requirement(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert {item.name for item in requirements} == {"numpy"}
    for item in requirements:
        assert item.url is not None
        url = urlparse(item.url)
        assert url.scheme == "https"
        assert url.hostname == "files.pythonhosted.org"
        assert "-cp311-cp311-macosx_11_0_" in url.path
        assert url.fragment.startswith("sha256=")
        digest = url.fragment.removeprefix("sha256=")
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)


def test_desktop_build_isolates_overrides_and_checks_final_binaries() -> None:
    scripts = PROJECT_ROOT / "desktop/pet/scripts"
    runtime = (scripts / "build-runtime-mac.sh").read_text(encoding="utf-8")
    sign = (scripts / "adhoc-sign-mac.sh").read_text(encoding="utf-8")
    spec = (PROJECT_ROOT / "desktop/pet/packaging/fish-memory-runtime.spec").read_text(
        encoding="utf-8"
    )
    assert "uv run --locked --isolated --python" in runtime
    assert "--with-requirements" in runtime
    assert "runtime-macos-12.3.txt" in runtime
    assert "build-onnxruntime-compat-mac.sh" in runtime
    assert "build-aec-compat-mac.sh" in runtime
    assert "build-sherpa-compat-mac.sh" in runtime
    assert '"${#ort_wheels[@]}" -ne 1' in runtime
    assert '"${#sherpa_wheels[@]}" -ne 2' in runtime
    assert "/opt/homebrew" not in runtime
    assert "FISH_MEMORY_AEC_ABSEIL_LIB" not in spec
    for script in (runtime, sign):
        assert "verify-macos-compatibility.py" in script
        assert "--target 12.3" in script
    assert sign.index("verify-macos-compatibility.py") < sign.index("codesign --force")


def test_compatibility_build_does_not_silently_downgrade_locked_runtimes() -> None:
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {
        package["name"]: package["version"]
        for package in lock["package"]
        if package["name"] in {"onnxruntime", "sherpa-onnx", "sherpa-onnx-core"}
    }
    assert versions == {
        "onnxruntime": "1.28.0",
        "sherpa-onnx": "1.13.4",
        "sherpa-onnx-core": "1.13.4",
    }, "Revalidate compatibility build sources and model outputs after runtime upgrades"


def test_public_documentation_is_not_ignored() -> None:
    ignore_rules = (
        (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    )

    assert "docs" not in ignore_rules
    assert "docs/" not in ignore_rules


def test_desktop_pet_has_a_local_mac_build() -> None:
    package = json.loads(
        (PROJECT_ROOT / "desktop/pet/package.json").read_text(encoding="utf-8")
    )

    assert "dist:mac:local" in package["scripts"]
    assert (
        "electronDist=node_modules/electron/dist"
        in package["scripts"]["dist:mac:local"]
    )
    assert "rimraf release dist" in package["scripts"]["dist:mac:local"]


def test_desktop_package_leaves_out_unused_weight() -> None:
    pet = PROJECT_ROOT / "desktop/pet"
    spec = (pet / "packaging/fish-memory-runtime.spec").read_text(encoding="utf-8")
    for module in ('"transformers"', '"hf_xet"'):
        assert module in spec.split("excludes=[", 1)[1].split("]", 1)[0]
    assert '"sherpa_onnx/lib/libonnxruntime."' in spec
    package = json.loads((pet / "package.json").read_text(encoding="utf-8"))
    assert package["build"]["afterPack"] == "scripts/prune-electron-locales.cjs"
    assert "make-dmg-mac.sh" in package["scripts"]["dist:mac"]
    assert (pet / "scripts/prune-electron-locales.cjs").is_file()
