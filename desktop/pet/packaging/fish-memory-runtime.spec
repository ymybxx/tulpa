from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_dynamic_libs


PACKAGING_ROOT = Path(SPECPATH).resolve()
PROJECT_ROOT = PACKAGING_ROOT.parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"

datas = collect_data_files("fish_memory", includes=["facial_test/*.json"])
# One maintained catalog is shared by TypeScript and the isolated Python role.
datas += [
    (
        str(PROJECT_ROOT / "desktop/pet/src/shared/facial-demo-parameters.json"),
        "fish_memory/facial_test",
    )
]
datas += collect_data_files("aec_audio_processing")
datas += [(str(PACKAGING_ROOT / "licenses" / "livekit-vad"), "licenses/livekit-vad")]
binaries = []
hiddenimports = [
    "aec_audio_processing",
    "sentencepiece",
    "sherpa_onnx",
    "sounddevice",
    "tokenizers",
    "webrtcvad",
]
if sys.platform == "darwin":
    hiddenimports += ["livekit.plugins.silero"]
binaries += collect_dynamic_libs("aec_audio_processing")

pg0_datas, pg0_binaries, pg0_hiddenimports = collect_all("pg0")
if sys.platform == "win32":
    # Keep pg0's pinned Python interface, but bundle our Unicode-capable Windows
    # build instead of the wheel's PostgreSQL executables with legacy code pages.
    pg0_executable = PACKAGING_ROOT.parent / "runtime/win32-x64/pg0/pg0.exe"
    if not pg0_executable.is_file():
        raise RuntimeError("Build the Windows UTF-8 pg0 runtime before packaging")
    pg0_datas = [(source, target) for source, target in pg0_datas
                 if Path(source).name.lower() != "pg0.exe"]
    pg0_binaries = [(source, target) for source, target in pg0_binaries
                    if Path(source).name.lower() != "pg0.exe"]
    pg0_binaries.append((str(pg0_executable), "pg0/bin"))
datas += pg0_datas
binaries += pg0_binaries
hiddenimports += pg0_hiddenimports
datas += collect_data_files("sherpa_onnx")
binaries += collect_dynamic_libs("sherpa_onnx")

analysis = Analysis(
    [str(SOURCE_ROOT / "fish_memory" / "desktop_runtime.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(PACKAGING_ROOT / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "face_alignment",
        # Pinned desktop models use tokenizer.json and our own downloader;
        # transformers is only a fallback for hand-installed custom models.
        "hf_xet",
        "librosa",
        "matplotlib",
        "onnx",
        "pandas",
        "pytest",
        "scipy",
        "sklearn",
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "cv2",
    ],
    noarchive=False,
)

# Some dependency hooks copy metadata for optional development backends merely
# because those packages happen to exist in the build environment. Keep the
# runtime independent of those optional packages even when a developer has them
# installed locally.
analysis.datas = [
    item
    for item in analysis.datas
    if not item[0].startswith(
        (
            "hf_xet-",
            "pytest",
            "torch-",
            "torchaudio-",
            "torchvision-",
            "transformers-",
        )
    )
]

# Sherpa's extension resolves `@rpath/libonnxruntime.1.dylib` to the copy
# beside the executable, and the onnxruntime Python binding links statically.
# The copies each wheel also ships in its own folder are never loaded.
UNUSED_ONNXRUNTIME_COPIES = (
    "onnxruntime/capi/libonnxruntime.",
    "sherpa_onnx/lib/libonnxruntime.",
)
analysis.binaries = [
    item
    for item in analysis.binaries
    if not item[0].replace("\\", "/").startswith(UNUSED_ONNXRUNTIME_COPIES)
]

pyz = PYZ(analysis.pure)

executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="fish-memory-runtime",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    target_arch="arm64" if sys.platform == "darwin" else None,
    codesign_identity=None,
)

collection = COLLECT(
    executable,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=False,
    name="fish-memory-runtime",
)
