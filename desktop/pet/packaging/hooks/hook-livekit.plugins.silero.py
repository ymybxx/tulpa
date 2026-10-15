"""Freeze only the official local VAD plugin and its required runtime assets."""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)

# Recursive metadata follows required dependencies with their markers, not the
# optional livekit-agents plugin extras. This also preserves wheel license files
# (including livekit-local-inference's LICENSE and MODEL_LICENSE) and metadata
# consulted by importlib.metadata / OpenTelemetry entry points at import time.
datas = copy_metadata("livekit-plugins-silero", recursive=True)
datas += collect_data_files(
    "livekit.plugins.silero", includes=["resources/silero_vad.onnx"]
)
datas += collect_data_files("livekit.rtc", includes=["resources/LICENSE.md"])
datas += collect_data_files("livekit.agents", includes=["resources/NOTICE"])

# rtc loads its FFI library through importlib.resources + ctypes. Keep its
# package-relative location. local_inference's _native extension is a hidden
# import below; collect_dynamic_libs retains any accompanying native libraries.
binaries = collect_dynamic_libs("livekit.rtc")
binaries += collect_dynamic_libs("livekit.local_inference")
# PyAV's macOS FFmpeg dylibs need their av/.dylibs layout preserved. Its upstream
# PyInstaller hook handles the compiled av submodules separately.
binaries += collect_dynamic_libs("av")

hiddenimports = [
    "livekit.plugins.silero.resources",
    "livekit.rtc.resources",
    "livekit.local_inference._native",
]
