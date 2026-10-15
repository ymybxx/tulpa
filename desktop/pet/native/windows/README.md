# Windows 11 x64 native layer

Build from a Visual Studio 2022 x64 developer shell with the Windows 11 SDK,
CMake 3.24+ and the project's installed Electron package:

```powershell
pwsh -NoProfile -File scripts/build-native-windows.ps1
npm run build:electron
node scripts/verify-windows-native.mjs
```

The script verifies official Electron header checksums, builds with the static
MSVC runtime, runs portable CTest cases, then installs the helper, N-API 8 addon
and license files under `runtime/win32-x64`. The addon delay-loads Node-API from
the actual host executable. Do not overwrite native resources while tulpa runs.

`fish-memory-perception-helper.exe` speaks protocolVersion 1 NDJSON on stdin /
stdout; stderr contains diagnostics, never frames or user text. Commands include
`configure`, `status`, `ping`, `claim_environment_frame`,
`cut_environment_window` and `shutdown`. Pass `--host-pid PID` to identify the
Electron host and its process tree for capture exclusion. `--permission-probe`
and `--capture-warmup [--audio-only|--with-system-audio]` use the existing desktop
permission-probe contracts. The normal protocol smoke test leaves capture off.
It also runs `--protocol-fixtures` through the production Electron parsers, with
synthetic WIC pixels, all nine interaction kinds, long Chinese/emoji text, and
screen-only / silent audio windows. This checks the real serializers without
reading any user's desktop, input, media devices or application profile.

Implemented capabilities:

- Screen: persistent Windows Graphics Capture of the primary monitor, float HDR
  input with explicit SDR tone mapping, JPEG at most 960px / 200KiB, memory-only
  three-second, single-use frames. Sampling changes preserve the capture session.
  Pause and discard frames on lock/secure desktop or when the host's visible
  windows are not excluded. The host must apply WDA_EXCLUDEFROMCAPTURE itself.
- Computer history: UIA bounded tree snapshots, frontmost-window changes, mouse
  click/drag/context menu, shortcuts/submission, selected text and accessible
  committed value changes. Password and recognized private-browser windows are
  excluded before values are read. Text input depends on a UIA ValuePattern:
  unsupported controls are omitted, never reconstructed from raw key codes.
  Browser private-window title detection is a conservative heuristic, not a
  claim that Windows exposes a universal incognito flag.
- System audio: process-loopback excluding the host process tree, via the
  accompanying audio module. Screen-only mode still emits disabled audio windows
  to drive the audiovisual history clock.
- Input addon: coalesced native pointer/press notifications, non-activating HWND
  setup, explicit-only input focus, and host/role Job Objects. Host Job must be
  initialized explicitly before spawning runtime services. Its handle is left
  open until process exit so normal Electron shutdown can flush files.

Camera, legacy OCR/sparse vision and the old independent workspace/accessibility
modules report unsupported. The currently enabled product path does not request
them. Automatic visibility on all virtual desktops, exclusive-fullscreen game
overlays, secure desktop access and elevated-app UIA completeness are not claimed.
WGC's system capture border remains visible unless Windows separately authorizes
borderless capture; the helper does not suppress that permission requirement.

Before release, run the Windows package/ABI gate and real Windows 11 x64 tests:
foreground app and IME focus, pointer pass-through, popup dismissal, 100/150/200%
mixed-DPI monitors, virtual desktops, fullscreen/borderless apps, HDR/SDR and
display reconnect, lock/unlock, audio playback-device changes, and crash cleanup.
macOS portable tests establish protocol/data logic only, not Windows SDK build,
native API behavior, driver compatibility or actual capture performance.
