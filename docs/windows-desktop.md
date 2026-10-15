# Windows 11 desktop

The first Windows target is Windows 11 on Intel/AMD x64. Windows 10 and Windows
on ARM are not part of this build. macOS keeps its existing native helpers and
build commands. Electron remains at the repository's existing pinned version.

## Develop from a Mac

Edit the same checkout and run the normal portable checks on macOS. Build the
Windows binaries on Windows x64 using the `Windows desktop candidate` workflow
in `.github/workflows/windows-desktop.yml`, or on a Windows development machine.
The workflow does not publish a GitHub Release. Push/dispatch only within the
repository owner's authorization.

GitHub's Windows runner is Windows Server, so a successful run covers compilation
and package components, not Windows 11 desktop usability, microphone behavior,
or graphics hardware. An Apple Silicon VM runs Windows ARM and does not replace
the x64 acceptance machine.

## Windows build

Install Git, Node.js 22 x64, uv, PowerShell 7.2+, CMake 3.24+, and Visual Studio
2022 or newer with Desktop development with C++ and a current Windows SDK,
and the Rust MSVC toolchain (Cargo). Windows packaging rebuilds pinned pg0 with
UTF-8 PostgreSQL executables; macOS continues using the existing wheel.
Run from a regular user account; the app does not require administrator rights.

```powershell
# Repository root
uv sync --locked --dev --extra voice --python 3.11.15
npm ci --prefix desktop/pet
cd desktop/pet
# Once, and again after Python backend changes:
npm run prepare:runtime:win
npm run dev:win
```

Windows development runs the verified frozen Python backend and its UTF-8 pg0,
while Electron keeps loading the renderer from Vite with hot updates. The
preparation command builds the backend and plugin interpreter once; `dev:win`
checks that those resources exist and does not rebuild Rust or Python. After
changing Python code, exit the app and rerun `prepare:runtime:win`. The regular
macOS source-based development command is unchanged.

Create a candidate installer after exiting development and packaged instances:

```powershell
npm run dist:win:local
node scripts/verify-windows-package.mjs release-win/win-unpacked
```

This prepares pinned Python/model resources, builds native modules, builds the
shared Electron/renderer code, and creates the x64 NSIS installer. The packaged
component gate uses temporary processes without opening devices, capturing the
screen or accessing a real user profile. It verifies native ABI/job cleanup,
helper protocol and an offline runtime doctor. Runtime PE files are checked for
x64 architecture. Local plugin source trees are optional; the generic plugin
SDK, interpreter and uv are bundled independently.

An additional database gate runs only in the disposable GitHub-hosted Windows
account. It starts the bundled pg0, checks the vector extension and SQL, and
closes it normally using a unique instance and temporary Chinese data directory.
The Windows-specific pg0 build honors `TULPA_PG0_HOME` so its registry and
extracted executables are isolated in a separate Chinese directory as well.
The desktop sets this to `data/pg0-runtime` inside its own application profile.
The gate remains explicitly enabled only on disposable hosted CI.

## Required Windows 11 acceptance before release

Use a clean x64 Windows 11 machine without development tools, then record its OS
build, GPU/driver, audio devices and installer SHA-256 with the results.

1. Install, first launch, Chinese/English onboarding, model preparation, chat,
   embedded database startup, relaunch and upgrade with retained data.
2. Transparent pet rendering, click-through outside the character, hover without
   focus theft, dragging, explicit text input, menus, DPI changes, multiple
   monitors, screen locking and sleep/resume.
3. Microphone permissions, natural interruption during playback, speaker
   self-echo, Bluetooth/headset/default-device switching and device removal.
4. Environment screen/audio alone and together; verify self-exclusion with an
   actual recording and verify screen-only history advances. Exercise HDR/SDR
   monitors, protected content and unavailable capture surfaces.
5. UI Automation history for browser, text editor and Chinese IME; confirm password
   controls and private-window history are excluded and unsupported/elevated
   applications report limits without inventing events.
6. Normal exit, forced host exit, restart, stale-service recovery and model-owner
   handoff. Confirm no orphan Python, pg0 or perception processes remain.
7. Unsigned-installer prompts, uninstall/reinstall and data preservation. Windows
   code signing and public distribution approval are separate release work.

Optional camera/legacy vision modules remain unavailable, matching the current
product's disabled policy. Do not infer support from the presence of a helper or
from successful unit tests.

The initial UIA implementation refreshes its tree on foreground-window/title
changes and observes accessible committed values. Same-window structure-change
diffs and browser URL extraction are not yet implemented; controls without
ValuePattern have no reconstructed text input. Those differences, virtual
desktop visibility and acoustic behavior need explicit acceptance before
claiming feature/experience parity with macOS.
