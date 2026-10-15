# tulpa desktop app

This directory contains the tulpa desktop application for macOS (Apple
Silicon) and Windows 11 x64; see [Windows desktop](../../docs/windows-desktop.md)
for the Windows build. The
Electron main process owns the Live2D pet, chat, settings, tray, local
configuration, and the bundled Python API/voice runtime.

## Interface language

Choose **简体中文 / English** in the first-time setup or under
**Settings → General → Interface language**. The choice is saved in
`userData/ui-preferences.json` and updates open windows and application menus
without restarting the companion. Existing installations keep Chinese by default.
This preference changes app copy only; character names, editable descriptions,
chat history, model responses and third-party plugin content stay in their original language.

UI translations use [i18next](https://www.i18next.com/) with bundled English
catalogs in `src/i18n/messages/`; the Chinese source is the fallback key. Use
`t(source, values)` for app-owned copy and subscribe with `useLocale()` in Preact
page roots. For imperative DOM, use `bindLocalizedText` / `bindLocalizedAttribute`
or explicit `data-i18n` markers with `localizeStatic`. Keep user content out of
translation lookup, and use a values callback when an interpolation contains
another translated label. Language changes must never remount an editor or
write AI/runtime configuration.

## Local call logs

AI call logs retain requests, responses, and embedded images for **7 days** by
default. Change **Settings → General → Log retention** to any whole number from
1 to 365; the running service picks up the change without restarting voice.
The standalone Python service accepts `FISH_MEMORY_LLM_LOG_RETENTION_DAYS`.

Cleanup runs when the API finishes starting and every 15 minutes while it runs,
including when there are no new calls. It deletes whole calls whose start time
is older than the retention window; a late response cannot recreate an expired
call. Existing expired logs are removed on the next cleanup. Images remain
viewable while their log is retained. Chat history and long-term memories are
separate and are not deleted by this cleanup. Usage totals are derived from the
remaining call logs, so a retention period below 7 days also limits the weekly
usage summary.

## Startup recovery

Model preparation failures expose a retry action in settings and the pet status.
Retry restarts the local service if it never became ready; when only voice
preparation failed, it restarts the voice layer and preserves the healthy API
and audio preferences. First-time setup also supports retrying preparation and
AI/Jev validation without re-entering the current draft.

## Development

`npm run dev` uses Vite, Electron, and the repository's `uv` environment so UI
changes do not require rebuilding Python. On first launch the settings wizard
asks for service and audio configuration. The Cubism Core and the official Mao
sample model ship inside the packaged app and are copied into place on first
launch (a download from the original hosts remains the fallback). A proxy on `127.0.0.1:7890` is detected
automatically.

The Cubism runtime, official sample, downloaded voice models, database, logs,
and imported models live under Electron `userData` and are never put in the
repository. The packaged app does carry the Cubism Core and the Mao sample
(since 2026-09-26, so the first launch needs no overseas download), which makes
it a redistributor of these Live2D files: confirm the terms below before any
public release. Applicable upstream terms are:

- https://www.live2d.com/en/sdk/license/
- https://www.live2d.com/en/learn/sample/model-terms/

Start the complete development application from this directory:

```bash
npm run dev
```

## Independent macOS application

```bash
npm run dist:mac:local
```

Quit tulpa before rebuilding. Packaging checks for processes using this
checkout's app or runtime at entry and before replacement, and stops if any are
still running. Overwriting a live PyInstaller runtime can break its later module
imports even when the newly built file is intact.

The command builds a clean arm64 `base + voice` PyInstaller runtime, builds the
Electron renderers, creates `release/mac-arm64/tulpa.app`, removes
quarantine metadata, and signs the result. The script prefers a keychain identity
named `Fish Memory Local Development`; `FISH_MEMORY_CODESIGN_IDENTITY` can name a
different identity. A stable certificate keeps the app and capture Helper's
designated requirements stable across rebuilds, so macOS can retain Screen
Recording permission. Without one, the script falls back to ad-hoc signing and
warns that the permission may need to be granted again. The resulting app does not
need the repository, Python, uv, Node.js, PostgreSQL, or a terminal. Models are
downloaded on first launch, except the Live2D Core and Mao sample and the
speech recognizer, which ship inside the app so the first launch works without
waiting. The small MIT-licensed Silero VAD
model used by the local LiveKit audio backend is included in the application;
it needs no model download or LiveKit Cloud connection.

### Stable local development signature

An ad-hoc signature identifies one exact build. Reopening that unchanged build
does not change its identity, but rebuilding it usually changes its code hash and
therefore its designated requirement. To let macOS recognize later local builds
as the same app, create one persistent development identity in Keychain Access:

1. Open **Keychain Access** and choose **Certificate Assistant > Create a
   Certificate**.
2. Set the name to `Fish Memory Local Development`.
3. Choose **Self Signed Root** for **Identity Type** and **Code Signing** for
   **Certificate Type**.
4. Enable **Let me override defaults**, choose a long validity period, then keep
   the remaining defaults and create the certificate in the login keychain.
5. Confirm that it is available to `codesign`:

   ```bash
   security find-identity -v -p codesigning
   ```

Keep that certificate and its private key. Replacing or deleting it creates a new
identity and can require permission again. The next `npm run dist:mac:local`
selects the identity automatically, signs the main app and its capture/audio
Helpers, and verifies that none of their designated requirements is tied to a
build-specific `cdhash`. The first build that switches from ad-hoc to this stable
identity still needs Screen Recording permission once.

The self-signed identity is only for development on the machine that owns its
private key. Do not distribute a self-signed build to users; public releases need
an Apple-issued Developer ID Application certificate and the normal notarization
flow. To select a different valid identity explicitly, pass its certificate SHA-1
or unique name, for example:

```bash
FISH_MEMORY_CODESIGN_IDENTITY=0123456789ABCDEF0123456789ABCDEF01234567 \
  npm run dist:mac:local
```

Setting `FISH_MEMORY_CODESIGN_IDENTITY=-` explicitly requests the old ad-hoc
behavior for a disposable build.

### macOS 12.3 compatibility builds

The desktop build uses Python 3.11 and a separate compatibility overlay; it
does not downgrade the project's development environment or change model files.
`packaging/runtime-macos-12.3.txt` selects the hash-pinned macOS 11/OpenBLAS
build of NumPy 2.4.6. The native build scripts retain ONNX Runtime 1.28,
AEC 1.0.1 and Sherpa 1.13.4, rebuilding them for macOS 12.3 with bundled static
Abseil and a shared compatible ONNX Runtime build. Do not downgrade ONNX
Runtime: a 1.19.2 candidate changed existing Smart Turn decisions. The compatible
runtime retains CPU inference and omits unused CoreML/Azure providers; current
voice, Smart Turn and local embedding code explicitly use CPU. Validated wheels
are cached under the repository's `.cache/macos-12.3-wheels/`; the first build
therefore takes longer than later builds.

Both the Python runtime and final `.app` pass
`scripts/verify-macos-compatibility.py` before release signing. This checks real
arm64 Mach-O deployment commands, not just package names or `Info.plist`.
It does not prove runtime compatibility on an old OS: a release still needs
macOS 12.3 device testing for audio, ASR, VAD, Smart Turn and local embeddings.
Do not lower binary version markers to bypass the check.

### Environment history

Environment perception also records timestamped screen/system-audio semantic
history. System sound is transcribed locally; the configured model receives
images and text. See [Environment history snapshots](../../docs/environment-history.md)
for capture coverage, macOS support, minute merging and voice-frame sampling.

### Local owner voice recognition (experimental)

Settings includes an independent, default-off owner recognition switch. Click
**录入主人声音**, wait for the recording indicator, and speak normally alone for
15 seconds. The first enrollment downloads the hash-pinned 28 MB CAM++ model.
Recording temporarily suspends only the Companion, not the authenticated chat
API, and resumes its previous enabled/disabled intent on completion or cancel.
Enrollment never invokes KWS, ASR, Ask, memory extraction, or a cloud voice API.

Only normalized speaker embeddings and model/backend metadata are saved in
`<userData>/data/owner-voice/owner-voice.json` (0600, atomic replacement).
Enrollment PCM stays in memory. Re-recording replaces the profile only after
quality/consistency checks succeed; deleting it disables owner recognition.
The original optional audio diagnostics remain a separate opt-in facility.

When enabled, bounded processed-PCM blocks are checked before reaching KWS,
ASR, voice drafts or playback interruption. The initial policy checks about
800 ms of effective speech at a time, with a final check for shorter utterances
of at least 300 ms. A very short tail may use bounded recent verified context.
Every new block is checked, not just the first utterance of a session. During
TTS, provisional matched speech can pause/duck playback; irreversible interrupt
waits until the final block passes. Rejected continuations abandon their draft
and resume paused playback. An unrelated rejected speaker after an already
verified END does not erase that earlier valid question. Pending speaker
verification may hold a Smart Turn reopen for at most 400 ms, never indefinitely.
The existing path is unchanged when the switch is off.

Missing/incompatible voice profiles or inference failures block microphone
admission while recognition is enabled; they never silently accept everyone.
Chat, memory and output remain available. Re-record or disable the setting to
recover. Changing the capture backend requires re-enrollment.

The model identity/hash and audio processing backend are recorded with the
profile. A future ERes2NetV2 switch can reuse the service/gate/UI boundaries, but
must change model metadata, re-enroll the voice and recalibrate thresholds;
embeddings from different models are not interchangeable. The initial 0.5
similarity threshold is experimental, not a measured false-accept guarantee.
Short/quiet/mixed speech and playback of the owner's own recorded voice can
still be misclassified: this feature is not authentication or source separation.

## Model management

Open **Models** in the App to manage Live2D characters without starting or
restarting the Python voice runtime. A complete model can be imported from a
folder or ZIP containing exactly one `.model3.json`. The App validates archive
paths, referenced resources, motions, expressions, and model structure before
installing it.

The three-column model page provides:

- a persistent model library with active-model markers and preview images;
- an independent Live2D preview that never drives or interrupts the desktop pet;
- editable state, emotion, and gesture pools containing any number of assets;
- standalone `.motion3.json` / `.exp3.json` and asset-ZIP imports;
- equal preview, rename, mapping, and deletion controls for every installed asset;
- display, hit-area, and lip-sync parameter editing with explicit save/discard.

Each semantic pool selects assets with equal probability and avoids immediately
repeating the previous asset when alternatives exist. A state keeps the same
selected motion until the state changes; a one-shot gesture temporarily
preempts it and then restores it. Expressions, motions, physics, and final
lip-sync can run together.

Installed files and the generated schema-v2 `pet.manifest.json` live under
Electron `userData`. Source folders and ZIPs are never modified, and selected
ZIP files are not copied into App storage; only validated extracted model files
are installed. There is no distinction between bundled and later-added motion
or expression assets.

## AI character generation

The model page includes an **AI Create Character** flow backed by one local
`FAL_KEY`. The key is stored only in Electron `userData/secrets.json` with mode
`0600`; it is never exposed through preload, renderer state, job JSON, or the
generated model. Configure it in Settings before starting a paid job.

The first release intentionally targets one owned full-body-girl master. It
generates four constrained front-pose candidates, waits for the user to choose
one, recursively decomposes and validates parts, writes them into the fixed
texture atlases, then waits for a second confirmation on the real Live2D
preview. Saved output is an ordinary schema-v2 model and uses the existing
motion, expression, state, gesture, physics, and lip-sync management.

Master binaries are not committed or bundled. Install an owned master folder
containing `master.template.json`, a schema-v2 `pet.manifest.json`, and the
complete Cubism runtime files. See
[`docs/ai-master-template.md`](docs/ai-master-template.md) for the format.

Jobs and downloaded fal results are kept under Electron `userData`. fal uploads
and generated URLs request a one-hour lifecycle, cancelled jobs discard late
responses, and completed/cancelled jobs remove local intermediate images. The
default hard budget is `$1.70`; a call that would cross it pauses before the
request and requires explicit confirmation of a higher limit.

The renderer supports Cubism 4 models and uses the local Cubism Web Core. Cubism
5 models are supported when paired with a compatible current Cubism Core. Before
public distribution, confirm the Live2D Expandable Application license terms.

## Design constraints

- The window is transparent, frameless, always on top, and click-through outside
  configured model hit areas.
- State/expression/audio events include a generation. Late events from cancelled
  generations are ignored.
- PCM mirroring is bounded and disposable. A slow or closed desktop client never
  blocks actual audio playback.
- Lip sync runs in an AudioWorklet and is applied after model motion/expression
  updates, so motions cannot overwrite the mouth in the same frame.
- Asset/model mutations are owned by Electron, use bounded extraction and
  atomic metadata saves, and never expose arbitrary filesystem paths to the UI.
