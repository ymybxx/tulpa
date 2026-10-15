# Environment history snapshots

The current host uses [Session History](session-history.md): five-second media
windows, local ASR with continuous speaker tracking, and unified per-session
compaction. Fixed-period media aggregation has been retired. The details below
document the legacy format and interfaces retained for existing records/clients.

Environment perception has two complementary history inputs: the existing
computer activity recorder and timestamped screen/system-audio snapshots.
Semantic history is stored as Observations, separately from conversation
Messages, Facts and memory Snapshots.

## Independent perception switches

| UI name | Config field | Capture | Ask input |
| --- | --- | --- | --- |
| 操作感知 | `operationPerceptionEnabled` | Application activity and operation records | Latest operation summary |
| 屏幕感知 | `screenPerceptionEnabled` | Current screen frame | One latest frame |
| 影音感知 | `audiovisualPerceptionEnabled` | Screen frames and system playback with hierarchical summaries | One-hour media history plus speech frames at 0, 1, 2… seconds |

The three choices are independent. Audiovisual perception already includes screen
input, so enabling both does not add a duplicate frame or change the screen
checkbox. Enabling operation and audiovisual perception supplies both histories.
Disabled sources are not read as fallback context. The original
`environmentPerceptionEnabled` migrates to operation and screen with the same value;
the new audiovisual setting defaults to false.

The desktop sends all three runtime flags to `/environment-perception/state` as
`operationEnabled`, `screenEnabled` and `audiovisualEnabled`. Legacy `{enabled}`
requests retain only the original operation-and-screen behavior.

## Capture and background processing

The desktop owns the capture lifecycle. Enabling audiovisual perception starts
periodic screen capture. On macOS 14.2 and newer, system playback uses an
exclusive Core Audio process tap that captures the global mix while excluding
Fish's application, audio helper and host process tree. Process-list changes
refresh that exclusion list in the background while capture continues with the
last installed list. macOS 13 through 14.1 retains the ScreenCaptureKit
compatibility backend and applies the same Fish-process filter where that API
permits it. Neither backend captures the microphone. macOS 12.3 continues to
capture images with audio marked unavailable.

Each capture session has an independent identity and monotonically increasing
ten-second window index. The recorder selects the available frame nearest the
window start and five-second point, preserving each frame's actual capture time.
One native timer drains a bounded sequential 16 kHz mono PCM16 buffer every ten
seconds. It keeps at most two windows of PCM, pads a short window with silence,
and reports the real captured sample count. This deliberately accepts small
boundary drift instead of maintaining per-chunk timestamps, coverage maps or a
second Electron window clock.

The local API accepts a bounded window and immediately returns `202`. System
sound reuses the Companion's authoritative SenseVoice recognizer through a
bounded local IPC request, so the process does not load a second ASR model and
voice turn timing is unchanged. The configured vision/text model receives the
two timestamped images and ASR text. For Qwen3.7 Flash this uses image/text input,
not unsupported raw audio. One model call returns `audiovisualSummary` for
media content and `activitySummary` for computer activity. The service stores
the redacted transcript as `audio`, preserving dialogue and
lyrics regardless of background-music classification. No music tags or empty
music markers are sent to the model. Minute merges also strip legacy music
wrappers while retaining their text. Music, sound effects and prosody cannot
be fully recovered from transcription.

System audio is first segmented locally with the pinned FP32 ONNX export of
`pyannote/segmentation-3.0`. Clean non-overlapping speech supplies CAM++ voice
embeddings, which are matched against a bounded in-memory voice table across
successive windows. This reuses the existing ONNX and sherpa runtimes and the
CAM++ model file; it does not load Whisper, WhisperX, PyTorch or an Omni model.
The speaker processing is additional local inference, and model-file size is
not a memory-use estimate. It creates a segmentation session and a CAM++
extractor; sharing model files does not imply sharing the owner verifier's
resident extractor or eliminating that extra memory. Speaker inference uses one
CPU worker with single-threaded model execution. It never reads the user's
enrolled owner-voice profile or saves new voice templates. Model downloads are
pinned by byte length and SHA-256.

The segmentation model has up to three local speaker channels per ten-second
window and represents at most two simultaneous speakers. The cross-window voice
table defaults to sixteen recent templates, evicting the least recently used
unprotected template at capacity. Identity numbers are never recycled; an evicted
person returning later may receive a new number. Matching thresholds and margins
are conservative starting points, not calibrated accuracy guarantees. Short
turns without enough clean speech, overlapping dialogue, music and similar
voices can still leave uncertain attribution. Correct acoustic labels also do
not guarantee that Flash will infer a character's name or a spoken address
correctly; the original labeled dialogue remains available to lower-layer merges.

Each detected interval is cropped from the original PCM and passed to the same
SenseVoice recognizer. The resulting text inherits that interval's timing and
speaker label; these are utterance intervals, not word-level alignment. A line
such as `[00:01.00—00:03.50] 声音abc123-01：...` carries time relative to its
ten-second window. Capture and Fish identities scope the local tracker, while a
fresh opaque voice namespace prevents a reset from reusing an old speaker's
label. When a local turn exists but its historical identity is unconfirmed, its
words retain a window-scoped label such as `本片段声音A（未关联历史人物）`.
These letters reset for each time window and are only local speaking-turn clues,
not proof of distinct real people. Different unknown local turns are never
merged merely because both global IDs are absent. Overlapping or unsegmented
speech remains explicitly unassigned. Acoustic
labels are fallible continuity evidence, not names, ages or family roles.

Creating or updating a voice template requires at least 1.2 seconds of clean
speech. Matching an existing template permits at least 0.5 seconds of clean
speech with a stricter similarity threshold and top-two margin; short matches
refresh recency without changing the template. Long turns are resolved first,
so a short turn can independently match a voice learned later in the same
window. Model-local cannot-link constraints still apply. The short thresholds
are candidates to validate, not universal speaker-identification standards.

For adjacent windows, a local boundary tracker keeps at most the last two
seconds of PCM (64,000 bytes) and their reliable identity evidence in memory.
Only an unresolved short turn touching the window boundary can trigger one
additional segmentation of the old two seconds plus up to two new seconds,
zero-padded to the model's ten-second input. Continuous reliable speech and
time-overlap correspondence must support inheritance; a competing voice,
overlapping speech, pause, or contradictory embedding prevents it. This never
creates or updates a template and never sends old PCM through ASR again. Capture
gaps, partial audio, timestamp discontinuity, reset and close clear the tail;
global templates can survive a short gap independently. A boundary inference
failure retains the normal current-window result.

The window processor serializes segmentation and cropped recognition before
allowing the next audio window to enter. Microphone recognition keeps priority:
if it preempts the remaining crops, already transcribed intervals are retained
with partial coverage. Segmentation or model-loading failure falls back to the
original whole-window transcript without inventing speaker labels. Empty or
excessively fragmented segmentation also retains the original transcription
path. A settings switch can disable speaker processing independently of
audiovisual perception.

The server permits two concurrent background jobs and at most twelve pending
windows. Text-only compaction is serial within the active Fish generation, so a
startup backfill cannot occupy both model slots needed by live ten-second work.
Retries are bounded, failed windows do not block later ones, and the desktop also
bounds queued/in-flight uploads. Raw JPEG and PCM remain in memory only. Besides
the bounded two-second continuity tail, they are released after processing,
cancellation or failure. Closing, locking, changing
the active Fish or disabling perception clears the desktop capture buffers and
the server's hot history view. Disabling server perception revokes pending work
by generation. Already saved semantic history is attached to Ask only when its
corresponding perception switch is enabled.

## Hierarchical Observations

Completed windows are saved under source `environment_history` and type
`semantic_snapshot_10s`. The input digest and deterministic ID make repeated
uploads idempotent and reject a changed payload under the same identity. Only
the model-written `audiovisualSummary`, `activitySummary`, and exact redacted
`audio` text are stored in the semantic document, together with timestamps,
media coverage and source identifiers. The evidence records segmentation status
and each interval's start, end, voice label, local turn ID and overlap flag,
without a second copy of its text. Debug evidence also records clean duration,
top-two similarities, assignment reason/source and optional boundary-check
reason. These diagnostics and internal local IDs do not enter model prompts.
If optional scores would exceed the 32 KiB envelope, they are omitted with an
explicit marker before any narrative text is lost. New audiovisual rows do not duplicate
those fields in
`documentMarkdown`, and do not store a model-written `visual` field. Raw frames
and PCM never enter Observation SQL or model diagnostics.

Migration 39 permits the full display summary in the SQL `summary` column for
`environment_history` semantic snapshots, up to 31,000 characters. It preserves
existing rows and the 2,000-character limit for other Observation sources and
types. The semantic document's 31,000-byte and enclosing payload's 32 KiB limits
remain the stricter application storage bounds; SQL display text is not clipped.

A minute group consists of six consecutive windows from one capture session.
All six must be successfully saved and their time ranges must be contiguous
before the text model merges them. Missing, failed or still-running windows do
not count as completed. Partially observed windows can still have a semantic
snapshot, with missing audio/images explicitly recorded. Windows with neither
images nor captured audio are skipped without a model call or fake Observation.

The minute result has type `semantic_snapshot_60s` and records the six source
Observation IDs. Its summary model receives six ordered `segments`, each pairing
both ten-second summaries with that same window's exact `audio` and actual time
range. The service joins and preserves the minute audio under each original
window's UTC time range without asking the model to rewrite it. Relative speech
times remain attached to their own window, and labels survive this merge
unchanged. The audio field permits 12,000 characters to accommodate those labels
and time headers; the complete document still must fit 31,000 UTF-8 bytes.

Five complete minute nodes are then merged into
`semantic_snapshot_300s`; six complete five-minute nodes are merged into
`semantic_snapshot_1800s`. Every parent records only its direct children, and
all children must belong to the same Fish and capture session with the exact
expected indexes and contiguous ranges. Five-minute requests retain each minute's
transcript alongside both summaries so the model can correct speaker attribution
and reconnect the story from the original words. Half-hour requests use only
the ordered five-minute summaries. Both parent documents deliberately keep
`audio=""`: full minute audio remains available at the lower layer, while
repeatedly concatenating it would exceed the Observation payload budget.

Every layer uses the same two-category contract. Media summaries describe
dialogue, narrative, game events, music or other played content; activity
summaries describe the user's computer operations, work, browsing and application
progress. A category with no supporting evidence stays empty. One Flash call
produces both categories, and parents merge each category into its corresponding
output rather than mixing the two or introducing another analysis stage.

Activity summaries stay concise at every duration: one sentence per independent
activity, usually one to three sentences, with more allowed when there are more
important activities. They retain what the user did, the necessary work or content
subject, and clear progress, outcomes or unresolved problems. Continuous browsing
or viewing of the same topic is combined. Pet appearance, window layout, repeated
idle states, recommendations and incidental log statistics are omitted unless
they are what the user is actively examining. Meaningful changes, payments,
errors and test outcomes remain. Viewing chat, diagnostics or history is recorded
as that activity; old text visible inside those views is not repeated as newly
observed facts. These are shared leaf-and-merge instructions, with no extra model
call, character slicing, ellipses or hard sentence-count limit.

The same media merge rules apply at every duration: preserve the order and
causality of events, keep established people and relationships consistent,
deduplicate repeated content, and leave uncertainty unresolved. Nodes shorter
than 30 minutes have no fixed few-hundred-character media summary limit; this
includes the five-minute layer, while activity remains concise. At 30 minutes
and above, the two summaries share a total 600-character compression budget.
Transport, model-output and Observation byte limits still
bound payloads at every layer. Merge inputs have a 128 KiB limit: optional prior
context is omitted first if necessary, and an oversized current input is rejected
without clipping children. Model outputs are capped at 3,000 tokens. Larger
periods use the same contract and threshold
rather than needing a new narrative prompt for each duration.

Summaries use the complete acoustic label when associating speech across windows;
they cannot merge labels merely because their final number matches. They retain
the labels or evidence-backed label-to-character mappings needed to follow
alternating dialogue. A spoken address such as “奶奶，你能借我……” identifies the
addressee, not the speaker. Names and family roles still require dialogue or
visual evidence. Unlabeled legacy/fallback transcription may contain multiple
speakers; newlines alone are never speaker evidence. Summaries distinguish
evidence-backed names/roles, dialogue partners, impersonated relationships and
later reveals; ambiguous or overlapping speech stays unassigned. Ongoing dialogue takes priority
over repeated browser chrome or desktop-pet appearance, and background speech
does not change ownership merely because the foreground app changes.

Before a merge, the service may reuse already-completed narrative context from
its bounded cache, restricted to the same Fish and capture session. Each category
can retain its latest valid preceding content across a gap of at most 60 seconds;
an empty category in a newer node does not erase the other ongoing narrative.
The service prefers aggregates when available and keeps complete category bodies.
This background context is bounded, is omitted rather than clipped when
unavailable or too large, and adds no SQL, model stage, or waiting for earlier
work. It supplies continuity only; it cannot extend current coverage or turn old
events into new ones. Empty or partial transcripts never establish that a whole
period was silent. Narrative context supplements acoustic continuity without
turning uncertain voice assignments into confirmed character identities.

Concurrent completion triggers only one merge per parent identity; a final
child arriving during an incomplete range check causes a recheck. Enabling
audiovisual perception hydrates the latest 70 minutes with one bounded query,
then repairs only missing hierarchy parents. Hydration includes complete parents
crossing the left boundary, using their ending time instead of excluding a parent
whose start is older than the nominal range. Its minute recovery scan uses the
same time bound, so startup work does not grow with the age of the database.
There is no permanent database polling. If a text merge returns unusable JSON
even after its one bounded repair, a deterministic local fallback keeps every
child's complete summaries and any available unassigned dialogue in order if they
fit the document's 31,000-byte UTF-8 budget. Category bodies and unclassified old
summaries stay separate in this fallback. Unassigned dialogue is preserved as
“听到的话” in the unclassified fallback text so Ask can still see it after the
recent-audio cache expires. The fallback leaves `audio` empty instead of storing
that same text twice. At 30 minutes and above, the combined
fallback summaries must also fit the same 600-character budget; otherwise the
parent stays unavailable and its children remain usable. No child is shortened
to make a substitute parent. The enclosing Observation still has its 32 KiB
limit. Transport and provider failures still use the normal retry policy.

At startup, and at most once every five minutes as minute nodes complete,
retention deletes at most 500 detail nodes older than two hours when a durable
direct parent explicitly names them.
This progressively removes 10-second, 60-second and 300-second duplication while
keeping 1800-second semantic nodes. The two-hour grace exceeds both the one-hour
Ask horizon and the upload retry window.

## Ask and speech frames

Ask reads an already published one-hour view from bounded process memory. It
does not issue Observation SQL, wait for pending model work, or run another
summary model. During active capture the range is anchored to the latest
completed window edge, avoiding fragmentation when Ask arrives milliseconds
after a ten-second boundary. If capture has been quiet for more than 20 seconds,
the real wall-clock one-hour boundary is applied again.

The selector keeps the newest minute at 10-second resolution, even when its
minute summary has completed. It anchors on the nearest widest complete block
beginning before the nominal one-hour cutoff, including when the cutoff falls
in a restart gap after that block. With enough
history the actual range can therefore exceed one hour; original timestamps are
never clipped or invented. Selection keeps the widest available non-overlapping
nodes regardless of an earlier capture's unfinished tail. The displayed layers
can only become finer: 1800, then 300, 60 and 10 seconds. When an earlier tail or
missing parent leaves finer nodes before a later coarse node, the view combines
those earlier texts into buckets at the later node's layer, retaining all text
and actual timestamps. Incomplete buckets explicitly show their shorter range.
This local presentation step neither invents durable parent Observations nor
calls a model, and never expands a ready parent into all its children merely to
preserve layer ordering. Thus a continuous
10:00–10:44 capture produces 10:00–10:30 at 30-minute resolution,
10:30–10:40 at 5-minute resolution, 10:40–10:43 at 1-minute resolution, and
10:43–10:44 at 10-second resolution. All entries remain in chronological order.
Missing or failed parents expose their available children instead of blocking
Ask; a coarse node is used near Ask only when its overlapping children are absent.

The model input groups contiguous display buckets of the same layer under
their own time range and granularity label (30min, 5min, 1min or 10s). A gap or
capture-session change starts another group; unavailable windows are never
presented as continuous coverage. The overall range supplies the UTC date, and
group ranges repeat the date when crossing midnight. Completed summaries and
recent speech are rendered in full, preserving paragraph breaks. There is no
per-line token truncation or window subsampling. The hierarchy and time horizon
bound selection; if compaction is unavailable, the full fine-grained history can
produce a larger prompt instead of silently losing words or windows.

Both summary categories enter the one-hour view under separate natural-language
labels, within the same chronological time groups. Media story progression stays
separate from computer activity even when the foreground app changes while
background audio continues. Exact recent system-playback `audio` remains a
separate, bounded one-minute in-memory section. Each audio window is shown under
its own UTC time range so repeated relative segment times cannot appear to be
simultaneous. Completed speaker segments enter this view before Flash finishes,
after text in both the transcript and individual segments has been redacted.
New snapshots contain
`audiovisualSummary`, `activitySummary`, and `audio`. Old `summary` values remain
readable as unclassified legacy observations; reading a record does not relabel
its content as media or activity. A later model merge may classify those old
observations when their content supplies clear evidence, while a local fallback
keeps their legacy category. Legacy `visual`, `uncertainties`, and
`documentMarkdown` values remain readable but are ignored by the new contract.
Operation summaries are loaded independently when operation perception is enabled;
they are never substituted for disabled or unavailable audiovisual history.

The answering model receives plain prose headed by “你周围的情况”, with sections
for “最近的电脑操作”, “你之前看到和听到的事”, and “你刚才听到的声音”. IDs, counters,
checkpoint/source names and processing timestamps stay in internal objects and
diagnostics. Eye-level image captions describe what the pet saw at the recorded
time. Observation answers and proactive speech use the same direct perception
perspective, while preserving the distinction between external text/dialogue and
instructions from the real user. Historical or unclear evidence is never treated
as a newly observed fact, and facial cues do not establish a person's true mood.

During user speech the Companion sends the logical utterance's first START time
and the candidate's frozen capture cutoff with the existing draft request. The
API selects the latest available image at utterance time zero, then at each
one-second boundary, from its bounded memory ring. Frames retain actual capture
timestamps; missing samples are not duplicated. A sub-second phrase has only its
initial frame. Each revision owns an
immutable ordered image package. Continued speech preserves the utterance start
while the next revision gets an updated package. Endpoint commit adopts the same
prepared task, without recapturing or rebuilding context. Typed Ask retains the
existing one-latest-frame behavior.

## Diagnostics

`GET /internal/fishes/{fish_id}/environment-perception/history/status` reports
pending/successful/failed work and the latest completed ranges. Window status is
available under `/history/windows/{snapshot_id}`. Failure logs include phase,
exception type and HTTP status, never raw images, audio, provider error bodies or
transcription fragments.

ScreenCaptureKit requires a valid macOS screen-recording grant for the current
application signature. Core Audio process taps also require
`NSAudioCaptureUsageDescription` and the user's system-audio recording grant.
Local ad-hoc packages can have a new CDHash after each build; if macOS rejects an
old grant, re-authorize the final package in System Settings. Screen and audio
state changes are diagnosed without captured content.

The older computer activity queue now backs off failed segments independently;
one old failure cannot keep all later segments pending. Its additional error
diagnostics record only structural validation information. Raw activity files
remain until their persisted summary has been verified.

Model reference: [Qwen3.7 Flash input capabilities](https://help.aliyun.com/zh/model-studio/qwen3-7-flash).
