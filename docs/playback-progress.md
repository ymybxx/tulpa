# Playback progress

Companion answers declare `playbackOwner: companion` in answer snapshots and start
with `consumedOffset: 0`. Text-only/browser answers retain the default `web` owner.
Pages observing Companion answers neither synthesize speech nor submit consumption.
A page interrupt request is delegated to Companion; the API acknowledges it only
after Companion has frozen and persisted the device-confirmed cutoff.

Two absolute Unicode-code-point offsets follow the original answer text:

- `displayOffset` reaches a sentence's end when its first PCM bytes are rendered.
  A sentence longer than 30 characters (DashScope can merge a paragraph into one)
  is revealed in step with its rendered share of PCM instead: exact once all of
  its audio has arrived, estimated from earlier sentences' bytes per character
  before that, and shown whole when there is nothing yet to estimate from.
- `consumedOffset` reaches that end only after all PCM preceding its end mark is
  rendered. Sentence announcements, received audio and generated text do not count
  as consumption. Unknown text annotations never authorize a guessed prefix.

PCM sinks retain zero-byte sentence-end marks in the same ordered queue as audio.
Draft PCM promotion replays these marks with the original audio; private buffering
never advances public progress. Device acknowledgements remain the authority.

`pet.playback.progress` publishes message/playback identity, revision, status,
both offsets and `visibleContent` (the currently authorized display prefix).
`/v1/pet/events?controlOnly=1` provides these controls without PCM. Publisher and
hub retain at most 32 message snapshots for reconnection. Generation SSE completion
does not terminate this playback stream. Interrupted/failed display retracts to
consumption, including when a complete LLM snapshot was already buffered.

A Companion answer owns one serial progress writer. Updates are coalesced, each
request has a 500 ms deadline, retries back off to 2 seconds, and final snapshots
are immutable. Subsequent draft context and final Ask wait for pending final
progress. Normal shutdown attempts a bounded final flush; forced termination can
only recover progress already acknowledged by the API.

Browser speech uses equivalent sentence-start and sentence-end AudioWorklet events.
Each carries a playback instance ID so old queued callbacks cannot affect a later
answer. Pure text display keeps its existing streaming behavior.

## Pet-side quick chat and speech bubbles

The desktop pet has a separate lightweight HUD (`pet-chat-window.ts`). Hovering
reveals the chat entry outside the model; the composer expands from the same
anchor. Layout uses the pet's display work area and reserves expanded dimensions
before placing the entry. It flips as a unit near an edge. It never resizes the
Live2D canvas. The existing native pointer monitor is shared; polling is only a
fallback when that monitor is unavailable and the HUD is visible.

“说话时显示气泡” is an independent desktop preference, stored atomically in
`pet-chat-ui.json`. Speech does not need a hovered pet or an open composer. The
card shows up to three scrolling lines, follows the published display prefix,
and stays briefly after completion. This is sentence/chunk synchronization, not
word alignment. Interrupted speech retracts to the confirmed consumed prefix.
Old completed snapshots do not replay on reconnect. Background/plugin/Agent
speech publishes the same display protocol through `PlaybackCaption`; its
existing consumption writer remains unchanged.

Quick text input uses authenticated `/v1/pet/chat` when Companion is enabled.
The one-shot `pet.chat.submit` command is not cached in reconnect snapshots;
`pet.chat.result` acknowledges the actual `/ask` acceptance. Companion freezes
its existing conversation/reply context, settles interrupted playback, then
uses the same normal answer/TTS pipeline. This does not enable the microphone.
When Companion is disabled, input uses normal `/ask` and text-only answer SSE.
No LLM or TTS provider keys are exposed to the HUD renderer.

A bounded request-id cache prevents duplicate submission after an uncertain
response. The composer keeps the draft on failure and remains usable. Known
pre-submission failures can retry; uncertain submission requires checking chat
history first. Closing the composer keeps its draft in memory; quitting the app
does not persist drafts.

Focused checks:

- `npm test -- src/shared/pet-chat.test.ts electron/pet-chat-service.test.ts`
- `pytest tests/pet/test_text_input.py tests/pet/test_routes.py
  tests/companion/test_playback_caption.py tests/companion/test_session.py`
