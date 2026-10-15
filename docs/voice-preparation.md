# Voice answer preparation

A voice revision owns one preparation task. A pause starts computation; the
speech endpoint adopts that task. It does not create a second transcript,
capture another screen image, reload the answer context, or compare two newly
assembled model requests.

## Ownership and input

The Companion retains its turn binding and exact Agent reply context for the
logical utterance. Each resumed utterance invalidates its previous candidate
revision. After the next pause, the complete utterance is decoded and prepared
again. An unchanged endpoint reuses the same completed or in-flight decode.
The audio queue budget limits starting a speculative decode; it does not
invalidate a matching decode that already exists. Reuse still checks missing
audio, new speech, in-flight frames, the PCM prefix, and turn ownership.
Before consuming its audio buffer, the endpoint joins any existing draft PUT
registration request. This prevents the commit request overtaking registration.
If speech resumes during that wait, the old endpoint relinquishes ownership
and the next candidate retains the whole utterance and its reply binding.

With LiveKit Silero, the candidate silence gate uses the activity decision from
the same inference window as the formal endpoint detector. It applies the
existing activation/deactivation thresholds before the endpoint's silence hold.
The first silent frame starts private ASR preparation. The candidate gate
authorizes server preparation after 100 ms of continuous silence, reusing that
completed or in-flight recognition instead of decoding again. The formal
350 ms endpoint and Smart Turn remain unchanged. No additional VAD inference
is needed.
Backends that do not expose window activity retain the standalone WebRTC VAD
fallback. Owner admission observes the original window activity before replacing
unchecked PCM with silence, and replacement frames carry a silent candidate
signal as well.

The independent `FISH_MEMORY_COMPANION_VOICE_DRAFT_CANDIDATE_SILENCE_MS`
setting defaults to `100`. Setting it to `0` authorizes the candidate on the
first non-speech frame, so ASR completion can immediately start its draft
request. Neither value changes Silero's probability smoothing, its 350 ms
endpoint hold, or Smart Turn. This setting counts non-speech frames after the
VAD classification; it does not count from the last acoustic speech sample.

ASR preparation and draft registration are separate operations. Early ASR
captures the complete utterance at an immutable PCM cutoff, but cannot upload
text, start TTS, or publish output. Any new speech or invalidated audio track
revokes that result. At the candidate gate, registration acquires a waitable
identity before yielding, so an endpoint cannot overtake its eventual PUT.
If recognition is still running, both registration and the endpoint join it.
Only one native decode can run at a time. A native call already executing in a
worker thread cannot be cancelled safely. If speech resumes and another pause
arrives before that call returns, the newer speculative snapshot is dropped
instead of queued; the endpoint keeps the complete PCM and uses authoritative
ASR once. Stale results cannot upload.
The production draft decoder shares the authoritative SenseVoice recognizer
instead of allocating a second ONNX model. Authoritative decoding has priority;
new speculative snapshots fall back when that shared model is busy rather than
queueing ahead of the endpoint. Smart Turn computes the same normalized Whisper
80-bin log-mel input with a small NumPy extractor, so the Companion does not load
the Transformers/PyTorch runtime solely for endpoint classification.
Speculative decoding stops when the buffered PCM exceeds the existing
`FISH_MEMORY_COMPANION_VAD_MAX_SPEECH_SECONDS` budget. The check reads the
buffer size without copying it; the authoritative endpoint still retains and
decodes the complete utterance, so this circuit breaker cannot truncate ASR.
Invalidation before the PUT starts also releases its registration waiter, so
new speech cannot leave finalization waiting for an obsolete native decode.

`AnswerContextLoader.load_for_draft` loads the current answer context once.
`ParallelAnswerGenerationPipeline.prepare_initial` detaches its messages,
Recall conversation, image metadata, Agent target, and task candidates from the
caller's mutable objects. The input hash records that prepared request for
diagnostics. Commit does not recompute a hash against refreshed context.

The server can adopt a revision while its context loader or model calls are
still running. It waits for the existing preparation handle outside the draft
registry lock. An adopting revision is protected from speculative TTL expiry;
resumption, replacement, or shutdown still revokes it.

## Computation and commit

| Route | Preparation | Commit |
| --- | --- | --- |
| Recall | R decision, query repair if needed, deep Recall, answer continuation | Publish buffered output and forward remaining output |
| Web | W decision, search, answer continuation | Publish buffered output and forward remaining output |
| Home query | H decision, catalog snapshot, scope planning, query result, answer continuation | Record the already prepared query outcome |
| Home control | H decision, catalog snapshot, scope planning, target selection, fixed actions | Validate current capabilities and execute those actions |
| Task | T decision, frozen target/candidate selection, interaction classification when needed | Persist task/run/steer or consume an interaction with a transaction guard |
| Exact Agent reply | Classify the frozen interaction or run result, prepare any required ordinary answer | Validate the same subject and apply its prepared disposition |

Preparation may call read services and LLMs. It does not create a formal user
message, publish speech, dispatch an Agent run, consume an interaction, or
control a device. Derived Recall caches may still be populated by the existing
Recall implementation.

The selected lane alone owns output. `PromotableAnswerSink` buffers it before
commit, then forwards the same buffer and subsequent output to the public sink.
Task acknowledgement is published only after task acceptance has been recorded.
Preparation is not a requirement to wait for a complete answer: when commit
arrives during a search, its selected prefix can be published while that same
search and continuation finish.

The first playable TTS segment can be synthesized during preparation. Public
playback adopts that same provider and its already-submitted text. If the
provider is still starting or the first text send is in flight, promotion
joins that existing operation. A healthy provider with no PCM yet can still
be adopted; its first audio packet is forwarded when it arrives. Subsequent
text is sent only after the original send finishes, without resubmitting the
prefix. Commit synchronously transfers the matching TTS generation's lifecycle
to public delivery. A subsequent speculative candidate cannot replace that
provider while the committed answer still owns it. Its ASR and LLM preparation
continue, but optional TTS startup is skipped until the committed provider is
explicitly cancelled or retired. Repeating the same committed key/revision
joins its existing startup. This priority check never blocks actual cancellation.
If provider startup or private SSE subscription completes
after the server has closed its private entry, an empty private EOF no longer
retires that provider. This handoff does not authorize playback; promotion
still validates the text, voice configuration, and current playback owner.
The public answer stream may begin with an empty snapshot and deliver its
first text in a delta. Session accumulates public text and flush offsets before
attempting the same TTS handoff for either event form. If the public prefix or
flush boundary has not caught up to private output, `DraftTtsPending` preserves
that provider and lets the event reader continue. A later delta, snapshot, or
segment retries adoption. Actual prefix conflicts or provider failures retain
the ordinary fallback, which receives all accumulated public text once.
Buffered audio retains its actual sentence ranges even when only part
of the submitted text has produced audio. A new TTS request is needed only
when the prepared provider or its ownership/configuration/range is invalid,
not because it has not produced its first packet yet.

An exact reply which requires no ordinary assistant answer uses
`PreparedReplyTurn`. It shares draft ownership/readiness tracking without
starting unused R/W/H/T calls. Replies which still require a conversational
answer retain their prepared reply decision alongside `PreparedInitialAnswer`.

Streaming LLM calls consume the HTTP response tail after the SSE `[DONE]` marker
so the next request can reuse the connection. This optional drain has a 100 ms
deadline and cannot alter already delivered content, usage, or completion state.
If the tail stalls or fails, only that connection is forfeited. Cancellation and
explicit early stream closure close the response immediately without draining.
The request payload, model settings, and first-content delivery stay unchanged.

## Cancellation and consistency

`PreparedInitialAnswer` owns initial lanes, the selected publisher, downstream
preparation, and the eventual commit worker. Cancellation revokes the output
sink before stopping workers, so even a late provider result cannot publish.
Cancelled uncommitted Home and Task plans have no mutations to undo.

Commit checks revision ownership, durable message identity, and operation
state. Home rechecks the selected device's availability and capabilities. Task
and exact reply commits retain their transaction and subject guards. These
checks cannot select a replacement target or rerun an LLM to reinterpret the
utterance. A failed guard ends that operation instead.

If preparation was unavailable or failed, the existing ordinary answer path
remains the fallback. It uses the same preparation/commit implementation. A
superseded revision is a conflict, not permission to submit its old utterance
through fallback.

## Extension points

- `answer/prepared_answer.py`: common preparation contract and owned answer
  lifecycle.
- `answer/parallel_generation.py`: lane selection, downstream computation, and
  the route-specific commit boundary.
- `answer/prepared_reply.py`: completed exact replies with no assistant output.
- `home/preparation.py` and `home/service.py`: immutable Home data and device
  execution.
- `agent/routing.py` and `agent/replies.py`: prepared task/reply decisions and
  transactional commits.
- `answer/voice_drafts.py`: revision registry, expiry, event observation, and
  adoption while preparation is in flight.
- `companion/draft_tts.py`: ownership of provider startup, first text submission,
  and handoff to public speech.
- `companion/draft_pcm.py`: bounded private PCM and ordered playback handoff.

A new route should expose a preparation result containing everything its
commit needs. Keep reversible computation in preparation; keep actual
mutations and current-state transaction guards in commit. Extend the captured
input for future multi-frame perception at the preparation boundary, rather
than adding another capture during publication.

## Verification

Tests use gated fake providers and stores to establish ordering: expensive
work starts before commit, commit adopts it without another request, private
output stays private, and cancelled work cannot publish or execute actions.
The Home and Agent tests also change real-time state between preparation and
commit to verify that frozen plans cannot silently expand or consume a changed
subject. These are lifecycle checks, not measured end-to-end latency gains.

For adopted drafts, `voice_draft.preparation` preserves the private performance
steps and their origin relative to the formal message. It does not move the
formal `total` or first-public-output clock back to the speculative start.

## Latency diagnosis

Use the persisted user-message `voice_draft.promoted` step to count prepared
answer adoption. Companion draft-provider and PCM promotion counters describe
separate stages; a prepared answer can be adopted without prepared audio.
Measure endpoint-to-playback separately from model first-content and TTS
readiness. The processor's `speech_end` marks its VAD endpoint, not the exact
last voiced audio sample.

The runtime trace includes bounded, content-free draft lifecycle events even
when audio diagnostics are disabled. Correlate their `runId`, `turnId`,
revision, and speech generation with candidate decisions and ASR reuse
outcomes before changing endpoint thresholds. A missing started attempt alone
does not establish whether silence was never detected or a candidate was
skipped. These events use the shared draft metadata privacy filter and do not
retain transcript text or PCM.

For the early-ASR path, `voice_draft_asr_prepare_triggered` and
`voice_draft_preparation_started` precede the 100 ms candidate gate.
`voice_draft_started` marks candidate authorization, while
`voice_draft_request_started` marks the actual draft PUT after recognition.
A transcript-ready event may therefore precede candidate authorization. These
are distinct clocks; the PUT timestamp does not identify the remote model's
inference start.
