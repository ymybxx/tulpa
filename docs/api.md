# API

[README](../README.md) | [Architecture](architecture.md)

The tulpa desktop app starts this local service on a free port on `127.0.0.1`
and talks to it over HTTP, SSE, and WebSocket. JSON uses camelCase response
fields. Errors use:

```json
{"error":{"code":"machine_readable_code","message":"description"}}
```

Stable runtime endpoints live under `/v1`. Local administration and
diagnostics live under `/internal` and are intentionally omitted from the
public OpenAPI schema. Legacy unversioned API paths are not registered.

## Authentication

All HTTP, SSE, and WebSocket endpoints require the service API key except
`/health` and `/ready`. The desktop app generates the key, stores it with its
other secrets, and passes it to the service as `FISH_MEMORY_API_KEY`.

HTTP and SSE clients send `Authorization: Bearer $FISH_MEMORY_API_KEY`.
WebSocket clients send the same header during the opening handshake.

The examples below use `API=http://127.0.0.1:<port>`; in development the
service port appears in the desktop app's log (`Uvicorn running on …`).

## Fish and Soul

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/fishes` | List active Fish identities |
| `POST` | `/v1/fishes` | Create a Fish with optional `role`, `setting`, and `name` |
| `DELETE` | `/v1/fishes/{fishId}` | Soft-delete a Fish from normal listings |
| `GET` | `/internal/fishes/{fishId}/management` | Soul, latest snapshots, entities, and assistant binding |
| `GET` | `/v1/fishes/{fishId}/soul` | Read the configured Soul fields |
| `PATCH` | `/v1/fishes/{fishId}/soul` | Replace Soul fields |
| `PUT` | `/v1/fishes/{fishId}/assistant` | Bind the native companion to this Fish |
| `GET` | `/v1/fishes/assistant` | Read the current companion binding |
| `DELETE` | `/v1/fishes/assistant` | Remove the binding |

Create example:

```bash
curl -sS -X POST $API/v1/fishes \
  -H "Authorization: Bearer $FISH_MEMORY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"role":"research partner","setting":"brief and precise","name":"Milo"}'
```

Soul text is treated as descriptive preference inside a fixed system framework;
it cannot grant tools or override system policy.

## Sessions and messages

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/fishes/{fishId}/sessions` | Create an empty Session |
| `GET` | `/v1/fishes/{fishId}/sessions` | List Sessions newest first |
| `GET` | `/v1/fishes/{fishId}/messages` | Read visible messages in a Session |

`GET /messages` accepts `sessionId`, `limit` (1-500), `before`, and `after`.
Without `sessionId`, Session-aware endpoints use the most recently created
Session. Cursor responses include `hasBefore`, `hasAfter`, `minSequence`, and
`maxSequence` when `before` or `after` is supplied.

Dynamic memory and recent history are Session-scoped. Static memory, Facts,
entities, relation Recall, and Soul remain Fish-scoped.

Snapshot management output is Item-based. Each current Item exposes its
self-contained Summary as `text`, `lastSupportedAt`, and `sourceFactCount` (the
current support count). Internal stable Item IDs, lineage, categories, status,
and deprecated fields are not part of the API. Dynamic max-age affects Answer's
always-present Working Memory but does not remove the Item from Relation Recall.

## Ask and streaming Answer

Create a user message:

```bash
curl -sS -X POST "$API/v1/fishes/$FISH_ID/ask" \
  -H "Authorization: Bearer $FISH_MEMORY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"content":"Remember that I meet Alice on Tuesday mornings."}'
```

The optional `sessionId` selects a Session. Content must be nonblank and no more
than 32,000 characters. The response contains `messageId`, `sessionId`,
`sequence`, `status`, and `createdAt`.

Fish created after the normalized Snapshot cutover use storage version 2.
Asking a version-1 Fish returns HTTP 409 with code
`legacy_fish_read_only` before a Session or Message is created. There is no
on-read conversion endpoint; recreate imported benchmark history or create a
new Fish.

Open the SSE stream:

```bash
curl -N \
  -H "Authorization: Bearer $FISH_MEMORY_API_KEY" \
  "$API/v1/fishes/$FISH_ID/answer?sessionId=$SESSION_ID"
```

Use `afterSequence` to wait for an answer newer than an observed sequence. SSE
events are:

- `answer.snapshot`: current complete buffer for a newly observed answer;
- `answer.delta`: only text appended since the previous event;
- `answer.completed`: terminal success;
- `answer.interrupted`: playback was interrupted;
- `answer.failed`: terminal generation failure.

Snapshot and terminal data include `messageId`, `sequence`, `status`/`content`,
and conversation `end` where applicable. Delta data includes `startOffset` and
`endOffset`. Event IDs use `{messageId}:{endOffset}`.

### Playback progress and interruption

```bash
curl -sS -X POST "$API/v1/fishes/$FISH_ID/answer/progress" \
  -H "Authorization: Bearer $FISH_MEMORY_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"messageId":"'$ANSWER_ID'","consumedOffset":42}'
```

Use the same body with `/answer/interrupt` to freeze the visible answer at the
reported offset. Text clients report displayed-character progress; voice clients
report the last completed TTS segment. Offsets are Unicode string offsets in the
stored answer content.

## Recall inspection

`POST /v1/fishes/{fishId}/memories/recall` exposes ranked memory Recall
independently from Answer:

```json
{"query":"Who is my manager?","debug":false}
```

This endpoint always runs the single Planner + Recall flow (`r=1`); it has no
`strategy` request field and returns no strategy fields. The Planner exposes
only `sourceRole`, `entityHints`, and `time.start`/`time.end`. No bounds mean no
time Scope; bounded Recall accepts evidence mentioned in the range **or** whose
event occurred in it. All returned Units remain semantic-ranked. If planning
degrades, the response exposes `degraded`; set `debug` to `true` for the trace,
plan error, and stage timings. Query text is limited to 6,000 characters.

Legacy Store time-field and ordering APIs remain available only outside the
production Planner and Recall path.

Relation evidence is the current projection of a stable Snapshot Item. Its
Summary and current supporting Facts form one Memory Unit. A one-support Item is
valid; a partially eligible multi-support Item degrades to its eligible
standalone Facts rather than exposing a Summary supported by out-of-scope
members. The public response and Answer context do not reveal Item IDs or ranks.

## Memory and performance inspection

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/internal/fishes/{fishId}/background-progress?sessionId=...` | Stable message processing progress |
| `GET` | `/internal/fishes/{fishId}/messages/{messageId}/performance` | Ordered server-side timing, prompt, output, and error steps |
| `POST` | `/internal/fishes/{fishId}/messages/{messageId}/tts-performance` | Replace browser/native TTS metrics for the message |
| `GET` | `/internal/fishes/{fishId}/entities/{entityId}/facts` | Paginate Facts linked to an entity |
| `GET` | `/v1/fishes/{fishId}/voice` | Resolve the Fish voice against the active local/DashScope provider |
| `PATCH` | `/v1/fishes/{fishId}/voice` | Persist the Fish model, voice ID, and voice kind |

Performance step timing uses microseconds relative to the message operation's
origin. Large JSON prompt/response payloads are returned as structured JSON for
inspection.

## Voice WebSockets

- `GET /v1/kws/status`: optional KWS availability/download state.
- `WS /v1/kws/stream`: browser PCM keyword detection.
- `GET /v1/tts/models`: supported provider models and voices.
- `GET /internal/tts/custom-voices`: list cloned voices from the configured DashScope account.
- `POST /internal/tts/custom-voices?targetModel=...`: create a cloned voice from a raw WAV body.
- `DELETE /internal/tts/custom-voices/{voiceId}`: delete a cloned voice; `force=true` clears Fish references.
- `WS /v1/tts/stream`: appendable streaming TTS tasks and binary PCM output.

These endpoints are optional. A base installation returns KWS as unavailable and
does not need native audio dependencies. Use the built-in voice debug page or the
native companion as the protocol reference while the WebSocket contract remains
pre-1.0.

See [Configuration](configuration.md) and [Troubleshooting](troubleshooting.md).
