# Configuration

[README](../README.md) | [Architecture](architecture.md)

The tulpa desktop app keeps its settings in its own data directory and starts
the local service with environment variables built from them
(`desktop/pet/electron/runtime-manager.ts`). Users change these through
Settings; this page documents the variables for development and debugging.
The service also reads `.env` from its working directory, and process
environment values take precedence. Do not commit `.env`; it may contain
provider credentials.

## API authentication

One API key protects HTTP, SSE, and WebSocket access. If
`FISH_MEMORY_API_KEY` is unset, the service generates a key on first startup and
stores it with owner-only permissions. The default file is
`.cache/fish-memory/api-key`.

| Variable | Default | Description |
| --- | --- | --- |
| `FISH_MEMORY_API_KEY` | generated | Explicit service key for managed deployments |
| `FISH_MEMORY_API_KEY_FILE` | `.cache/fish-memory/api-key` | Persistent generated-key file |

The desktop app always passes an explicit key. Changing
`FISH_MEMORY_API_KEY` requires restarting the service. Use HTTPS whenever the
service is exposed beyond localhost.

## Required provider settings

| Variable | Default | Description |
| --- | --- | --- |
| `FISH_MEMORY_LLM_PROVIDER` | required; currently `openai` | OpenAI-compatible provider type |
| `FISH_MEMORY_LLM_API_KEY` | required | Chat credential |
| `FISH_MEMORY_LLM_MODEL` | required | Chat model used by Answer, Fact, snapshots, entities, planner, and judge |
| `FISH_MEMORY_LLM_BASE_URL` | required | OpenAI-compatible `/v1` base URL |
| `FISH_MEMORY_ANSWER_ENABLE_SEARCH` | `false` | Allow the Answer decision to request a forced provider web-search stage |

Web search is only used after the first answer says it is needed. Fact,
snapshot, entity, and Recall planner calls never enable provider search.

## Embeddings

Embeddings are configured independently from chat. The default runs locally on
CPU and downloads the ONNX model on first startup.

| Variable | Default | Description |
| --- | --- | --- |
| `FISH_MEMORY_EMBEDDING_PROVIDER` | `local` | `local` or `openai-compatible` |
| `FISH_MEMORY_EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Local Hugging Face repository or remote model name |
| `FISH_MEMORY_EMBEDDING_DIMENSIONS` | `384` | Expected model output dimension, from 1 to 2000 |
| `FISH_MEMORY_EMBEDDING_BATCH_SIZE` | `32` | Maximum texts encoded per local or remote batch |
| `FISH_MEMORY_EMBEDDING_LOCAL_MODEL_DIR` | automatic Hugging Face cache | Optional pre-downloaded local model directory |
| `FISH_MEMORY_EMBEDDING_LOCAL_NUM_THREADS` | `2` | ONNX Runtime CPU inference threads |
| `FISH_MEMORY_EMBEDDING_LOCAL_MAX_TOKENS` | `512` | Maximum tokens per local embedding input |
| `FISH_MEMORY_EMBEDDING_API_KEY` | unset | Dedicated remote embedding credential; never falls back to the chat key |
| `FISH_MEMORY_EMBEDDING_BASE_URL` | unset | OpenAI-compatible remote embedding `/v1` base URL |

Remote mode uses the OpenAI-compatible embeddings request format. Providers
with a different wire protocol require a small client adapter rather than only
configuration changes.

DashScope remote embedding example:

```dotenv
FISH_MEMORY_EMBEDDING_PROVIDER=openai-compatible
FISH_MEMORY_EMBEDDING_API_KEY=your_DashScope_API_key
FISH_MEMORY_EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
FISH_MEMORY_EMBEDDING_MODEL=text-embedding-v4
FISH_MEMORY_EMBEDDING_DIMENSIONS=1024
```

Each Fish records its embedding provider, model, and dimensions. Different Fish
can therefore use different dimensions in one database. Once a Fish has Facts or
messages, the API and background processors reject a different embedding space
with an `embedding_space_mismatch` error; changing it requires an explicit full
rebuild so vectors from different spaces are never mixed.

The embedding configuration is process-wide. One running service instance serves
the Fish matching its active embedding space; the database format itself allows
Fish created under other dimensions to coexist without mixing their vectors.

## Server and database

| Variable | Default | Description |
| --- | --- | --- |
| `FISH_MEMORY_DATABASE_URL` | `pg0://fish-memory` | Embedded pg0 name or PostgreSQL URL |
| `FISH_MEMORY_HOST` | `127.0.0.1` | HTTP bind address |
| `FISH_MEMORY_PORT` | `8000` | HTTP port |
| `FISH_MEMORY_SYSTEM_PROMPT` | built-in interaction rules | Global answer policy before per-Fish Soul |

The built-in policy covers interruption handling, current-user input, and memory
truthfulness. Identity, personality, relationship, and speaking style come from
the Fish's Soul configuration; generic assistant traits are only defaults for
empty role or setting fields. The current Soul takes precedence over old identity
memories, but cannot change host interaction rules, output protocols, tool
permissions, or actual capabilities. Chat and plugin speech use the same Soul
renderer. An explicit `FISH_MEMORY_SYSTEM_PROMPT` override is still used verbatim
and should keep character-specific instructions in Soul to avoid conflicts.

Soul values are serialized as JSON inside `character_config`, with angle brackets
escaped so field content cannot close that section or add unescaped headings.
Only character-related requirements apply; instructions inside those values to
override host rules or authorization are invalid. This reduces prompt ambiguity,
but serialization is not a security sandbox: output validation and tool execution
checks must remain in code, and plugins are executable code rather than untrusted
chat text.

Examples:

```dotenv
FISH_MEMORY_DATABASE_URL=pg0://fish-memory
```

```dotenv
FISH_MEMORY_DATABASE_URL=postgresql://user:password@127.0.0.1:5432/fish_memory
```

The external database must allow `CREATE EXTENSION vector`. Migrations run at
startup.

## Answer and context

| Variable | Default | Description |
| --- | ---: | --- |
| `FISH_MEMORY_ANSWER_HISTORY_MESSAGE_LIMIT` | `200` | Maximum recent messages loaded from the current Session |
| `FISH_MEMORY_HISTORY_TOKEN_LIMIT` | `20000` | Trigger per-session History expiration and prefix compaction; local token estimate |
| `FISH_MEMORY_HISTORY_TOKEN_RETAIN` | half of limit (`10000`) | Legacy message-window retention; unified History always keeps an uncompressed suffix below half of limit |
| `FISH_MEMORY_RECALL_CONTEXT_TOKEN_BUDGET` | `50000` | Budget for recalled source messages and degraded Fact items |

History trimming keeps complete messages and removes old messages in batches so
the stable prefix changes less often. It does not recalculate provider tokenizers
on every request. Recalled source messages that do not fit degrade to compact
Facts rather than being partially cut.

## Home Assistant

Home Assistant control is optional and disabled by default. When enabled, an
independent router runs beside the normal Answer pipeline: explicit common
commands use a local fast path, while ambiguous requests use one compact LLM
call constrained to the current Home Assistant entity catalog. A Home
Assistant outage never blocks normal chat.

| Variable | Default | Description |
| --- | --- | --- |
| `FISH_MEMORY_HOME_ASSISTANT_ENABLED` | `false` | Enable the independent home router |
| `FISH_MEMORY_HOME_ASSISTANT_URL` | `http://127.0.0.1:8123` | Home Assistant base URL |
| `FISH_MEMORY_HOME_ASSISTANT_TOKEN` | unset | Required long-lived access token when enabled |
| `FISH_MEMORY_HOME_ROUTER_TIMEOUT_SECONDS` | `5` | Compact Router LLM timeout |
| `FISH_MEMORY_HOME_ROUTER_CONCURRENCY` | `8` | Maximum concurrent home routing tasks |

Create the token from the Home Assistant user profile. It stays server-side
and is never logged, returned to clients, persisted in action rows, or sent
to an LLM. Matching uses HA entity, device, area, state, and service metadata,
so Xiaomi Home, direct pairings, Matter, Zigbee, and other integrations share
the same path.

Action progress is independent from chat and memory. The chat page renders it
under the source user message, while the native companion speaks only state
query results, failures, and ambiguous targets after the current answer audio.
Ordinary successful controls are not spoken twice. External clients can use:

- `GET /v1/fishes/{fish_id}/home-actions` for refresh and diagnostics;
- `GET /v1/fishes/{fish_id}/home-actions/events` for resumable SSE updates.

Each action has a stable `sequence`, while every status change advances
`eventSequence`. Resume the event stream with `afterEventSequence`; this keeps
`routing`, `executing`, and terminal updates observable even though they update
the same action row.

Set all three variables to enable the integration:

```dotenv
FISH_MEMORY_HOME_ASSISTANT_ENABLED=true
FISH_MEMORY_HOME_ASSISTANT_URL=http://127.0.0.1:8123
FISH_MEMORY_HOME_ASSISTANT_TOKEN=your_long_lived_access_token
```

## Background memory processing

| Variable | Default | Description |
| --- | ---: | --- |
| `FISH_MEMORY_BACKGROUND_POLL_INTERVAL_SECONDS` | `1.0` | Idle processor poll interval |
| `FISH_MEMORY_BACKGROUND_LLM_CONCURRENCY` | `1` | Per-process LLM concurrency shared by each processor class |
| `FISH_MEMORY_BACKGROUND_EMBEDDING_CONCURRENCY` | `4` | Concurrent embedding calls during Fact processing |
| `FISH_MEMORY_FACT_EXTRACTION_MAX_RETRIES` | `3` | Terminal retry limit for Fact and snapshot work |
| `FISH_MEMORY_FACT_LLM_TIMEOUT_SECONDS` | `120` | Timeout for one Fact extraction call |
| `FISH_MEMORY_FACT_BATCH_SIZE` | `8` | Stable messages leased per Fact processor batch |
| `FISH_MEMORY_SNAPSHOT_BATCH_SIZE` | `100` | Facts consumed per Dynamic/Static update |
| `FISH_MEMORY_DYNAMIC_SNAPSHOT_MAX_AGE_DAYS` | `7` | Hard age limit for current Dynamic items |

Increase concurrency only after checking provider rate limits and database
capacity. Dynamic and Static processors are independent, but each requires Fact
progress before consuming a batch.

## Native companion

Install this optional layer with:

```bash
uv sync --dev --extra voice
```

| Variable | Default | Description |
| --- | --- | --- |
| `FISH_MEMORY_API_URL` | `http://127.0.0.1:8000` | API used by the native companion |
| `FISH_MEMORY_ASSISTANT_POLL_INTERVAL_SECONDS` | `1.0` | Poll interval while no assistant Fish is bound |
| `FISH_MEMORY_TTS_PROVIDER` | `auto` | `auto`, `dashscope`, or `local` |
| `FISH_MEMORY_TTS_API_KEY` | unset | Independent DashScope TTS credential; never falls back to the LLM key |
| `FISH_MEMORY_TTS_WEBSOCKET_URL` | DashScope Beijing endpoint | Provider WebSocket URL |
| `FISH_MEMORY_TTS_CUSTOMIZATION_URL` | DashScope customization endpoint | Voice cloning/list/delete endpoint; normally unchanged |
| `FISH_MEMORY_TTS_UPLOAD_URL` | DashScope temporary upload endpoint | Upload endpoint used by recorded voice samples; normally unchanged |
| `FISH_MEMORY_TTS_LOCAL_MODEL_DIR` | automatic cache path | Local Chinese-English Matcha model directory |
| `FISH_MEMORY_TTS_LOCAL_NUM_THREADS` | `2` | Local TTS CPU inference threads |
| `FISH_MEMORY_WAKE_WORD` | `小鱼` | Default wake word when the selected Fish has no override |
| `FISH_MEMORY_KWS_MODEL_DIR` | automatic cache path | Local sherpa-onnx KWS model |
| `FISH_MEMORY_KWS_CHUNK_SIZE` | `8` | KWS decode chunk, accepted values 8 or 16 |
| `FISH_MEMORY_KWS_NUM_THREADS` | `1` | KWS inference threads |
| `FISH_MEMORY_KWS_SCORE` | `1.0` | Keyword score |
| `FISH_MEMORY_KWS_THRESHOLD` | `0.25` | Keyword trigger threshold |
| `FISH_MEMORY_ASR_MODEL_DIR` | automatic cache path | Local SenseVoice model directory |
| `FISH_MEMORY_ASR_NUM_THREADS` | `2` | ASR inference threads |

With `auto`, an independent TTS key selects DashScope; without one, tulpa uses
the local `matcha-icefall-zh-en` model. The TTS implementation never reads the LLM key.
The first local synthesis downloads and warms the missing TTS model under
`.cache/fish-memory/tts`. Local mode provides one built-in Chinese-English voice;
model and voice selection remain available in DashScope mode. Install the `voice`
extra to enable local audio.

The character page persists model and voice selection per Fish. In DashScope mode
it also records a 5–20 second WAV sample, uploads it through the
temporary DashScope upload flow, and creates a model-bound cloned voice. Deleting a
voice that is still referenced requires confirmation; confirmed deletion clears
those Fish selections so they deterministically fall back to the provider default.

The first voice startup also downloads missing KWS, ASR, and VAD models under
`.cache/fish-memory`, and resolves Smart Turn through the configured path or the
Hugging Face cache. A base-only memory install does not load native voice libraries.

## Audio capture and turn detection

| Variable | Default | Description |
| --- | ---: | --- |
| `FISH_MEMORY_COMPANION_AUDIO_BLOCK_SIZE` | `160` | 16 kHz capture samples per audio callback block |
| `FISH_MEMORY_COMPANION_AEC_DELAY_MODE` | `auto` | AEC reference alignment mode: `auto` or `fixed` |
| `FISH_MEMORY_COMPANION_AEC_DELAY_MS` | `50` | Fixed/fallback render-reference delay |
| `FISH_MEMORY_COMPANION_VAD_MODE` | `2` | WebRTC VAD aggressiveness used by the fast gate |
| `FISH_MEMORY_COMPANION_VAD_MODEL_DIR` | automatic cache path | Silero VAD model directory |
| `FISH_MEMORY_COMPANION_VAD_THRESHOLD` | `0.25` | Silero speech probability threshold |
| `FISH_MEMORY_COMPANION_VAD_MIN_SILENCE_MS` | `350` | Stable silence required to close an utterance |
| `FISH_MEMORY_COMPANION_VAD_MIN_SPEECH_MS` | `80` | Stable speech required to open an utterance |
| `FISH_MEMORY_COMPANION_VAD_WINDOW_SIZE` | `512` | Silero window samples at 16 kHz |
| `FISH_MEMORY_COMPANION_VAD_MAX_SPEECH_SECONDS` | `20.0` | Maximum buffered utterance length |
| `FISH_MEMORY_COMPANION_VAD_NUM_THREADS` | `1` | Silero inference threads |
| `FISH_MEMORY_COMPANION_UTTERANCE_COMMIT_DELAY_MS` | `0` | Optional delay after endpoint before Ask |
| `FISH_MEMORY_COMPANION_SMART_TURN_ENABLED` | `true` | Use semantic completeness after Silero's soft endpoint |
| `FISH_MEMORY_COMPANION_SMART_TURN_MODEL_PATH` | Hugging Face cache | Optional local Smart Turn v3.2 CPU ONNX path |
| `FISH_MEMORY_COMPANION_SMART_TURN_THRESHOLD` | `0.5` | Strict completeness probability threshold |
| `FISH_MEMORY_COMPANION_SMART_TURN_REOPEN_MS` | `800` | Grace window when the utterance seems incomplete |
| `FISH_MEMORY_COMPANION_SMART_TURN_TIMEOUT_MS` | `500` | Per-inference timeout before session fallback to Silero |
| `FISH_MEMORY_COMPANION_SMART_TURN_NUM_THREADS` | `1` | Smart Turn ONNX inference threads |
| `FISH_MEMORY_COMPANION_VOICE_DRAFT_ENABLED` | `true` | Pre-generate reversible ASR, Answer, and first-segment TTS work during the authoritative endpoint wait; failures fall back to the ordinary path |
| `FISH_MEMORY_COMPANION_VOICE_DRAFT_CANDIDATE_SILENCE_MS` | `100` | Continuous VAD non-speech before authorizing an LLM draft; ASR prepares from the first non-speech frame. `0` removes this extra wait without changing endpoint detection |
| `FISH_MEMORY_OWNER_VOICE_ENABLED` | `false` | Enable local owner voice admission after enrollment; missing or incompatible profiles block microphone admission while enabled |
| `FISH_MEMORY_OWNER_VOICE_DIR` | `.cache/fish-memory/owner-voice` | Local speaker profile and model directory; desktop builds use their application data directory |
| `FISH_MEMORY_OWNER_VOICE_THRESHOLD` | `0.5` | Experimental speaker similarity threshold, not an authentication guarantee |
| `FISH_MEMORY_COMPANION_EARLY_INTERRUPT_SNAPSHOT_MS` | `300` | Acoustic speech duration before the one-shot SenseVoice snapshot |
| `FISH_MEMORY_COMPANION_EARLY_INTERRUPT_MIN_CHARS` | `2` | Minimum compact characters required for early cancellation |
| `FISH_MEMORY_COMPANION_SINGLE_CHAR_MIN_SPEECH_MS` | `200` | Estimated speech required to submit a one-character SenseVoice final; punctuation-only finals are always rejected |

`FISH_MEMORY_COMPANION_VAD_START_FRAMES` and
`FISH_MEMORY_COMPANION_VAD_END_SILENCE_MS` remain accepted for compatibility;
the synchronized companion path uses Silero duration settings for turn closure.

See [Voice answer preparation](voice-preparation.md) for draft ownership and
[desktop owner voice recognition](../desktop/pet/README.md#local-owner-voice-recognition-experimental)
for enrollment, cancellation, and profile compatibility.

## Local audio diagnostics

Diagnostics are off by default and must not be enabled in privacy-sensitive
deployments without user consent.

| Variable | Default | Description |
| --- | ---: | --- |
| `FISH_MEMORY_AUDIO_DIAGNOSTICS_ENABLED` | `false` | Persist local diagnostic sessions |
| `FISH_MEMORY_AUDIO_DIAGNOSTICS_DIR` | `.cache/fish-memory/audio-diagnostics` | Local output directory |
| `FISH_MEMORY_AUDIO_DIAGNOSTICS_RETENTION_DAYS` | `7` | Automatic retention period |
| `FISH_MEMORY_AUDIO_DIAGNOSTICS_PRE_SECONDS` | `5.0` | Audio retained before a trigger |
| `FISH_MEMORY_AUDIO_DIAGNOSTICS_POST_SECONDS` | `5.0` | Audio retained after a trigger |

Diagnostics can include microphone capture, render reference, AEC-cleaned audio,
question audio, VAD/ASR events, and timing metadata. They are not uploaded or
stored in the business database.

See [Troubleshooting](troubleshooting.md) before changing audio thresholds.
