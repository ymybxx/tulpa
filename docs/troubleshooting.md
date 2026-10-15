# Troubleshooting

[README](../README.md) | [Configuration](configuration.md)

## Application stays at startup

Startup resolves the database, opens a pool, runs migrations, starts optional
KWS preparation, creates provider clients, and starts background supervisors.

1. Check PostgreSQL connectivity with `psql` when using an external URL.
2. Confirm pgvector is installed and the database user can create extensions.
3. Check that only one process owns the configured port.
The embedded pg0 database is persistent. Quitting tulpa does not delete its
instance, and another pg0 owner may keep PostgreSQL running.

## SOCKS proxy import error

The base dependency uses `httpx[socks]`, so a fresh locked install includes
`socksio`. If the environment predates the lock update:

```bash
uv sync --dev
```

Do not install ad hoc packages into `.venv`; resync the declared environment.

## Fact progress appears stuck

The newest message is intentionally not eligible until a newer message makes it
stable. A displayed target may therefore be one below raw message count.

For a genuine stall:

1. open the message performance view and inspect `llm.fact_extraction`;
2. inspect `background_task_progress.last_error` through the UI/API;
3. verify provider rate limits and response JSON validity;
4. check `FISH_MEMORY_FACT_LLM_TIMEOUT_SECONDS` and retry count;
5. restart the process to resume terminally unfinished leased work.

After the retry limit, a failed message advances the processor watermark and
records the error. It should not block later messages forever.

## Dynamic or Static progress is zero

Snapshots do not consume partial Fact progress. They wait until the eligible Fact
range is caught up, then process a larger batch. Dynamic and Static processors
run independently. Dynamic progress is Session-scoped; make sure the UI/API is
showing the intended `sessionId`.

Each nonempty Fact batch runs a first-stage pre-update. Dynamic and Static use
independent eligibility prompts; the Static prompt must reject short-term Facts
even when existing Items contain historical noise. Direct Creates need no second
call. Every proposed Update gets a separate confirmation call with the target
Item's current support Facts and recent context. Static confirmation can return
an explicit Ignore for a misrouted short-term Fact. Inspect these performance
steps in order:

- `llm.dynamic_snapshot.preupdate` or `llm.static_profile.preupdate`;
- one `*.confirm` step for each marked Update;
- `snapshot.aggregate` and `snapshot.persist`.

One invalid confirmation fails the whole batch closed. No Header version,
Item, or watermark is partially advanced. Confirmations share the configured
background LLM semaphore, so many marked Updates may complete out of order but
commit together. A stale Header version also discards the complete model result
and retries from fresh Items.

Pre-update has three operation-level normalizations. An explicit `[]` with
otherwise-valid fields has no selectable evidence, so the service drops it and
records `dropped_operations`. Repeated indexes inside one otherwise-valid
operation are deduplicated in first-seen order and recorded as `normalizations`;
the selected Fact set is unchanged. If several operations claim the same Fact
index, all operations touching that index are dropped and recorded with
`reason=ambiguous_source_fact_indexes`; unrelated operations continue. The Fact
itself remains stored and directly recallable. Missing/null/non-list indexes,
extra fields, and bad targets/ranges still fail the batch closed. There is no
repair LLM call.

Malformed LLM-derived pre-update, Entity-resolution, or confirmation output is
retried. Once retries are exhausted, an empty mutation advances the affected
Snapshot watermark and a `*.derivation_skipped` performance step records the
stage. This is non-blocking because Facts remain the source of truth. Evidence
consistency, database, Entity merge, and persistence errors still write a
blocking task error and must be investigated.

Current Items live in `memory_snapshot_items`. Check `support_fact_ids` for the
current Summary and `lineage_fact_ids` for its append-only history. An Update may
remove current support but never removes lineage or invalidates the Fact.

## A Snapshot Summary changed to an unrelated topic

Compare the pre-update and confirmation diagnostics. The pre-update is allowed
to propose an incorrect topic match; the confirmation must return Create-only
when the new Fact does not continue the target Item. In that case the target
Item, its support set, Item version, and Full projection remain unchanged, while
the new topic receives a different Item.

If the wrong Item still changed, inspect:

1. the target `itemIndex` and its request-local ordering;
2. the confirmation's `supportFacts`, `newFacts`, and `removeSupportIndexes`;
3. `memory_snapshot_item_versions` for the exact mutation basis;
4. whether more than one aggregate revision was created for one batch.

Do not repair this by deleting a source Fact or adding a deprecated status.
Correct the pre-update/confirmation evidence contract instead.

## A Snapshot Summary omits an unresolved detail

A Summary must cover the answer-relevant propositions of every Fact in its
current support, not merely share a topic. Compare the Create or confirmation
prompt/output with `support_fact_ids`: objects, relations, negation, conditions,
states, unfinished actions, quantities, time, and order must remain represented.
If independent propositions cannot be kept in one self-contained Summary, the
correct mutation is separate Creates (or Create-only for a too-narrow proposed
Update), not a broad topic bucket. The underlying Facts stay directly
recallable; do not invalidate or delete them to hide an omission.

## Snapshot is current but Relation readiness is stuck

Readiness is Item-projection based. For every current Item, its
`memory_snapshot_item_embeddings` row must have the same Fish, Session (or NULL
for Static), Snapshot type, dimensions, and `item_version`. Relation progress
must also reach the current Header version. A missing or stale Full row keeps the
Fish unready even if the progress watermark already advanced; the projection
worker will reconcile it. Query-dependent Mask rows never block readiness.

Projection failures do not roll back an already committed Snapshot. Inspect
`dynamic_relations`/`static_relations` errors, embedding-space compatibility,
and the Item Summary hash. A Summary change regenerates Full and clears old
Item masks; a support-only change may reuse the Full vector while updating
projection metadata.

## Recall gives a wrong or empty answer

Use the message performance view and `/recall` endpoint to separate failures:

- no useful Fact exists: inspect Fact extraction prompt/output;
- useful Fact exists but was not returned: inspect semantic query, entity hints,
  time field, relation expansion, and ranks;
- evidence was returned but answer is wrong: inspect continuation prompt/output;
- time question is wrong: compare `occurredStart`/`occurredEnd` with
  `mentionedAt` and planner field `occurred`, `mentioned`, or `either`.

Facts with `invalidated_at` are excluded from Recall. Snapshot Item Updates do
not set it: removed support remains directly recallable as historical Fact
evidence, while the current Item Summary expands only its current support set.

## Recall query lost a known Entity or became unexpectedly narrow

Inspect the final Ask message's `answer.rquery_validation` and `recall.total`
steps. `recall.total` preserves `originalQuery`, `generatedQuery`, `finalQuery`,
and the repair outcome. A generated query that omitted a known original Entity
causes one query-only repair after the shallow answer's `answer.first_content`;
the repair is then revalidated. Failed/invalid repair or validation-store errors
must use the original query, and there is no repair loop. Different aliases of
the same Entity ID are equivalent and should not repair. Entity matching is a
validation guard only: it must not become a Recall scope filter.

If all known Entities were preserved but the query still lost a collection,
condition, count, negation, time, or ordering qualifier, diagnose the R-lane
prompt/output. That semantic case intentionally has no heuristic repair.

## Recall Context looks smaller or has hour-only times

Continuation evidence is compact `{"u":[...]}` transport. `t:"r"` marks a
Relation with `s`, `f`, `start`, and `end`; `t:"f"` marks a standalone Fact;
`x`, `m`, and optional `o` mean Fact text, mentioned hour, and occurred hour or
range. UTC-hour precision applies only to that LLM payload. SQL filtering,
sorting, and planner time scopes retain full precision.

Compare selected Unit IDs/order and Primary/Reserve expansion before changing
the token budget. Compression must not add lower-ranked Units. Reserve Relations
may send only direct anchors, but their `start`/`end` still span all current
support Facts. Original source-message Recall transport is no longer a
production branch; inspect Relation Units and standalone Facts instead.

## Voice models stay at 0%

The first native companion startup downloads KWS, ASR, and Silero VAD models.
Progress is logged by component. Confirm disk space, proxy reachability, and that
the cache path is writable. Partial downloads are resumed or replaced by the
model preparation layer; do not commit `.cache`.

Model preparation is skipped after a valid local model is present.

## TTS plays once and then fails

The native client connection and the provider task are separate:

- one client WebSocket can carry successive TTS tasks;
- a provider task is finished or cancelled per answer segment;
- a stale provider connection must be discarded and reconnected before reuse.

Inspect logs for provider timeout, `ConnectionClosedError`, or client disconnect.
A client disconnect during the final event should not be treated as a provider
failure. For `FISH_MEMORY_TTS_PROVIDER=dashscope`, verify the independent
`FISH_MEMORY_TTS_API_KEY` and WebSocket endpoint. For local TTS, install the `voice`
extra and check the model download log under `.cache/fish-memory/tts`.

## Voice interruption is inaccurate

Do not tune one threshold from a transcript alone. Enable local diagnostics,
reproduce once, and compare:

- raw microphone capture;
- TTS render reference;
- AEC-cleaned audio;
- committed question audio;
- VAD/ASR event times and playback offsets.

If speech is clear in capture but absent after AEC, inspect reference alignment
and double-talk behavior. If it is clear after AEC but missing from the question,
inspect VAD turn boundaries. If question audio is complete but text is wrong,
the issue is ASR/model selection rather than AEC.

Diagnostics contain user audio. Keep them local, use short retention, and disable
recording after diagnosis.


For protocol details, see [API](api.md).
