# Computer History model output

The current desktop sends individual operation events directly into
[Session History](session-history.md) and does not invoke these periodic summary
prompts. This document describes the legacy endpoints and stored output format.

The segment, incremental checkpoint, checkpoint consolidation and final merge
prompts all include a complete JSON Schema generated from their Pydantic output
models, plus an example of the required nesting. A checkpoint has exactly two
top-level objects, `fullSummary` and `deltaCapsule`; their fields cannot be moved
to the checkpoint root.

The existing limits remain explicit: title 200 characters, description 500,
memory summary 4,000, and each context/detail item 1,500. Context arrays and
recording sections retain their item-count limits. Individual list items are
validated before text normalization, so an oversized item cannot pass by being
silently clipped. Model responses remain bounded to 32 KiB, independently of
the existing Markdown and Observation storage budgets.

Only the LLM response parser accepts these lossless container corrections:

- A JSON-encoded summary object is decoded once; plain narrative is not treated
  as a structured summary.
- A nonempty array containing only nonblank strings in `memorySummary` is joined
  in order and then validated against the ordinary string limit.
- A string in a context array or `details` becomes one array item. Empty strings,
  mixed types, unknown fields and oversized content still fail validation.

Persisted models and request schemas keep their strict types. These corrections
do not invent missing fields, discard unsupported fields or raise size limits.

If a bounded response still fails JSON/schema validation, the service allows
one additional `complete_json` call to repair that response's structure. This
call contains only the original model response and the required schema. It
does not resend the raw activity, semantic event timeline, prior context or
original request. The instructions prohibit adding, deleting or reinterpreting
facts, and require full-summary and delta scopes to remain separate. A long
description may be rewritten concisely only while retaining its facts in the
other permitted fields.

The repaired response passes through the same strict parser. A second failure
does not trigger another repair, advance a checkpoint or finalize an
Observation. The caller retains its frozen raw segment and existing retry
schedule. Initial provider failures and responses above the transport size
limit do not trigger structural repair. Cancellation propagates normally.

Diagnostics retain schema paths, error types, input types/lengths, response
byte counts and repair outcome types. They omit exception messages, raw input,
model text and credentials. Tests use synthetic output shapes; diagnosis does
not require another manual upload of historical activity.
