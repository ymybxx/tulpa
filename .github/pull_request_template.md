## Problem

<!-- What user-visible or engineering problem does this solve? -->

## Changes

<!-- Keep the implementation summary concrete and scoped. -->

## Verification

<!-- List exact commands and results. -->

- [ ] Tests fail without the fix or otherwise prove the new contract.
- [ ] `uv run ruff format --check src tests`
- [ ] `uv run ruff check .`
- [ ] `uv run ty check src`
- [ ] Relevant `uv run pytest` tests pass.
- [ ] PostgreSQL integration tests ran when storage behavior changed.
- [ ] Desktop: `npm run typecheck` and relevant `npx vitest run` tests pass in `desktop/pet`.
- [ ] Desktop changes were tried in the running app on macOS and/or Windows.

## Compatibility and migrations

<!-- Note API, prompt, schema, configuration, or persisted-data impact. -->

- [ ] No compatibility impact, or the impact and migration are documented.
- [ ] Public docs are updated when contracts changed.
- [ ] `CHANGELOG.md` is updated for user-visible changes.

## Performance and accuracy

<!-- Required when latency, tokens, retrieval, memory quality, or voice changes. -->

- [ ] Before/after evidence is included, or this change has no measurable impact.

## Repository hygiene

- [ ] No secrets, `.env`, datasets, result files, databases, models, caches,
      diagnostics, or user audio are included.
- [ ] The change is not mixed with unrelated refactoring.
