# Versioning and releases

[README](../README.md) | [Changelog](../CHANGELOG.md)

tulpa follows [Semantic Versioning](https://semver.org/) using `MAJOR.MINOR.PATCH`.
Git release tags use the form `vX.Y.Z`. The desktop app version
(`desktop/pet/package.json`) and the local service version (`pyproject.toml`)
move together.

## Pre-1.0 policy

The project is currently `0.x`. Until 1.0:

- PATCH releases fix defects, security issues, and documentation without
  intentionally changing supported behavior;
- MINOR releases may change settings, storage, and local API contracts when the
  change is documented in the changelog;
- changes that touch persisted user data (database, settings, imported models)
  must migrate it automatically.

## Release checklist

1. Choose the version from user-visible impact.
2. Move `[Unreleased]` entries in `CHANGELOG.md` into a dated release section.
3. Update `version` in `pyproject.toml` and `desktop/pet/package.json`, then
   refresh `uv.lock` and `package-lock.json`.
4. Run the verification set:

   ```bash
   uv lock --check
   uv run ruff format --check src tests
   uv run ruff check .
   uv run pytest -q
   cd desktop/pet && npm run typecheck && npm test
   ```

5. Build the installers on each platform (`npm run dist:mac` on macOS,
   `npm run dist:win:local` on Windows) and smoke-test a clean install:
   onboarding, voice wake, chat, model import, and quitting without leftover
   processes.
6. Check `desktop/pet/THIRD_PARTY_NOTICES.md` against what the installers
   actually bundle.
7. Commit, create an annotated `vX.Y.Z` tag, push both, and attach the
   installers to a GitHub release with notes from the changelog.

## Database compatibility

Migrations are forward-only and run at startup. A released migration is
immutable. Schema rollback is not promised.
