---
name: validate
description: Run all validation checks (docs sync, CI/E2E tests, tox sync, infra sync, CI improvements) against recent changes before committing.
compatibility: opencode
metadata:
  audience: contributors
  workflow: pre-commit
---

# Post-Change Validation

Run all validation checks against recent changes. This mirrors the pre-commit hookify rules but can be invoked at any time.

The implementation is shared with the Claude-compatible skill at `.claude/skills/validate/`.

## Step 0 — Determine scope

Run `git diff --name-only HEAD` and `git status --porcelain` to identify modified files. Use this to decide which checks are relevant:

| Changed files match | Checks to run |
|---|---|
| `README.md`, `CLAUDE.md`, `AGENTS.md`, `docs/`, or any source in `shardyfusion/` | 1. Documentation sync |
| Any source or test file | 2. CI & E2E tests |
| `tests/`, `tox.ini` | 3. Tox sync |
| `docker/`, `.github/`, `tox.ini`, `pyproject.toml` | 4. Infrastructure sync |
| Any file | 5. CI improvements (advisory) |

If no files have changed, report "Nothing to validate" and stop.

## Check 1 — Documentation sync

Verify that documentation is consistent with the code changes:

1. **README.md** — Reflects the current public API, install instructions, usage examples, and development workflow. Any new features, changed commands, or modified APIs are documented.
2. **CLAUDE.md** — Reflects the current module structure, commands, architecture, testing notes, and error hierarchy. Specifically:
   - The "Architecture" section matches actual module dependencies
   - The "Commands" section lists current build/test/lint commands
   - The "Error Hierarchy" table includes all error classes
   - The "Public API Summary" lists all exports from `__init__.py`
   - The "Testing Notes" section covers any new test directories or markers
3. **AGENTS.md** — Reflects the current project structure, build commands, test commands, and coding style.
4. **docs/ directory** — MkDocs source files are consistent with code changes. Run the automated docs validation script:

   ```bash
   uv run python scripts/check_docs.py
   ```

   This script verifies:
   - **Public-API references** — every `::: shardyfusion.<symbol>` mkdocstrings reference and every qualified `shardyfusion.X.Y` mention in inline code or fenced Python blocks resolves to a real attribute.
   - **Config dataclass fields** — every field name listed in a docs configuration table for tracked dataclasses (`WriteConfig`, `ShardingSpec`, `OutputOptions`, `ManifestOptions`, `VectorSpec`, `RetryConfig`, `S3ConnectionOptions`) exists on that class.
   - **Pyproject extras** — every `--extra <name>` and `shardyfusion[<name>]` line in `docs/**/*.md` references an extra defined in `[project.optional-dependencies]`.
   - **Extras matrix sync** — every runtime extra is documented in `docs/use-cases/extras-matrix.md`.
   - **CLI surface** — every `shardy <subcommand>` invocation and every `--flag` referenced in CLI pages exists in `shardyfusion/cli/app.py`.

   Exit code is `0` when everything passes, `1` when any check fails. The output is a structured Markdown report grouped by check and file. Fix any `NOT FOUND`, `NO SUCH FIELD`, or `not found in` failures before continuing.

If any documentation is stale or missing coverage, **fix it** before continuing.

## Check 2 — CI & E2E tests

Run the full test suite and fix any failures:

1. **Run `just ci`** — quality checks (lint, format, type-check), unit tests, and integration tests. All must pass.
2. **Run `just d-e2e`** — end-to-end tests against Garage S3 via compose. All must pass.

If any tests fail, diagnose and fix the failures before continuing.

## Check 3 — Tox sync

Verify that `tox.ini` is consistent with the test suite:

1. **Test directory coverage** — All `tests/` subdirectories have corresponding tox environments.
2. **Stale environments** — No tox environments reference removed or renamed test paths.
3. **Label completeness** — Tox labels (`quality`, `unit`, `integration`, `e2e`) include all relevant environments.
4. **New environments** — If new test directories or writer paths were added, corresponding tox environments exist and are included in appropriate labels.

If any inconsistencies are found, **fix `tox.ini`** before continuing.

## Check 4 — Infrastructure sync

Verify that Docker, GitHub Actions, and tox all use consistent Python and Java versions:

1. **Python version consistency** across `docker/ci.Dockerfile`, `tox.ini`, and `.github/workflows/ci.yml` (currently Python 3.11, 3.12, 3.13, 3.14).
2. **GitHub Actions workflow consistency** — `ci.yml`, `release.yml`, `docs.yml` use consistent Python versions.
3. **Java version consistency** — Java 17 (temurin) used consistently across all Spark-requiring configurations.
4. **Component Python version constraints**:
   - **Dask writer** (`daskwriter`): Python 3.11–3.13 only (no 3.14)
   - **Ray writer** (`raywriter`): Python 3.11–3.13 only (no 3.14)
   - **`all` composite env**: must not include py314 if it installs Dask or Ray extras
5. **Version drift** — Flag and fix any interpreter or runtime version drift between these files.

If any inconsistencies are found, **fix them** before continuing.

## Check 5 — CI improvements (advisory)

Review the changes and suggest improvements only with concrete justification:

- **New dependency category** — suggest adding an import smoke-test in CI.
- **New file type** — suggest adding an appropriate linter or validator.
- **New writer path** — suggest corresponding tox environments, routing contract tests, and integration test scenarios.
- **New test markers** — suggest adding them to CI matrix configurations.
- **Type-checking gaps** — suggest adding pyright project configs if not covered.

This check is **advisory only** — suggestions do not block validation.

## Summary

After running all applicable checks, present a summary table:

```
| Check                  | Status |
|------------------------|--------|
| 1. Documentation sync  | ...    |
| 2. CI & E2E tests      | ...    |
| 3. Tox sync            | ...    |
| 4. Infrastructure sync | ...    |
| 5. CI improvements     | ...    |
```

Status values: PASS, FAIL (with details), SKIP (not relevant to changes), ADVISORY (suggestions only).

If any check has status FAIL, fix the issues and re-run the failed checks until all pass. Only then declare validation complete.
