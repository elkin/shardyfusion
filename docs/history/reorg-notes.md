# Documentation reorganization (2026-04-21)

The docs were restructured from a concept-first layout (one page per major concept: writer, reader, vector-search, ...) into a **use-case-driven layout** (one page per task the user wants to accomplish, plus thin architecture/reference/operations sections).

## What moved where

### Removed top-level pages

The following pages were superseded and removed. Their content was distributed across use-cases, architecture, and reference:

| Old | New home |
|---|---|
| `index.md` | rewritten as a clickable use-case map |
| `getting-started.md` | merged into per-use-case "Install" + "Minimal example" sections |
| `how-it-works.md` | split across `architecture/*` |
| `writer.md` | `architecture/writer-core.md` + per-framework use-cases |
| `writers/spark.md` | `use-cases/build-spark-{slatedb,sqlite}.md` |
| `writers/dask.md` | `use-cases/build-dask-{slatedb,sqlite}.md` |
| `writers/ray.md` | `use-cases/build-ray-{slatedb,sqlite}.md` |
| `writers/python.md` | `use-cases/build-python-{slatedb,sqlite}.md` |
| `reader.md` | `use-cases/read-{sync,async}-{slatedb,sqlite}.md` + `architecture/manifest-and-current.md` |
| `vector-search.md` | `use-cases/build-vector-*.md` + `use-cases/build-python-{slatedb-lancedb,sqlite-vec}.md` |
| `manifest-stores.md` | `architecture/manifest-stores.md` |
| `cli.md` | `use-cases/operate-cli.md` + `reference/cli.md` |
| `observability.md` | `architecture/observability.md` + `use-cases/operate-{prometheus,otel}-metrics.md` |
| `error-handling.md` | `architecture/error-model.md` |
| `glossary.md` | `reference/glossary.md` |
| `api.md` | `reference/api.md` |
| `release.md` | merged into `contributing/` |

### Operations

| Old | New |
|---|---|
| `operations.md` | `operations/index.md` (verbatim) |
| `cloud-testing.md` | `operations/cloud-testing.md` (verbatim) |
| `tox-and-dependency-matrix.md` | `operations/tox-matrix.md` (verbatim) |

### Engineering notes

The 9 dated notes were copied verbatim to `history/historical-notes/`. Six ADRs were extracted to `history/design-decisions/`. Two still-open plans were also copied to `history/implementation-plans/`.

| Source note | ADR(s) | Plans |
|---|---|---|
| `2026-03-14-manifest-history-and-rollback.md` | ADR-001 | — |
| `2026-03-15-reliable-loser-cleanup-plan.md` | — (informs ADR-003) | open |
| `2026-03-24-test-coverage-gap-closure.md` | — | open |
| `2026-03-25-consistent-writer-retry.md` | ADR-004 | — |
| `2026-03-25-run-registry-for-deferred-cleanup.md` | ADR-003 | — |
| `2026-03-28-categorical-cel-routing.md` | ADR-002 | — |
| `2026-04-04-sharded-vector-search.md` | ADR-005 | — |
| `2026-04-06-vector-search-review.md` | informs ADR-006 | — |
| `2026-04-19-lancedb-vector-migration.md` | ADR-006 | — |

### New sections

- `use-cases/` — 20 task-oriented pages, each following an 11-section template: When to use / When NOT to use / Install / Minimal example / Configuration / Functional properties / Non-functional properties / Guarantees / Weaknesses / Failure modes & recovery / See also.
- `architecture/` — 11 internals pages.
- `contributing/` — 8 pages on how to extend / contribute.
- `reference/` — `api.md`, `config.md`, `cli.md`, `glossary.md`.
- `history/` — ADRs + open plans + verbatim notes (this section).

## No redirects

The project has no users yet. Old paths are deleted, not redirected.

## Validation

A repo-local `validate-docs` skill at `.claude/skills/validate-docs/` cross-checks documented facts against source code.
