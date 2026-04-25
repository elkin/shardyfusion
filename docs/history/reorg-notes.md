# Documentation reorganization

## 2026-04-25 — Top tabs and unified operations section

The MkDocs Material layout now uses top-level tabs for the main documentation sections. The former **Operate** and **Operations** top-level nav groups were merged into one **Operations** tab.

Path changes:

- `operations/index.md` → `operate/production.md`
- `operations/cloud-testing.md` → `operate/cloud-testing.md`
- `operations/tox-matrix.md` → `operate/tox-matrix.md`

## 2026-04-22 — Tree-structured use cases

The use-case docs were reorganized from a flat list (18 pages at the same level) into a **decision tree**:

- Three top-level use-case families: **Sharded KV storage**, **Sharded KV + Vector**, **Sharded Vector search**.
- Each family has a conceptual **overview** page (shared concepts: sharding, manifests, two-phase publish, safety).
- Overview pages branch into **build** and **read** sections.
- Writer build pages are **merged by writer flavor** (one page covers both SlateDB and SQLite backends).
- KV+Vector build pages use **content tabs** for Python/Spark/Dask/Ray examples.
- Reader pages are **split by backend** (SlateDB vs SQLite) under sync/async.
- **Operate** docs were promoted to a top-level nav section.

### Directory changes

| Old flat path | New tree path |
|---|---|
| `use-cases/build-python-slatedb.md` | `use-cases/kv-storage/build/python.md` |
| `use-cases/build-python-sqlite.md` | merged into `use-cases/kv-storage/build/python.md` |
| `use-cases/build-spark-slatedb.md` | `use-cases/kv-storage/build/spark.md` |
| `use-cases/build-spark-sqlite.md` | merged into `use-cases/kv-storage/build/spark.md` |
| `use-cases/build-dask-slatedb.md` | `use-cases/kv-storage/build/dask.md` |
| `use-cases/build-dask-sqlite.md` | merged into `use-cases/kv-storage/build/dask.md` |
| `use-cases/build-ray-slatedb.md` | `use-cases/kv-storage/build/ray.md` |
| `use-cases/build-ray-sqlite.md` | merged into `use-cases/kv-storage/build/ray.md` |
| `use-cases/build-python-slatedb-lancedb.md` | `use-cases/kv-vector/build/composite.md` |
| `use-cases/build-python-sqlite-vec.md` | `use-cases/kv-vector/build/unified.md` |
| `use-cases/build-vector-lancedb-standalone.md` | `use-cases/vector/build/lancedb.md` |
| `use-cases/build-vector-sqlite-vec-standalone.md` | `use-cases/vector/build/sqlite-vec.md` |
| `use-cases/read-sync-slatedb.md` | `use-cases/kv-storage/read/sync/slatedb.md` |
| `use-cases/read-sync-sqlite.md` | `use-cases/kv-storage/read/sync/sqlite.md` |
| `use-cases/read-async-slatedb.md` | `use-cases/kv-storage/read/async/slatedb.md` |
| `use-cases/read-async-sqlite.md` | `use-cases/kv-storage/read/async/sqlite.md` |
| `use-cases/operate-cli.md` | `operate/cli.md` |
| `use-cases/operate-manifest-history-and-rollback.md` | `operate/history-rollback.md` |
| `use-cases/operate-prometheus-metrics.md` | `operate/prometheus-metrics.md` |
| `use-cases/operate-otel-metrics.md` | `operate/otel-metrics.md` |

New pages added:

- `use-cases/kv-storage/overview.md`
- `use-cases/kv-vector/overview.md`
- `use-cases/vector/overview.md`
- `use-cases/kv-storage/build/index.md`
- `use-cases/kv-storage/read/index.md`
- `use-cases/vector/read/sync.md`
- `use-cases/vector/read/async.md`
- `use-cases/kv-vector/read/sync.md`
- `use-cases/kv-vector/read/async.md`
- `operate/index.md`

### Diagrams restored

Mermaid diagrams from the pre-2026-04-21 deep-dive docs (`writers/*.md`, `reader.md`, `how-it-works.md`) were restored and updated to match current source code behavior. These include data-flow flowcharts for all four writers, manifest lifecycle sequence diagrams, reader refresh safety diagrams, and cold-start fallback flowcharts.

---

## 2026-04-21 — Use-case-driven layout

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
