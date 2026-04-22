# History

This section preserves the project's design and engineering history.

## Sub-sections

- **[Design decisions](design-decisions/index.md)** — short ADRs (Architecture Decision Records) capturing the *why* of major architectural choices. Each ADR links back to its source engineering note.
- **[Implementation plans](implementation-plans/index.md)** — still-open work tracked from engineering notes. Move to `historical-notes/` when complete.
- **[Historical notes](historical-notes/index.md)** — original dated engineering notes, preserved verbatim.

## Index of ADRs

- [ADR-001 — Two-phase manifest publish](design-decisions/adr-001-two-phase-publish.md)
- [ADR-002 — Categorical CEL routing](design-decisions/adr-002-categorical-cel-routing.md)
- [ADR-003 — Run registry for deferred cleanup](design-decisions/adr-003-run-registry-deferred-cleanup.md)
- [ADR-004 — Consistent writer retry](design-decisions/adr-004-consistent-writer-retry.md)
- [ADR-005 — Sharded vector search](design-decisions/adr-005-sharded-vector-search.md)
- [ADR-006 — LanceDB as vector backend](design-decisions/adr-006-lancedb-vector-backend.md)

## Open implementation plans

- [Reliable loser cleanup](implementation-plans/2026-03-15-reliable-loser-cleanup-plan.md)
- [Test coverage gap closure](implementation-plans/2026-03-24-test-coverage-gap-closure.md)

## Reorganization

The docs were reorganized in 2026-04-21 from a concept-first layout into a use-case-driven layout. See [`reorg-notes.md`](reorg-notes.md) for what moved where.
