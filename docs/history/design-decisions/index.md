# Architecture Decision Records

Each ADR captures the context, decision, and consequences for a foundational design choice. ADRs are immutable once accepted — superseding decisions become new ADRs that reference the prior one.

## Accepted ADRs

- [ADR-001: Two-phase publish](adr-001-two-phase-publish.md) — manifest write then atomic `_CURRENT` swap
- [ADR-002: Categorical CEL routing](adr-002-categorical-cel-routing.md) — finite-domain expressions with explicit `routing_values`
- [ADR-003: Run registry & deferred cleanup](adr-003-run-registry-deferred-cleanup.md) — durable run records drive loser cleanup
- [ADR-004: Consistent writer retry](adr-004-consistent-writer-retry.md) — single `RetryConfig` shape across writer flavors
- [ADR-005: Sharded vector search](adr-005-sharded-vector-search.md) — per-shard vector indexes with K-amplification merge
- [ADR-006: LanceDB vector backend](adr-006-lancedb-vector-backend.md) — default high-quality vector backend
