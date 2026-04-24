# ADR-002: Categorical CEL routing

**Status:** Accepted (2026-03-28)
**Source:** [`historical-notes/2026-03-28-categorical-cel-routing.md`](../historical-notes/2026-03-28-categorical-cel-routing.md)

## Context

Original sharding strategies (hash, range) are blind to record semantics. Some workloads need routing on a categorical field (e.g. `tenant_id`, `region`) so that:

- All records with the same value land on the same shard.
- The set of routing values is known at write time and stored in the manifest, so readers can route a query without scanning data.

Hash routing of a categorical field works but defeats locality and complicates targeted reads.

## Decision

Add a **categorical CEL routing strategy**:

- `ShardingSpec` carries `cel_expr` (CEL expression returning a routing token), `cel_columns` (input columns), and `routing_values` (the closed set of allowed tokens).
- CEL evaluation uses `cel-expr-python>=0.1` (NOT `celpy`).
- The token set is captured into the manifest (format version 3 required).
- Unknown tokens at read time raise `UnknownRoutingTokenError` (a `ValueError` subclass).
- `infer_routing_values_from_data=True` lets the writer derive the token set from the input.

## Consequences

- Routing is deterministic and locality-preserving for categorical fields.
- Reader can answer "which shard holds tenant X" without I/O.
- Manifest format bumped to v3; v1/v2 lack `routing_values`.
- New error type for unknown tokens (separate from generic routing errors).

## Related

- [`architecture/sharding.md`](../../architecture/sharding.md)
- [`architecture/routing.md`](../../architecture/routing.md)
