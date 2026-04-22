# Sharding

A **shard** is one physical database (`db_id` ∈ `[0, num_dbs)`). Sharding decisions are described by `ShardingSpec` (`shardyfusion/sharding_types.py:111`).

## Strategies

`ShardingStrategy` (`sharding_types.py:89`) has two values:

| Strategy | Routing function | When to use |
|---|---|---|
| `HASH` | `xxh3_db_id(canonical_bytes(key), num_dbs)` (`routing.py:71`) | Uniform distribution over int, string, or bytes keys. |
| `CEL` | Compiled CEL expression — direct (int return) or categorical (string→shard map) | Custom routing: tenant pinning, geo affinity, hot-key isolation. |

## Key encoding

`KeyEncoding` (`sharding_types.py:66`) determines how the key bytes are formed before hashing. It lives on `WriteConfig.key_encoding`, **not** on `ShardingSpec`:

- `U64BE` — 8-byte big-endian unsigned int (default).
- `U32BE` — 4-byte big-endian unsigned int.
- `UTF8` — UTF-8 string.
- `RAW` — bytes passed through.

`canonical_bytes` (`routing.py:42`) produces them deterministically; the same encoding is used by readers when looking up keys, so writer and reader agree by construction. See [`routing.md`](routing.md).

## ShardingSpec fields

```
ShardingSpec(
    strategy: ShardingStrategy = HASH,
    routing_values: list[int|str|bytes] | None = None,   # CEL only
    cel_expr: str | None = None,                          # CEL only
    cel_columns: dict[str, str] | None = None,            # CEL only — column → CelType
    max_keys_per_shard: int | None = None,                # HASH only — discover num_dbs from data
    infer_routing_values_from_data: bool = False,         # CEL categorical only
)
```

`num_dbs` is **not** on `ShardingSpec` — it lives on `WriteConfig.num_dbs`. It must be set explicitly for HASH (or computed from `max_keys_per_shard`). For CEL it is **never set explicitly**: it is derived from `routing_values` (categorical) or discovered from data (direct mode via `discover_cel_num_dbs`). See [`history/design-decisions/adr-002-categorical-cel-routing.md`](../history/design-decisions/adr-002-categorical-cel-routing.md).

## CEL: direct, categorical, and inferred

`shardyfusion/cel.py` exposes two builders for `ShardingSpec`:

| Mode | Builder | Routing rule |
|---|---|---|
| Direct | `cel_sharding(expr, columns, *, routing_values=None)` (`cel.py:364`) | Expression returns an int; result `mod num_dbs` is the `db_id`. `num_dbs` is set on `WriteConfig` or discovered from data via `discover_cel_num_dbs` (`_writer_core.py:243`). |
| Categorical | `cel_sharding_by_columns(*columns, num_shards=None, separator=":")` (`cel.py:415`) | Builds a CEL expression that concatenates the named columns. Token must be in `routing_values`. Unknown tokens raise `UnknownRoutingTokenError` (`cel.py:173`). |
| Inferred categorical | Set `infer_routing_values_from_data=True` on `ShardingSpec` | The writer scans the input once to build `routing_values`; single-process Python writer only. |

Categorical mode is preferred when:

- Routing keys are a small known set (tenant codes, regions).
- You need readers to look up "which shard does tenant `eu-west` live in" without re-evaluating CEL.

Direct mode is preferred when:

- Routing is a hash-of-something computed at write time.
- You don't need symbolic per-shard names.

CEL routing requires manifest format **v3** (see [`manifest-and-current.md`](manifest-and-current.md)).

## Validation

`validate_routing_values` (`sharding_types.py:43`) rejects empty lists, duplicates, and non-string values at config-build time. `validate_cel_columns` (`cel.py:562`) checks that declared CEL column types are compatible with the Arrow schema observed at write time.

## What "num_dbs" means

`num_dbs` is **immutable for the lifetime of a snapshot's shard layout**. Changing it requires writing a new run with the new shard count; readers cannot re-shard data on the fly. The reader fetches `num_dbs` from the manifest, so a new build with a different `num_dbs` is observed atomically via `_CURRENT`.

## See also

- [`routing.md`](routing.md) — hash function and `SnapshotRouter`.
- [`writer-core.md`](writer-core.md) — how `num_dbs` is resolved (`resolve_num_dbs`).
- [`history/design-decisions/adr-002-categorical-cel-routing.md`](../history/design-decisions/adr-002-categorical-cel-routing.md).
