# Sharding

A **shard** is one physical database (`db_id` ∈ `[0, num_dbs)`). Sharding decisions are described by `ShardingSpec` (`shardyfusion/sharding_types.py:111`).

## Strategies

`ShardingStrategy` (`sharding_types.py:89`) has two values:

| Strategy | Routing function | When to use |
|---|---|---|
| `HASH` | `hash_db_id(key, num_dbs, hash_algorithm)` (`routing.py`) | Uniform distribution over int, string, or bytes keys. |
| `CEL` | Compiled CEL expression — direct (int return) or categorical (string→shard map) | Custom routing: tenant pinning, geo affinity, hot-key isolation. |

## Key encoding

`KeyEncoding` determines how keys are serialized for storage and lookup. It lives on `KeyValueWriteConfig.key_encoding` (`config.kv.key_encoding`), **not** on `ShardingSpec`:

- `U64BE` — 8-byte big-endian unsigned int (default).
- `U32BE` — 4-byte big-endian unsigned int.
- `UTF8` — UTF-8 string.
- `RAW` — bytes passed through.

Routing uses `canonical_bytes` (`routing.py:42`) independently from `KeyEncoding`: `int` keys hash as signed little-endian int64, `str` keys hash as UTF-8, and `bytes` pass through. Readers use `KeyEncoding` only after routing, when encoding the lookup key for the shard backend. See [`routing.md`](routing.md).

## KV public sharding configs

User-facing KV writer config is split by strategy:

- `HashShardedWriteConfig(sharding=HashShardingConfig(...))` carries `num_dbs` or `max_keys_per_shard`.
- `CelShardedWriteConfig(sharding=CelShardingConfig(...))` carries `cel_expr`, `cel_columns`, optional `routing_values`, and `infer_routing_values_from_data`.

`HashShardingSpec` and `CelShardingSpec` in `sharding_types.py` are manifest/internal strategy records. `HashShardingConfig` and `CelShardingConfig` are the public writer config groups that validate user input early.

For HASH, `num_dbs` is required unless computed from `max_keys_per_shard`. For CEL, `num_dbs` is never provided explicitly: it is derived from `routing_values` (categorical) or discovered from data (direct mode via `discover_cel_num_dbs`). See [`history/design-decisions/adr-002-categorical-cel-routing.md`](../history/design-decisions/adr-002-categorical-cel-routing.md).

## CEL: direct, categorical, and inferred

`shardyfusion/cel.py` exposes two builders for `CelShardingSpec`; writers accept public `CelShardingConfig` through `CelShardedWriteConfig`:

| Mode | Builder | Routing rule |
|---|---|---|
| Direct | `cel_sharding(expr, columns, *, routing_values=None)` (`cel.py:364`) | Expression returns an int; result `mod num_dbs` is the `db_id`. `num_dbs` is discovered from data via `discover_cel_num_dbs`. |
| Categorical | `cel_sharding_by_columns(*columns, num_shards=None, separator=":")` (`cel.py:415`) | Builds a CEL expression that concatenates the named columns. Token must be in `routing_values`. Unknown tokens raise `UnknownRoutingTokenError` (`cel.py:173`). |
| Inferred categorical | Set `infer_routing_values_from_data=True` on `CelShardingConfig` | The writer scans the input once to build `routing_values`; single-process Python writer only. |

Categorical mode is preferred when:

- Routing keys are a small known set (tenant codes, regions).
- You need readers to look up "which shard does tenant `eu-west` live in" without re-evaluating CEL.

Direct mode is preferred when:

- Routing is a hash-of-something computed at write time.
- You don't need symbolic per-shard names.

CEL routing requires the current manifest format **v4** (see [`manifest-and-current.md`](manifest-and-current.md)).

`hash_algorithm` is still serialized for CEL manifests. The CEL `shard_hash()` function remains fixed to `xxh3_64`; `hash_algorithm` does not change CEL expression semantics.

## Validation

`validate_routing_values` (`sharding_types.py:43`) rejects empty lists, duplicates, and non-string values at config-build time. `validate_cel_columns` (`cel.py:562`) checks that declared CEL column types are compatible with the Arrow schema observed at write time.

## What "num_dbs" means

`num_dbs` is **immutable for the lifetime of a snapshot's shard layout**. Changing it requires writing a new run with the new shard count; readers cannot re-shard data on the fly. The reader fetches `num_dbs` from the manifest, so a new build with a different `num_dbs` is observed atomically via `_CURRENT`.

## See also

- [`routing.md`](routing.md) — hash function and `SnapshotRouter`.
- [`writer-core.md`](writer-core.md) — how `num_dbs` is resolved (`resolve_num_dbs`).
- [`history/design-decisions/adr-002-categorical-cel-routing.md`](../history/design-decisions/adr-002-categorical-cel-routing.md).
