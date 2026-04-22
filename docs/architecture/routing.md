# Routing

Routing maps a key to a `db_id` deterministically. The same function is used by writers (to assign rows to shards) and by readers (to look up which shard holds a given key). If the two ever disagreed, lookups would silently miss data — so this is the single most important invariant in the project. It is enforced by `tests/unit/writer/core/test_routing_contract.py` (Hypothesis property tests).

## The HASH formula

```python
db_id = xxh3_64(canonical_bytes(key), seed=0) % num_dbs
```

Implemented as `xxh3_db_id` (`shardyfusion/routing.py:71`).

- `xxh3_64` is the 64-bit variant of xxHash3 with `seed=0`.
- `canonical_bytes` (`routing.py:42`) encodes the key based on `WriteConfig.key_encoding`. `U64BE`/`U32BE` produce big-endian unsigned ints; `UTF8` produces UTF-8 bytes; `RAW` passes bytes through.
- The modulus is taken in unsigned 64-bit space.

`xxh3_digest` (`routing.py:62`) returns the raw 64-bit digest if you need it (e.g. for diagnostics or test fixtures).

## The CEL path

For `ShardingStrategy.CEL`, the `db_id` is the result of evaluating a compiled CEL expression against the row, then either:

- **Direct mode**: `result % num_dbs` (result must be `int`).
- **Categorical mode**: the result is a string token; the token is looked up in a precomputed `routing_values → db_id` map built by `build_categorical_routing_lookup` (`cel.py:303`). Unknown tokens raise `UnknownRoutingTokenError` (`cel.py:173`).

The categorical map is materialized into the manifest at build time (see [`manifest-and-current.md`](manifest-and-current.md), format v3) so that readers can look up "which shard holds tenant `eu-west`" by name without compiling CEL themselves.

## SnapshotRouter

`SnapshotRouter` (`routing.py:84`) is the reader-side router. It is constructed via `SnapshotRouter.from_build_meta(...)` (`routing.py:118`) from a parsed manifest's `RequiredBuildMeta`. Public surface:

- `get_shard(key)` (`routing.py:167`) — returns the `db_id` for a single key.
- `route(keys)` (`routing.py:182`) — vectorized routing.
- `group_keys(keys)` (`routing.py:190`) — groups keys by `db_id` for fan-out.
- `group_keys_allow_missing(keys)` (`routing.py:208`) — same, but tolerates keys whose categorical routing token isn't in the manifest.
- `route_with_context(rows)` (`routing.py:313`) — CEL routing requiring per-row context.
- `is_lazy` (property, `routing.py:177`) — whether routing requires CEL evaluation.

`ShardLookup` (`routing.py:21`) is the protocol that mounted shard adapters implement; the router combines it with the routing function to get end-to-end `key → adapter → bytes`.

## Why xxHash3, why seed=0

xxHash3 is fast (gigabytes per second per core), portable across platforms, and has good avalanche properties — which matters because the modulus operation amplifies bias. `seed=0` is a deliberate non-choice: there is no benefit from a non-zero seed unless you need adversarial-input resistance, and changing seeds would silently re-shard every existing snapshot.

## What is *not* hashed

- Row values other than the key column.
- Manifest metadata.
- Run IDs.

The hash is purely a function of the key bytes. This means the same logical key always lands on the same shard *for a given `num_dbs`* — but changing `num_dbs` between runs reshuffles everything. See [`sharding.md`](sharding.md).

## See also

- [`sharding.md`](sharding.md) — `ShardingSpec`, `KeyEncoding`.
- [`writer-core.md`](writer-core.md) — `route_key`, `discover_cel_num_dbs`.
- [`adapters.md`](adapters.md) — `ShardLookup` implementations per backend.
