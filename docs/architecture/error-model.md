# Error model

All shardyfusion exceptions inherit from `ShardyfusionError` (`shardyfusion/errors.py:15`). Errors are organized by *responsibility*: who can act on them.

## Hierarchy

```
ShardyfusionError                       errors.py:15
├── ConfigValidationError               errors.py:26   # caller misconfigured something
├── ShardAssignmentError                errors.py:37   # row could not be routed to a shard
├── ShardCoverageError                  errors.py:47   # one or more shards had no winner
├── DbAdapterError                      errors.py:62   # backend driver failure
│   ├── SqliteAdapterError              sqlite_adapter.py:47        (not in public __all__)
│   ├── SqliteVecAdapterError           sqlite_vec_adapter.py:94    (not in public __all__)
│   └── CompositeAdapterError           composite_adapter.py:57     (not in public __all__)
├── ManifestBuildError                  errors.py:75   # writer failed to assemble a manifest
├── PublishManifestError                errors.py:84   # manifest object PUT failed
├── PublishCurrentError                 errors.py:94   # _CURRENT pointer PUT failed
├── ManifestParseError                  errors.py:109  # downloaded manifest is malformed
├── ReaderStateError                    errors.py:119  # reader API misuse
├── PoolExhaustedError                  errors.py:129  # connection pool exhausted
├── ManifestStoreError                  errors.py:139  # generic store failure on read
├── ShardWriteError                     errors.py:149  # shard write failed
├── S3TransientError                    errors.py:160  # retry-eligible S3 error
├── VectorIndexError                    errors.py:174  # vector index build failure  (not in public __all__)
└── VectorSearchError                   errors.py:184  # vector search failure       (not in public __all__)

# Deprecated alias (in public __all__):
SlateDbApiError = DbAdapterError                       errors.py:72
```

Errors marked "not in public `__all__`" must be imported from their owning module (e.g. `from shardyfusion.sqlite_adapter import SqliteAdapterError`). All other errors above are re-exported from `shardyfusion`.

## Responsibility model

| Error | Who acts? | What action? |
|---|---|---|
| `ConfigValidationError` | Caller | Fix the config. Build doesn't start. |
| `ShardAssignmentError` | Caller | Fix the data or the routing config (e.g. unknown CEL token). |
| `ShardCoverageError` | Operator | Investigate why some shards had zero successful attempts (e.g. all retries exhausted). |
| `DbAdapterError` | Operator | Backend-specific debugging; usually retried by the engine. |
| `ManifestBuildError` | Operator | Internal — file a bug. |
| `PublishManifestError` | Operator | Often transient; rerun the writer. |
| `PublishCurrentError` | Operator | Manifest object exists but invisible; rerun publishes a new pointer. |
| `ManifestParseError` | Operator | Manifest is corrupt or from incompatible version; check writer/reader version compatibility. |
| `ReaderStateError` | Caller | Fix application code (e.g. using a closed reader). |
| `PoolExhaustedError` | Caller / Operator | Increase pool size or reduce concurrency. |
| `ManifestStoreError` | Operator | Storage backend issue. |
| `ShardWriteError` | Operator | Inspect underlying cause; engine usually retries. |
| `S3TransientError` | (auto) | Retried by `_retry_s3_operation` (`storage.py:84`). |
| `VectorIndexError` | Operator | Vector backend (LanceDB / sqlite-vec) failed to build the index. |
| `VectorSearchError` | Caller | Bad query (e.g. unsupported metric, dimension mismatch). |

## Transient vs permanent

`_is_transient_s3_error` (`storage.py:44`) classifies S3 errors. Transient errors (5xx, throttling, network) are wrapped in `S3TransientError` and retried with exponential backoff inside `_retry_s3_operation`. Permanent errors (4xx other than throttling) propagate immediately.

## Errors at the public surface

The reader's `get(key)` method can raise:

- `ManifestStoreError` (cannot read `_CURRENT`).
- `ManifestParseError` (manifest is corrupt or unsupported version).
- `DbAdapterError` (shard backend failure).
- `ReaderStateError` (API misuse).
- `PoolExhaustedError` (under load).

The writer's `build()` method can raise any error in the hierarchy. Recovery is documented per use-case.

## Special: `UnknownRoutingTokenError`

`UnknownRoutingTokenError` (`shardyfusion/cel.py:173`) inherits from `ValueError`, **not** from `ShardyfusionError` — it is raised during CEL evaluation when a row's routing token isn't in the declared `routing_values` list. Callers typically wrap CEL evaluation in their own error handling.

## See also

- [`writer-core.md`](writer-core.md) — how writer errors arise.
- [`adapters.md`](adapters.md) — adapter-specific errors.
- [`retry-and-cleanup.md`](retry-and-cleanup.md) — what is retried automatically.
