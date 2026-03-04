# Reader Side

Primary APIs:

- `ShardedReader` — non-thread-safe, for single-threaded services
- `ConcurrentShardedReader` — thread-safe with lock/pool modes

Both readers load:

1. `_CURRENT` pointer
2. manifest referenced by `_CURRENT`
3. one shard reader per `db_id`
4. a router that maps keys to shard IDs

Example:

```python
from shardyfusion import ShardedReader

reader = ShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/shardyfusion-reader",
)

value = reader.get(123)
values = reader.multi_get([1, 2, 3])

changed = reader.refresh()
reader.close()
```

For thread-safe usage:

```python
from shardyfusion import ConcurrentShardedReader

reader = ConcurrentShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/shardyfusion-reader",
    thread_safety="lock",
    max_workers=4,
)
```

Custom manifest reader mode:

- provide `manifest_reader` when manifests are not in default S3 JSON format
- use `FunctionManifestReader` for simple adapter-style integrations
