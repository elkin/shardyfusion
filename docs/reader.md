# Reader Side

Primary API:

- `SlateShardedReader`

The reader loads:

1. `_CURRENT` pointer
2. manifest referenced by `_CURRENT`
3. one shard reader per `db_id`
4. a router that maps keys to shard IDs

Example:

```python
from shardyfusion import SlateShardedReader

reader = SlateShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/shardyfusion-reader",
)

value = reader.get(123)
values = reader.multi_get([1, 2, 3])

changed = reader.refresh()
reader.close()
```

Custom manifest reader mode:

- provide `manifest_reader` when manifests are not in default S3 JSON format
- use `FunctionManifestReader` for simple adapter-style integrations
