# Writer Side

Primary entrypoint:

- `write_sharded_slatedb(df, config, spark_conf_overrides=None, cache_input=False, storage_level=None)`

Key guarantees:

- one-writer-per-db partitioning contract (`num_dbs` partitions)
- retry/speculation safety via attempt-isolated shard output URLs
- deterministic winner selection per `db_id`

Typical usage:

```python
from slatedb_spark_sharded import SlateDbConfig, ValueSpec, write_sharded_slatedb

config = SlateDbConfig(
    num_dbs=8,
    s3_prefix="s3://bucket/prefix",
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
)

result = write_sharded_slatedb(
    df,
    config,
    spark_conf_overrides={"spark.speculation": "false"},
)
```

Manifest + CURRENT behavior:

1. winner shard metadata is assembled into a manifest artifact
2. manifest is published first
3. `_CURRENT` JSON pointer is published second

See `Architecture` for end-to-end data flow details.
