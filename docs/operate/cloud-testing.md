# Cloud Testing Checklist

This document provides a structured checklist for validating shardyfusion against real cloud infrastructure. All tests below are manual — they are not part of the automated CI pipeline.

## AWS S3 + Spark (EMR)

### Prerequisites

- AWS account with EMR and S3 permissions
- S3 bucket provisioned (e.g. `s3://my-org-shardyfusion-test/`)
- EMR cluster with Spark 3.5 or 4.x, Python 3.11-3.13, Java 17

### Write Test

```bash
# On EMR master node
pip install shardyfusion[writer-spark-slatedb]
```

```python
from shardyfusion import WriteConfig, ValueSpec
from shardyfusion.writer.spark import write_sharded

config = WriteConfig(num_dbs=8, s3_prefix="s3://my-org-shardyfusion-test/spark-test")
result = write_sharded(df, config, key_col="id", value_spec=ValueSpec.binary_col("payload"))

# Verify
assert result.stats.rows_written > 0
assert len(result.winners) == 8
print(f"Manifest: {result.manifest_ref}")
```

- [ ] `num_dbs=8` completes successfully
- [ ] `num_dbs=64` completes successfully
- [ ] `num_dbs=256` completes successfully
- [ ] Manifest JSON is valid and readable

### Read Test

```python
from shardyfusion import ShardedReader

reader = ShardedReader(
    s3_prefix="s3://my-org-shardyfusion-test/spark-test",
    local_root="/tmp/reader-test",
)

value = reader.get(42)
batch = reader.multi_get([1, 2, 3, 42, 100])
info = reader.snapshot_info()
print(info)
reader.close()
```

- [ ] `get()` returns correct values
- [ ] `multi_get()` returns all expected keys
- [ ] `snapshot_info()` shows correct metadata
- [ ] Reader works from a separate machine (not the EMR cluster)

### Cross-Writer Test

- [ ] Write with Spark, read with Python reader on a different host
- [ ] Routing produces identical shard assignments

## AWS S3 + Dask

### Prerequisites

- Python 3.11-3.13 environment with `pip install "shardyfusion[writer-dask-slatedb]"`
- Use `pip install "shardyfusion[writer-dask-sqlite]"` to exercise the SQLite backend instead
- S3 credentials configured via env vars or `~/.aws/credentials`

### Test

```python
import dask.dataframe as dd
from shardyfusion import WriteConfig, ValueSpec
from shardyfusion.writer.dask import write_sharded

ddf = dd.from_pandas(pdf, npartitions=4)
config = WriteConfig(num_dbs=8, s3_prefix="s3://my-org-shardyfusion-test/dask-test")
result = write_sharded(ddf, config, key_col="id", value_spec=ValueSpec.binary_col("payload"))
```

- [ ] Hash sharding completes
- [ ] CEL sharding completes (with `ShardingSpec(strategy=ShardingStrategy.CEL, ...)`)
- [ ] Rate limiting verified (`max_writes_per_second=1000`)
- [ ] Read test with `ShardedReader` passes

## AWS S3 + Ray

### Prerequisites

- Python 3.11-3.13 environment with `pip install shardyfusion[writer-ray-slatedb]`
- Ray cluster or local mode

### Test

```python
import ray
from shardyfusion import WriteConfig, ValueSpec
from shardyfusion.writer.ray import write_sharded

ds = ray.data.from_items([{"id": i, "payload": b"data"} for i in range(10000)])
config = WriteConfig(num_dbs=8, s3_prefix="s3://my-org-shardyfusion-test/ray-test")
result = write_sharded(ds, config, key_col="id", value_spec=ValueSpec.binary_col("payload"))
```

- [ ] Hash sharding completes
- [ ] Range sharding completes
- [ ] `shuffle_strategy` restored after write (check `DataContext.shuffle_strategy`)
- [ ] Read test with `ShardedReader` passes

## AWS S3 + Python Writer

### Test

```python
from shardyfusion import WriteConfig
from shardyfusion.writer.python import write_sharded

records = [{"id": i, "payload": f"value-{i}".encode()} for i in range(10000)]
config = WriteConfig(num_dbs=4, s3_prefix="s3://my-org-shardyfusion-test/python-test")

# Single-process
result = write_sharded(records, config, key_fn=lambda r: r["id"], value_fn=lambda r: r["payload"])

# Multi-process
result = write_sharded(records, config, key_fn=lambda r: r["id"], value_fn=lambda r: r["payload"], parallel=True)
```

- [ ] Single-process mode completes
- [ ] Parallel mode completes
- [ ] Rate limiting verified
- [ ] Read test passes

## GCS with S3-Compatible API

### Prerequisites

- GCS bucket with interoperability access keys
- HMAC keys generated in GCS console

### Configuration

```python
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions

credential_provider = StaticCredentialProvider(
    access_key_id="GOOG...",
    secret_access_key="...",
)
connection_options: S3ConnectionOptions = {
    "endpoint_url": "https://storage.googleapis.com",
    "addressing_style": "path",
}

config = WriteConfig(
    num_dbs=4,
    s3_prefix="s3://my-gcs-bucket/shardyfusion-test",
    credential_provider=credential_provider,
    s3_connection_options=connection_options,
)
```

- [ ] Write completes via GCS interoperability layer
- [ ] Manifest readable from GCS
- [ ] Reader works with GCS S3-compatible endpoint

## Performance Benchmarks

### Write Throughput (rows/sec)

| Backend | num_dbs=8 | num_dbs=64 | num_dbs=256 |
|---|---|---|---|
| Spark (EMR m5.xlarge x4) | _____ | _____ | _____ |
| Dask (local, 4 workers) | _____ | _____ | _____ |
| Ray (local, 4 workers) | _____ | _____ | _____ |
| Python (single-process) | _____ | _____ | _____ |
| Python (parallel) | _____ | _____ | _____ |

### Read Latency (ms)

| Operation | p50 | p95 | p99 |
|---|---|---|---|
| `get()` (warm cache) | _____ | _____ | _____ |
| `get()` (cold) | _____ | _____ | _____ |
| `multi_get(100 keys)` | _____ | _____ | _____ |
| `refresh()` | _____ | _____ | _____ |

### Refresh Under Load

- [ ] 10 threads calling `get()` during `refresh()` — no errors
- [ ] Old readers closed only after all in-flight reads complete

## Cost Estimates

| Scenario | S3 PUT | S3 GET | Data Transfer | Approximate Cost |
|---|---|---|---|---|
| Write 1M rows / 8 shards | ~16 PUTs | — | ~100MB | < $0.01 |
| Write 10M rows / 64 shards | ~640 PUTs | — | ~1GB | < $0.10 |
| Read 10K gets | — | ~10K GETs | ~10MB | < $0.01 |
| Full test suite (all backends) | ~2K PUTs | ~50K GETs | ~5GB | < $1.00 |

## Cleanup

After testing, remove test data:

```bash
aws s3 rm s3://my-org-shardyfusion-test/ --recursive
```
