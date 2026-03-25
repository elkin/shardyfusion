# Operations

## CI Workflows

- **`CI`** workflow:
  - quality gates (`ruff`, `pyright`)
  - package build smoke checks
  - unit and integration test jobs (parallel within each stage)
  - Java 17 (temurin) required for Spark jobs
  - Python support is currently capped at 3.11-3.13; py3.14 is excluded from tox and CI until all readers, writers, and backends support it
  - Weekly scheduled build on Mondays at 06:00 UTC
- **`Docs`** workflow:
  - docs build on pull requests
  - docs publish to GitHub Pages on `main`
- **`Release`** workflow:
  - tag-triggered release validation + PyPI publish

## Running Checks Locally

```bash
# Quality
tox -m quality

# Package check
tox -e package

# Unit matrix (parallel tox envs)
tox -m unit
tox p -m unit -p 2

# Integration subsets
tox -e py311-read-slatedb-integration,py311-sparkwriter-spark4-slatedb-integration

# Full CI equivalent
just ci

# E2E (requires container runtime)
just d-e2e
```

### Memory Tuning for Tests

To avoid OOM when parallelizing:

- per-unit-env pytest is capped to `-n 2`
- tox environment-level parallelism should be capped (`tox p -p 2`)

## Manifest History and Rollback

### Viewing History

List recent manifests to see the snapshot build timeline:

```bash
# CLI
uv run shardy history --limit 10

# Programmatic
from shardyfusion import S3ManifestStore

store = S3ManifestStore(s3_prefix="s3://bucket/prefix")
for ref in store.list_manifests(limit=10):
    print(f"{ref.published_at}  {ref.run_id}  {ref.ref}")
```

### Rolling Back

When a bad snapshot is deployed (data corruption, wrong source data, config error), roll back to a known-good manifest:

**Via CLI:**

```bash
# Roll back to the previous manifest
uv run shardy rollback --offset 1

# Roll back to a specific run
uv run shardy rollback --run-id abc123

# Roll back to a specific manifest ref
uv run shardy rollback --ref s3://bucket/prefix/manifests/.../manifest
```

**Programmatic:**

```python
from shardyfusion import S3ManifestStore

store = S3ManifestStore(s3_prefix="s3://bucket/prefix")

# Find the manifest to roll back to
refs = store.list_manifests(limit=5)
target = refs[1]  # previous manifest

# Update the current pointer
store.set_current(target.ref)
```

Rollback updates the `_CURRENT` pointer only — shard data and old manifests remain untouched. Readers pick up the change on their next `refresh()`.

!!! warning
    Rollback affects all readers pointing at this `_CURRENT`. Coordinate with your team before rolling back in production.

## Run Registry

Each writer invocation also publishes one run record under
`output.run_registry_prefix` (default `runs`) as:

`s3://bucket/prefix/runs/<timestamp>_run_id=<run_id>_<uuid>/run.yaml`

Operational notes:

- the record is writer-owned metadata, not part of the reader contract
- status transitions are `running` -> `succeeded` or `failed`
- successful runs store `manifest_ref`; failed runs may leave it `null`
- `updated_at` and `lease_expires_at` are refreshed while the writer is alive

This record is intended for deferred cleanup and operational inspection. The
current cleanup CLI does not consume the run registry yet.

## Health Monitoring

### Reader Health Checks

Use `reader.health()` in health check endpoints to expose reader status:

```python
from datetime import timedelta
health = reader.health(staleness_threshold=timedelta(minutes=10))
```

`ReaderHealth` status values:

| Status | Meaning | Action |
|---|---|---|
| `healthy` | Manifest loaded and within staleness threshold | None |
| `degraded` | Manifest loaded but older than staleness threshold | Consider triggering refresh or investigating pipeline |
| `unhealthy` | Reader is closed or no manifest loaded | Restart reader or investigate |

### Alert Thresholds

Suggested alert thresholds based on your write pipeline cadence:

| Pipeline Cadence | `staleness_threshold` | Alert Level |
|---|---|---|
| Hourly | `timedelta(hours=2)` | Warning if no refresh in 2× cadence |
| Daily | `timedelta(days=2)` | Warning if no refresh in 2× cadence |
| On-demand | Varies | Alert on `unhealthy` status only |

### Prometheus Integration

```python
from shardyfusion.metrics.prometheus import PrometheusCollector
from prometheus_client import start_http_server

collector = PrometheusCollector()
reader = ConcurrentShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/reader",
    metrics_collector=collector,
)

# Expose Prometheus metrics on port 8000
start_http_server(8000)
```

Key metrics to monitor:

- `shardyfusion_reader_get_duration_seconds` — read latency
- `shardyfusion_s3_retries_total` — S3 instability
- `shardyfusion_rate_limiter_throttled_total` — rate limiter activations

## DB Manifest Store Operations

### Table Schema

Database manifest stores use two tables (auto-created when `ensure_table=True`):

- **`shardyfusion_manifests`** — one row per write (run_id PK, payload JSONB, created_at)
- **`shardyfusion_pointer`** — append-only current tracking (updated_at, manifest_ref)

### Pointer Table Growth

The pointer table is append-only (one INSERT per `publish()` or `set_current()` call). For high-frequency pipelines, periodically prune old rows:

```sql
-- Keep only the last 100 pointer entries
DELETE FROM shardyfusion_pointer
WHERE ctid NOT IN (
    SELECT ctid FROM shardyfusion_pointer
    ORDER BY updated_at DESC
    LIMIT 100
);
```

### Connection Pool Considerations

The `connection_factory` callable is invoked for each manifest store operation. For production use, return connections from a pool:

```python
import psycopg2.pool

pool = psycopg2.pool.ThreadedConnectionPool(1, 10, "dbname=mydb")

store = PostgresManifestStore(
    connection_factory=pool.getconn,
)
```

## Troubleshooting

### ManifestParseError on Startup

**Symptom:** Reader fails to initialize with `ManifestParseError`.

**Cause:** The current manifest is malformed (schema migration, partial write, corruption).

**Resolution:** The reader's [cold-start fallback](reader.md#cold-start-fallback) automatically tries previous manifests (up to `max_fallback_attempts=3` by default). If all manifests are invalid:

1. Check manifest contents: `uv run shardy --ref <manifest-ref> info`
2. Roll back to a known-good manifest: `uv run shardy rollback --offset 1`
3. Investigate the write pipeline that produced the malformed manifest

### S3 Retry Exhaustion

**Symptom:** `S3TransientError` after multiple retries.

**Cause:** Persistent S3 connectivity issues or rate limiting.

**Resolution:**

1. Check S3 service health
2. Increase retry budget: `RetryConfig(max_retries=5, initial_backoff=timedelta(seconds=2.0))`
3. Monitor `S3_RETRY` and `S3_RETRY_EXHAUSTED` metric events

### Pool Checkout Timeout

**Symptom:** `PoolExhaustedError` from `ConcurrentShardedReader` in pool mode.

**Cause:** All reader copies for a shard are checked out and none returned within `pool_checkout_timeout`.

**Resolution:**

1. Increase `max_workers` to add more reader copies per shard
2. Increase `pool_checkout_timeout` (default 30s)
3. Investigate slow reads that hold checkouts too long

### Rate Limiter in Async Context

**Symptom:** Event loop blocked or `RuntimeError` in async code.

**Cause:** Calling synchronous `acquire()` (which uses `time.sleep`) from async code.

**Resolution:**

- Use `acquire_async()` (uses `asyncio.sleep`) — available on `TokenBucket`
- Or use `try_acquire()` (pure arithmetic, guaranteed non-blocking) paired with `asyncio.sleep`:

```python
result = limiter.try_acquire(tokens)
if not result:
    await asyncio.sleep(result.deficit)
```

See the [reader-side rate limiting](reader.md#reader-side-rate-limiting) docs for async patterns.
