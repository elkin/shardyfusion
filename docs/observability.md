# Observability

## MetricsCollector Protocol

shardyfusion emits structured metric events throughout the write and read lifecycle. To observe them, implement the `MetricsCollector` protocol:

```python
from shardyfusion.metrics import MetricEvent, MetricsCollector

class MyCollector:
    """Example collector that counts events."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.last_payload: dict[str, dict] = {}

    def emit(self, event: MetricEvent, payload: dict) -> None:
        self.counts[event.value] = self.counts.get(event.value, 0) + 1
        self.last_payload[event.value] = payload
```

Pass the collector to the writer or reader:

```python
collector = MyCollector()

# Writer
result = write_sharded(records, config, ..., metrics_collector=collector)

# Or via config
config = WriteConfig(num_dbs=8, s3_prefix="...", metrics_collector=collector)

# Reader
reader = ShardedReader(
    s3_prefix="...", local_root="/tmp/reader",
    metrics_collector=collector,
)
```

**Implementation requirements:**

- Must be thread-safe (called from multiple threads in concurrent reader)
- Called synchronously — buffer internally if blocking is a concern
- Should silently ignore unknown events for forward compatibility

## MetricEvent Catalog

### Writer Lifecycle

| Event | Payload | Description |
|---|---|---|
| `WRITE_STARTED` | `elapsed_ms` | Write pipeline initiated |
| `SHARDING_COMPLETED` | `elapsed_ms`, `duration_ms` | Shard assignment phase complete |
| `SHARD_WRITE_STARTED` | `elapsed_ms`, `db_id` | Individual shard write begins |
| `SHARD_WRITE_COMPLETED` | `elapsed_ms`, `db_id`, `row_count`, `duration_ms` | Individual shard write finishes |
| `SHARD_WRITES_COMPLETED` | `elapsed_ms`, `duration_ms`, `rows_written` | All shards written |
| `BATCH_WRITTEN` | `elapsed_ms`, `db_id`, `batch_size` | Single batch flushed to adapter |
| `MANIFEST_PUBLISHED` | `elapsed_ms` | Manifest uploaded to S3 |
| `CURRENT_PUBLISHED` | `elapsed_ms` | CURRENT pointer updated |
| `WRITE_COMPLETED` | `elapsed_ms`, `rows_written` | Entire write pipeline done |

### Reader Lifecycle

| Event | Payload | Description |
|---|---|---|
| `READER_INITIALIZED` | _(empty)_ | Reader constructed and state loaded |
| `READER_GET` | `duration_ms`, `found` | Single key lookup completed |
| `READER_MULTI_GET` | `duration_ms`, `num_keys` | Multi-key lookup completed |
| `READER_REFRESHED` | `changed` | Refresh attempt (changed=True if new snapshot loaded) |
| `READER_CLOSED` | `num_handles` | Reader closed |

### Infrastructure

| Event | Payload | Description |
|---|---|---|
| `S3_RETRY` | `attempt`, `max_retries`, `delay_s` | S3 operation being retried |
| `S3_RETRY_EXHAUSTED` | `attempts` | All S3 retry attempts failed |
| `RATE_LIMITER_THROTTLED` | `wait_seconds` | Rate limiter imposed a wait |

## Logging

shardyfusion uses Python's `logging` module with the `shardyfusion` logger namespace. Key log events are emitted as structured keyword arguments:

```python
import logging
logging.getLogger("shardyfusion").setLevel(logging.DEBUG)
```

Notable log events:

- `write_started` / `write_completed` — pipeline lifecycle
- `winner_selected` — shard winner selection (DEBUG level)
- `reader_initialized` / `reader_refreshed` / `reader_closed` — reader lifecycle
- `reader_state_acquired` / `reader_state_released` — refcount changes (DEBUG level)
- `manifest_published` / `current_published` — publish milestones
- `s3_retry` — transient S3 retries with attempt and delay

## Example: Monitoring Dashboard Integration

```python
import time
from shardyfusion.metrics import MetricEvent

class PrometheusCollector:
    """Sketch: emit Prometheus-style metrics."""

    def emit(self, event: MetricEvent, payload: dict) -> None:
        match event:
            case MetricEvent.READER_GET:
                histogram_observe("reader_get_duration_ms", payload["duration_ms"])
                counter_inc("reader_get_total", labels={"found": str(payload["found"])})
            case MetricEvent.S3_RETRY:
                counter_inc("s3_retries_total")
            case MetricEvent.RATE_LIMITER_THROTTLED:
                histogram_observe("rate_limiter_delay_s", payload["wait_seconds"])
            case _:
                pass  # Ignore unknown events
```
