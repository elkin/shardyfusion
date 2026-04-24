# Emit OpenTelemetry metrics

Use the **`OtelCollector`** to export shardyfusion writer/reader metrics via OpenTelemetry.

## When to use

- Your stack is standardized on OTel / OTLP.
- You want push-based metric export.

## When NOT to use

- You're standardized on Prometheus — use [Prometheus metrics](prometheus-metrics.md).
- You don't need observability — pass `metrics=None` (the default).

## Install

```bash
uv add 'shardyfusion[metrics-otel]'
```

## Minimal example

```python
from shardyfusion.metrics import OtelCollector

metrics = OtelCollector(prefix="shardyfusion_")

# Pass into any reader/writer
reader = ShardedReader(..., metrics_collector=metrics)
```

## Configuration

`OtelCollector(prefix="shardyfusion_")`:

- `prefix` is prepended to all metric names.
- OTel SDK configuration (endpoint, headers, resource attributes) is picked up from standard OTel env vars (`OTEL_EXPORTER_OTLP_ENDPOINT`, etc.).

## Events emitted

Same event surface as [Prometheus metrics](prometheus-metrics.md):

- `write.started`, `write.shard.completed`, `write.completed`
- `read.get`, `read.multi_get`
- `s3.retry`
- `vector.reader.search`

The collector translates events into OTel counters and histograms.

## Functional / Non-functional properties

- Push-based; metrics are exported via OTLP.
- No background threads inside the collector (relies on OTel SDK's batch processor).

## Guarantees

- Events are emitted via `if metrics is not None`. Passing `None` skips emission.

## Weaknesses

- OTel SDK must be configured externally (env vars or programmatic setup).
- Multi-process writers require per-process OTel SDK instances.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| `opentelemetry` missing | `ImportError` at construction | Install `metrics-otel` extra. |
| OTLP export failure | Logged warning; metrics may be dropped | Check OTel collector connectivity. |

## See also

- [Prometheus metrics](prometheus-metrics.md)
- [`architecture/observability.md`](../architecture/observability.md)
