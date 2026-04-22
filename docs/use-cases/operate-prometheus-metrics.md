# Emit Prometheus metrics

Use the **`PrometheusCollector`** to scrape shardyfusion writer/reader metrics with Prometheus.

## When to use

- Your stack already runs Prometheus.
- You want a pull-based scrape model.

## When NOT to use

- You're standardized on OTel / OTLP — use [`operate-otel-metrics.md`](operate-otel-metrics.md).
- You don't need observability — pass `metrics=None` (the default).

## Install

```bash
uv add 'shardyfusion[metrics-prometheus]'
```

`prometheus_client` is lazy-imported inside `PrometheusCollector.__init__`.

## Minimal example

```python
from prometheus_client import CollectorRegistry, start_http_server
from shardyfusion.metrics import PrometheusCollector

registry = CollectorRegistry()
metrics = PrometheusCollector(registry=registry, prefix="shardyfusion_")

start_http_server(9090, registry=registry)

# Pass into any reader/writer
reader = ShardedReader(..., metrics_collector=metrics)
```

## Configuration

`PrometheusCollector(registry=None, prefix="shardyfusion_")`:

- `registry=None` ⇒ uses the default global registry.
- `prefix` is prepended to all metric names.

## Events emitted

`MetricEvent` is a `str, Enum`. Notable values:

- `write.started`, `write.shard.completed`, `write.completed`
- `read.get`, `read.multi_get`
- `s3.retry`
- `vector.reader.search`

The collector translates events into Prometheus counters/histograms. Backends register lazily; first event creates the metric.

## Functional / Non-functional properties

- Pull-based; Prometheus scrapes your HTTP endpoint.
- No background threads inside the collector.

## Guarantees

- Events are emitted via `if metrics is not None`. There is no no-op recorder; passing `None` skips emission.

## Weaknesses

- Process-local registry: in multi-process writers (e.g. parallel Python writer) each subprocess has its own registry. Use the multiprocess Prometheus pattern if you need aggregation.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| `prometheus_client` missing | `ImportError` at construction | Install `metrics-prometheus` extra. |
| Duplicate metric registration | Prometheus error | Reuse a single registry per process. |

## See also

- [`operate-otel-metrics.md`](operate-otel-metrics.md).
- [`architecture/observability.md`](../architecture/observability.md).
