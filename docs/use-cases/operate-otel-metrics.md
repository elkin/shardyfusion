# Emit OpenTelemetry metrics

Use the **`OtelCollector`** to emit shardyfusion writer/reader metrics through OpenTelemetry.

## When to use

- Your stack is OTel-native (OTLP exporter, OTel collector).
- You want push-based metric export.

## When NOT to use

- Pull-based Prometheus is preferred — use [`operate-prometheus-metrics.md`](operate-prometheus-metrics.md).

## Install

```bash
uv add 'shardyfusion[metrics-otel]'
```

`opentelemetry-api` is lazy-imported inside `OtelCollector.__init__`.

## Minimal example

```python
from opentelemetry import metrics as otel_metrics
from opentelemetry.sdk.metrics import MeterProvider
from shardyfusion.metrics import OtelCollector

provider = MeterProvider(metric_readers=[...])  # your OTLP/Prometheus reader
otel_metrics.set_meter_provider(provider)

metrics = OtelCollector(meter_provider=provider, meter_name="shardyfusion")

reader = ShardedReader(..., metrics_collector=metrics)
```

## Configuration

`OtelCollector(meter_provider=None, meter_name="shardyfusion")`:

- `meter_provider=None` ⇒ uses the global meter provider.
- `meter_name` becomes the OTel `Meter` name.

## Events emitted

Same `MetricEvent` set as the Prometheus backend (see [`operate-prometheus-metrics.md`](operate-prometheus-metrics.md)). Translated into OTel counters/histograms.

## Functional / Non-functional properties

- Push-based via the configured `MeterProvider`.
- Lazy instrument creation on first event.

## Guarantees

- `metrics=None` skips emission entirely (no overhead).

## Weaknesses

- You own the `MeterProvider` lifecycle (export interval, batching, shutdown).
- No built-in span/trace integration — metrics only.

## Failure modes & recovery

| Failure | Surface | Recovery |
|---|---|---|
| `opentelemetry-api` missing | `ImportError` at construction | Install `metrics-otel` extra. |
| No exporter configured | Silent drop | Wire OTLP/Console exporter on the provider. |

## See also

- [`operate-prometheus-metrics.md`](operate-prometheus-metrics.md).
- [`architecture/observability.md`](../architecture/observability.md).
