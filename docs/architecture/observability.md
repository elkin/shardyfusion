# Observability

`shardyfusion/metrics/` provides optional Prometheus and OpenTelemetry integrations. The core writer and reader code emit metrics through a backend-agnostic `MetricsCollector` protocol; concrete backends are wired up only if the corresponding extra is installed.

shardyfusion **emits metrics only**. There is no built-in tracing/spans surface — applications wanting traces should instrument their own call sites around the writer/reader entry points.

## Modules

| Module | Purpose |
|---|---|
| `shardyfusion/metrics/_protocol.py` | `MetricsCollector` protocol (`record_event`, `record_value`). |
| `shardyfusion/metrics/_events.py` | `MetricEvent(str, Enum)` — canonical event names. |
| `shardyfusion/metrics/prometheus.py` | `PrometheusCollector(registry=None, prefix="shardyfusion_")`. |
| `shardyfusion/metrics/otel.py` | `OtelCollector(meter_provider=None, meter_name="shardyfusion")`. |
| `shardyfusion/metrics/__init__.py` | Re-exports. |

Both backend collectors lazily import their dependency (`prometheus_client`, `opentelemetry`) inside `__init__`, so the module is importable even if the extra is missing — but instantiation fails fast with a clear `pip install shardyfusion[<extra>]` message.

## Backends

| Backend | Extra | Class |
|---|---|---|
| Prometheus | `metrics-prometheus` | `PrometheusCollector` |
| OpenTelemetry | `metrics-otel` | `OtelCollector` |
| Default (no metrics) | (none) | `None` |

The writer/reader entry points accept `metrics: MetricsCollector | None`. **Passing `None` (default) skips all emission entirely** — there is no no-op recorder class; emission is gated by an `if metrics is not None` check at each call site, so the runtime cost when disabled is a single attribute test.

## Event surface

Events are values of `MetricEvent` (a `str` `Enum`). Selected examples — the canonical list lives in `shardyfusion/metrics/_events.py`:

Writer:

- `write.started`
- `write.shard.completed`
- `write.shard.failed`
- `write.completed`
- `write.failed`

Manifest / publish:

- `manifest.built`
- `manifest.published`
- `current.published`

S3:

- `s3.retry`

Reader:

- `reader.shard.opened`
- `reader.shard.closed`

Vector:

- `vector.reader.search`

For the canonical, current list, query the source — `validate-docs` checks that any event name documented in use-case pages matches a real enum value.

## Wiring

```python
from shardyfusion.metrics import PrometheusCollector
from shardyfusion.writer.python import write_sharded

collector = PrometheusCollector()  # uses global default registry
write_sharded(records, config, key_fn=..., value_fn=..., metrics=collector)
```

Or with OTel:

```python
from shardyfusion.metrics import OtelCollector

collector = OtelCollector()  # uses global default MeterProvider
```

## Cardinality

Per-key labels are deliberately **not** emitted. Per-shard labels are emitted (cardinality bounded by `num_dbs`). Per-run labels are emitted (cardinality grows with retention; clean up old runs).

## What is *not* exported

- Per-key access patterns (cardinality explosion).
- Raw S3 response bodies.
- Manifest contents.

If you want application-level observability of which keys are read, instrument your application — not shardyfusion.

## See also

- [`optional-imports.md`](optional-imports.md) — how the extras gate is implemented.
- [`operate/prometheus-metrics.md`](../operate/prometheus-metrics.md).
- [`operate/otel-metrics.md`](../operate/otel-metrics.md).
