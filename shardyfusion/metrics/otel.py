"""OpenTelemetry metrics publisher for shardyfusion.

Requires the ``metrics-otel`` extra: ``pip install shardyfusion[metrics-otel]``
"""

from __future__ import annotations

from typing import Any

from ._events import MetricEvent


class OtelCollector:
    """Maps MetricEvent to OpenTelemetry Counter and Histogram instruments.

    Constructor: OtelCollector(meter_provider=None, meter_name="shardyfusion")
    Uses the global MeterProvider if none provided.
    """

    def __init__(
        self,
        meter_provider: Any = None,
        meter_name: str = "shardyfusion",
    ) -> None:
        from opentelemetry import metrics as otel_metrics

        if meter_provider is not None:
            meter = meter_provider.get_meter(meter_name)
        else:
            meter = otel_metrics.get_meter(meter_name)

        # Writer lifecycle
        self._writes_started = meter.create_counter(
            "shardyfusion.writes_started", description="Write operations started"
        )
        self._writes_completed = meter.create_counter(
            "shardyfusion.writes_completed", description="Write operations completed"
        )
        self._write_duration = meter.create_histogram(
            "shardyfusion.write_duration",
            unit="s",
            description="Write operation duration",
        )
        self._shard_writes_completed = meter.create_counter(
            "shardyfusion.shard_writes_completed", description="Shard writes completed"
        )
        self._shard_write_duration = meter.create_histogram(
            "shardyfusion.shard_write_duration",
            unit="s",
            description="Shard write duration",
        )
        self._batches_written = meter.create_counter(
            "shardyfusion.batches_written", description="Write batches committed"
        )

        # Reader lifecycle
        self._reader_gets = meter.create_counter(
            "shardyfusion.reader_gets", description="Reader get operations"
        )
        self._reader_get_duration = meter.create_histogram(
            "shardyfusion.reader_get_duration",
            unit="s",
            description="Reader get duration",
        )
        self._reader_multi_gets = meter.create_counter(
            "shardyfusion.reader_multi_gets", description="Reader multi_get operations"
        )
        self._reader_multi_get_duration = meter.create_histogram(
            "shardyfusion.reader_multi_get_duration",
            unit="s",
            description="Reader multi_get duration",
        )

        # Infrastructure
        self._s3_retries = meter.create_counter(
            "shardyfusion.s3_retries", description="S3 retries"
        )
        self._s3_retries_exhausted = meter.create_counter(
            "shardyfusion.s3_retries_exhausted", description="S3 retries exhausted"
        )
        self._rate_limiter_throttled = meter.create_counter(
            "shardyfusion.rate_limiter_throttled",
            description="Rate limiter throttle events",
        )
        self._rate_limiter_wait = meter.create_histogram(
            "shardyfusion.rate_limiter_wait",
            unit="s",
            description="Rate limiter wait duration",
        )
        self._rate_limiter_denied = meter.create_counter(
            "shardyfusion.rate_limiter_denied", description="Rate limiter denial events"
        )

        # Circuit breaker
        self._circuit_breaker_opened = meter.create_counter(
            "shardyfusion.circuit_breaker_opened", description="Circuit breaker opened"
        )
        self._circuit_breaker_closed = meter.create_counter(
            "shardyfusion.circuit_breaker_closed", description="Circuit breaker closed"
        )
        self._circuit_breaker_rejected = meter.create_counter(
            "shardyfusion.circuit_breaker_rejected",
            description="Circuit breaker rejections",
        )

    def emit(self, event: MetricEvent, payload: dict[str, Any]) -> None:
        """Handle a metric event by updating the appropriate OTel instruments."""
        if event == MetricEvent.WRITE_STARTED:
            self._writes_started.add(1)
        elif event == MetricEvent.WRITE_COMPLETED:
            self._writes_completed.add(1)
            if "elapsed_ms" in payload:
                self._write_duration.record(payload["elapsed_ms"] / 1000.0)
        elif event == MetricEvent.SHARD_WRITE_COMPLETED:
            self._shard_writes_completed.add(1)
            if "duration_ms" in payload:
                self._shard_write_duration.record(payload["duration_ms"] / 1000.0)
        elif event == MetricEvent.BATCH_WRITTEN:
            self._batches_written.add(1)
        elif event == MetricEvent.READER_GET:
            self._reader_gets.add(1)
            if "duration_ms" in payload:
                self._reader_get_duration.record(payload["duration_ms"] / 1000.0)
        elif event == MetricEvent.READER_MULTI_GET:
            self._reader_multi_gets.add(1)
            if "duration_ms" in payload:
                self._reader_multi_get_duration.record(payload["duration_ms"] / 1000.0)
        elif event == MetricEvent.S3_RETRY:
            self._s3_retries.add(1)
        elif event == MetricEvent.S3_RETRY_EXHAUSTED:
            self._s3_retries_exhausted.add(1)
        elif event == MetricEvent.RATE_LIMITER_THROTTLED:
            self._rate_limiter_throttled.add(1)
            if "wait_seconds" in payload:
                self._rate_limiter_wait.record(payload["wait_seconds"])
        elif event == MetricEvent.RATE_LIMITER_DENIED:
            self._rate_limiter_denied.add(1)
        elif event == MetricEvent.CIRCUIT_BREAKER_OPENED:
            self._circuit_breaker_opened.add(1)
        elif event == MetricEvent.CIRCUIT_BREAKER_CLOSED:
            self._circuit_breaker_closed.add(1)
        elif event == MetricEvent.CIRCUIT_BREAKER_REJECTED:
            self._circuit_breaker_rejected.add(1)
        # Unknown events silently ignored
