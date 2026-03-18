"""Prometheus metrics publisher for shardyfusion.

Requires the ``metrics-prometheus`` extra: ``pip install shardyfusion[metrics-prometheus]``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._events import MetricEvent

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry


class PrometheusCollector:
    """Maps MetricEvent to Prometheus counters and histograms.

    Constructor: PrometheusCollector(registry=REGISTRY, prefix="shardyfusion_")
    Accepts optional ``registry`` for test isolation.
    Duration payloads (elapsed_ms, duration_ms) are converted to seconds for Prometheus convention.
    Unknown events are silently ignored.
    """

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        prefix: str = "shardyfusion_",
    ) -> None:
        from prometheus_client import REGISTRY, Counter, Histogram

        self._registry = registry or REGISTRY
        p = prefix

        # Writer lifecycle
        self._writes_started = Counter(
            f"{p}writes_started_total",
            "Write operations started",
            registry=self._registry,
        )
        self._writes_completed = Counter(
            f"{p}writes_completed_total",
            "Write operations completed",
            registry=self._registry,
        )
        self._write_duration = Histogram(
            f"{p}write_duration_seconds",
            "Write operation duration",
            registry=self._registry,
        )
        self._shard_writes_completed = Counter(
            f"{p}shard_writes_completed_total",
            "Shard writes completed",
            registry=self._registry,
        )
        self._shard_write_duration = Histogram(
            f"{p}shard_write_duration_seconds",
            "Shard write duration",
            registry=self._registry,
        )
        self._batches_written = Counter(
            f"{p}batches_written_total",
            "Write batches committed",
            registry=self._registry,
        )

        # Reader lifecycle
        self._reader_gets = Counter(
            f"{p}reader_gets_total", "Reader get operations", registry=self._registry
        )
        self._reader_get_duration = Histogram(
            f"{p}reader_get_duration_seconds",
            "Reader get duration",
            registry=self._registry,
        )
        self._reader_multi_gets = Counter(
            f"{p}reader_multi_gets_total",
            "Reader multi_get operations",
            registry=self._registry,
        )
        self._reader_multi_get_duration = Histogram(
            f"{p}reader_multi_get_duration_seconds",
            "Reader multi_get duration",
            registry=self._registry,
        )

        # Infrastructure
        self._s3_retries = Counter(
            f"{p}s3_retries_total", "S3 retries", registry=self._registry
        )
        self._s3_retries_exhausted = Counter(
            f"{p}s3_retries_exhausted_total",
            "S3 retries exhausted",
            registry=self._registry,
        )
        self._rate_limiter_throttled = Counter(
            f"{p}rate_limiter_throttled_total",
            "Rate limiter throttle events",
            registry=self._registry,
        )
        self._rate_limiter_wait = Histogram(
            f"{p}rate_limiter_wait_seconds",
            "Rate limiter wait duration",
            registry=self._registry,
        )
        self._rate_limiter_denied = Counter(
            f"{p}rate_limiter_denied_total",
            "Rate limiter denial events",
            registry=self._registry,
        )

    def emit(self, event: MetricEvent, payload: dict[str, Any]) -> None:
        """Handle a metric event by updating the appropriate Prometheus instruments."""
        if event == MetricEvent.WRITE_STARTED:
            self._writes_started.inc()
        elif event == MetricEvent.WRITE_COMPLETED:
            self._writes_completed.inc()
            if "elapsed_ms" in payload:
                self._write_duration.observe(payload["elapsed_ms"] / 1000.0)
        elif event == MetricEvent.SHARD_WRITE_COMPLETED:
            self._shard_writes_completed.inc()
            if "duration_ms" in payload:
                self._shard_write_duration.observe(payload["duration_ms"] / 1000.0)
        elif event == MetricEvent.BATCH_WRITTEN:
            self._batches_written.inc()
        elif event == MetricEvent.READER_GET:
            self._reader_gets.inc()
            if "duration_ms" in payload:
                self._reader_get_duration.observe(payload["duration_ms"] / 1000.0)
        elif event == MetricEvent.READER_MULTI_GET:
            self._reader_multi_gets.inc()
            if "duration_ms" in payload:
                self._reader_multi_get_duration.observe(payload["duration_ms"] / 1000.0)
        elif event == MetricEvent.S3_RETRY:
            self._s3_retries.inc()
        elif event == MetricEvent.S3_RETRY_EXHAUSTED:
            self._s3_retries_exhausted.inc()
        elif event == MetricEvent.RATE_LIMITER_THROTTLED:
            self._rate_limiter_throttled.inc()
            if "wait_seconds" in payload:
                self._rate_limiter_wait.observe(payload["wait_seconds"])
        elif event == MetricEvent.RATE_LIMITER_DENIED:
            self._rate_limiter_denied.inc()
        # Unknown events are silently ignored (forward-compatible)
