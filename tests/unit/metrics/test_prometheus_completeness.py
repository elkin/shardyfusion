"""Tests for Prometheus collector event completeness.

Exercises MetricEvent values that are handled by PrometheusCollector
but were not covered by the existing test_prometheus.py.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry

from shardyfusion.metrics import MetricEvent
from shardyfusion.metrics.prometheus import PrometheusCollector


def _collector() -> tuple[PrometheusCollector, CollectorRegistry]:
    registry = CollectorRegistry()
    return PrometheusCollector(registry=registry), registry


def _counter_value(registry: CollectorRegistry, name: str) -> float:
    return registry.get_sample_value(f"{name}_total") or 0.0


def _histogram_count(registry: CollectorRegistry, name: str) -> float:
    return registry.get_sample_value(f"{name}_count") or 0.0


class TestShardWriteCompleted:
    def test_counter_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.SHARD_WRITE_COMPLETED, {"duration_ms": 100.0})
        assert _counter_value(reg, "shardyfusion_shard_writes_completed") == 1.0

    def test_histogram_observes(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.SHARD_WRITE_COMPLETED, {"duration_ms": 2000.0})
        assert _histogram_count(reg, "shardyfusion_shard_write_duration_seconds") == 1.0


class TestBatchWritten:
    def test_counter_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.BATCH_WRITTEN, {})
        collector.emit(MetricEvent.BATCH_WRITTEN, {})
        assert _counter_value(reg, "shardyfusion_batches_written") == 2.0


class TestReaderMultiGet:
    def test_counter_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.READER_MULTI_GET, {"duration_ms": 50.0})
        assert _counter_value(reg, "shardyfusion_reader_multi_gets") == 1.0

    def test_histogram_observes(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.READER_MULTI_GET, {"duration_ms": 300.0})
        assert (
            _histogram_count(reg, "shardyfusion_reader_multi_get_duration_seconds")
            == 1.0
        )


class TestS3RetryExhausted:
    def test_counter_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.S3_RETRY_EXHAUSTED, {})
        assert _counter_value(reg, "shardyfusion_s3_retries_exhausted") == 1.0


class TestRateLimiterDenied:
    def test_counter_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.RATE_LIMITER_DENIED, {})
        assert _counter_value(reg, "shardyfusion_rate_limiter_denied") == 1.0


class TestSilentlyIgnoredEvents:
    """Events not mapped to Prometheus instruments should not raise."""

    def test_sharding_completed(self) -> None:
        collector, _reg = _collector()
        collector.emit(MetricEvent.SHARDING_COMPLETED, {"elapsed_ms": 100})

    def test_manifest_published(self) -> None:
        collector, _reg = _collector()
        collector.emit(MetricEvent.MANIFEST_PUBLISHED, {"elapsed_ms": 50})

    def test_current_published(self) -> None:
        collector, _reg = _collector()
        collector.emit(MetricEvent.CURRENT_PUBLISHED, {"elapsed_ms": 30})

    def test_reader_initialized(self) -> None:
        collector, _reg = _collector()
        collector.emit(MetricEvent.READER_INITIALIZED, {})

    def test_reader_refreshed(self) -> None:
        collector, _reg = _collector()
        collector.emit(MetricEvent.READER_REFRESHED, {})

    def test_reader_closed(self) -> None:
        collector, _reg = _collector()
        collector.emit(MetricEvent.READER_CLOSED, {})

    def test_shard_write_retried(self) -> None:
        collector, _reg = _collector()
        collector.emit(MetricEvent.SHARD_WRITE_RETRIED, {"db_id": 0, "attempt": 1})

    def test_shard_write_retry_exhausted(self) -> None:
        collector, _reg = _collector()
        collector.emit(MetricEvent.SHARD_WRITE_RETRY_EXHAUSTED, {"db_id": 0})
