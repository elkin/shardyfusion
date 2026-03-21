"""Tests for PrometheusCollector (metrics-prometheus extra)."""

from __future__ import annotations

import pytest
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


def _histogram_sum(registry: CollectorRegistry, name: str) -> float:
    return registry.get_sample_value(f"{name}_sum") or 0.0


class TestWriteStarted:
    def test_counter_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.WRITE_STARTED, {})
        assert _counter_value(reg, "shardyfusion_writes_started") == 1.0

    def test_multiple_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.WRITE_STARTED, {})
        collector.emit(MetricEvent.WRITE_STARTED, {})
        assert _counter_value(reg, "shardyfusion_writes_started") == 2.0


class TestWriteCompleted:
    def test_counter_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.WRITE_COMPLETED, {"elapsed_ms": 500.0})
        assert _counter_value(reg, "shardyfusion_writes_completed") == 1.0

    def test_histogram_observes_seconds(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.WRITE_COMPLETED, {"elapsed_ms": 1500.0})
        assert _histogram_count(reg, "shardyfusion_write_duration_seconds") == 1.0
        assert _histogram_sum(
            reg, "shardyfusion_write_duration_seconds"
        ) == pytest.approx(1.5)

    def test_no_duration_without_elapsed_ms(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.WRITE_COMPLETED, {})
        assert _counter_value(reg, "shardyfusion_writes_completed") == 1.0
        assert _histogram_count(reg, "shardyfusion_write_duration_seconds") == 0.0


class TestReaderGet:
    def test_counter_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.READER_GET, {"duration_ms": 10.0})
        assert _counter_value(reg, "shardyfusion_reader_gets") == 1.0

    def test_histogram_observes_seconds(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.READER_GET, {"duration_ms": 250.0})
        assert _histogram_sum(
            reg, "shardyfusion_reader_get_duration_seconds"
        ) == pytest.approx(0.25)


class TestS3Retry:
    def test_counter_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.S3_RETRY, {})
        collector.emit(MetricEvent.S3_RETRY, {})
        assert _counter_value(reg, "shardyfusion_s3_retries") == 2.0


class TestRateLimiterThrottled:
    def test_counter_increments(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.RATE_LIMITER_THROTTLED, {"wait_seconds": 0.5})
        assert _counter_value(reg, "shardyfusion_rate_limiter_throttled") == 1.0

    def test_histogram_observes_wait(self) -> None:
        collector, reg = _collector()
        collector.emit(MetricEvent.RATE_LIMITER_THROTTLED, {"wait_seconds": 0.3})
        assert _histogram_sum(
            reg, "shardyfusion_rate_limiter_wait_seconds"
        ) == pytest.approx(0.3)


class TestUnknownEvent:
    def test_silently_ignored(self) -> None:
        """Unknown/future events must not raise."""
        collector, _reg = _collector()
        # Since MetricEvent is an enum, we can't create a truly unknown value,
        # but we verify all current events are handled without error.
        for event in MetricEvent:
            collector.emit(event, {})


class TestRegistryIsolation:
    def test_two_collectors_independent(self) -> None:
        """Two collectors with separate registries don't share state."""
        c1, r1 = _collector()
        c2, r2 = _collector()
        c1.emit(MetricEvent.WRITE_STARTED, {})
        assert _counter_value(r1, "shardyfusion_writes_started") == 1.0
        assert _counter_value(r2, "shardyfusion_writes_started") == 0.0
