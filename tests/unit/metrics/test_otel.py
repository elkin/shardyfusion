"""Tests for OtelCollector (metrics-otel extra)."""

from __future__ import annotations

import pytest

pytest.importorskip(
    "opentelemetry.sdk", reason="requires metrics-otel extra and opentelemetry-sdk"
)

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from shardyfusion.metrics import MetricEvent
from shardyfusion.metrics.otel import OtelCollector


def _collector() -> tuple[OtelCollector, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    return OtelCollector(meter_provider=provider), reader


def _find_metric(reader: InMemoryMetricReader, name: str) -> list[float]:
    """Extract all data point values for a given metric name.

    Counters expose ``.value``, histograms expose ``.sum``.
    """
    data = reader.get_metrics_data()
    values: list[float] = []
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if metric.name == name:
                    for dp in metric.data.data_points:
                        values.append(getattr(dp, "value", None) or dp.sum)
    return values


class TestWriteStarted:
    def test_counter_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.WRITE_STARTED, {})
        vals = _find_metric(reader, "shardyfusion.writes_started")
        assert len(vals) == 1
        assert vals[0] == 1

    def test_multiple_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.WRITE_STARTED, {})
        collector.emit(MetricEvent.WRITE_STARTED, {})
        vals = _find_metric(reader, "shardyfusion.writes_started")
        assert len(vals) == 1
        assert vals[0] == 2


class TestWriteCompleted:
    def test_counter_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.WRITE_COMPLETED, {"elapsed_ms": 500.0})
        vals = _find_metric(reader, "shardyfusion.writes_completed")
        assert vals[0] == 1

    def test_histogram_records_seconds(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.WRITE_COMPLETED, {"elapsed_ms": 1500.0})
        vals = _find_metric(reader, "shardyfusion.write_duration")
        assert len(vals) == 1
        assert vals[0] == pytest.approx(1.5)

    def test_no_duration_without_elapsed_ms(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.WRITE_COMPLETED, {})
        counter_vals = _find_metric(reader, "shardyfusion.writes_completed")
        assert counter_vals[0] == 1
        hist_vals = _find_metric(reader, "shardyfusion.write_duration")
        assert len(hist_vals) == 0


class TestReaderGet:
    def test_counter_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.READER_GET, {"duration_ms": 10.0})
        vals = _find_metric(reader, "shardyfusion.reader_gets")
        assert vals[0] == 1

    def test_histogram_records_seconds(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.READER_GET, {"duration_ms": 250.0})
        vals = _find_metric(reader, "shardyfusion.reader_get_duration")
        assert vals[0] == pytest.approx(0.25)


class TestS3Retry:
    def test_counter_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.S3_RETRY, {})
        collector.emit(MetricEvent.S3_RETRY, {})
        vals = _find_metric(reader, "shardyfusion.s3_retries")
        assert vals[0] == 2


class TestRateLimiterThrottled:
    def test_counter_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.RATE_LIMITER_THROTTLED, {"wait_seconds": 0.5})
        vals = _find_metric(reader, "shardyfusion.rate_limiter_throttled")
        assert vals[0] == 1

    def test_histogram_records_wait(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.RATE_LIMITER_THROTTLED, {"wait_seconds": 0.3})
        vals = _find_metric(reader, "shardyfusion.rate_limiter_wait")
        assert vals[0] == pytest.approx(0.3)


class TestUnknownEvent:
    def test_all_events_handled_without_error(self) -> None:
        """All current events are handled without error."""
        collector, _reader = _collector()
        for event in MetricEvent:
            collector.emit(event, {})
