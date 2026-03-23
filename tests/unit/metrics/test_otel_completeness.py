"""Tests for OTel collector event completeness.

Exercises MetricEvent values that are handled by OtelCollector
but were not covered by the existing test_otel.py.
"""

from __future__ import annotations

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from shardyfusion.metrics import MetricEvent
from shardyfusion.metrics.otel import OtelCollector


def _collector() -> tuple[OtelCollector, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    return OtelCollector(meter_provider=provider), reader


def _find_metric(reader: InMemoryMetricReader, name: str) -> list[float]:
    data = reader.get_metrics_data()
    values: list[float] = []
    for rm in data.resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                if metric.name == name:
                    for dp in metric.data.data_points:
                        values.append(getattr(dp, "value", None) or dp.sum)
    return values


class TestShardWriteCompleted:
    def test_counter_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.SHARD_WRITE_COMPLETED, {"duration_ms": 100.0})
        vals = _find_metric(reader, "shardyfusion.shard_writes_completed")
        assert len(vals) == 1 and vals[0] >= 1


class TestBatchWritten:
    def test_counter_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.BATCH_WRITTEN, {})
        collector.emit(MetricEvent.BATCH_WRITTEN, {})
        vals = _find_metric(reader, "shardyfusion.batches_written")
        assert len(vals) == 1 and vals[0] >= 2


class TestReaderMultiGet:
    def test_counter_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.READER_MULTI_GET, {"duration_ms": 50.0})
        vals = _find_metric(reader, "shardyfusion.reader_multi_gets")
        assert len(vals) == 1 and vals[0] >= 1


class TestS3RetryExhausted:
    def test_counter_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.S3_RETRY_EXHAUSTED, {})
        vals = _find_metric(reader, "shardyfusion.s3_retries_exhausted")
        assert len(vals) == 1 and vals[0] >= 1


class TestRateLimiterDenied:
    def test_counter_increments(self) -> None:
        collector, reader = _collector()
        collector.emit(MetricEvent.RATE_LIMITER_DENIED, {})
        vals = _find_metric(reader, "shardyfusion.rate_limiter_denied")
        assert len(vals) == 1 and vals[0] >= 1


class TestSilentlyIgnoredEvents:
    """Events not explicitly handled should not raise."""

    def test_all_events_accepted(self) -> None:
        collector, _reader = _collector()
        for event in MetricEvent:
            collector.emit(
                event, {"elapsed_ms": 100, "duration_ms": 50, "wait_seconds": 0.1}
            )
