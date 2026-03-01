"""Tests for slatedb_spark_sharded.metrics module."""

from slatedb_spark_sharded.metrics import MetricEvent, emit
from slatedb_spark_sharded.testing import ListMetricsCollector


class TestMetricEvent:
    def test_enum_values_are_stable(self):
        """Verify enum values don't change — external collectors may depend on them."""
        assert MetricEvent.WRITE_STARTED == "write.started"
        assert MetricEvent.WRITE_COMPLETED == "write.completed"
        assert MetricEvent.SHARD_WRITE_STARTED == "write.shard.started"
        assert MetricEvent.SHARD_WRITE_COMPLETED == "write.shard.completed"
        assert MetricEvent.SHARD_WRITES_COMPLETED == "write.shards_completed"
        assert MetricEvent.SHARDING_COMPLETED == "write.sharding_completed"
        assert MetricEvent.MANIFEST_PUBLISHED == "write.manifest_published"
        assert MetricEvent.CURRENT_PUBLISHED == "write.current_published"
        assert MetricEvent.BATCH_WRITTEN == "write.batch_written"
        assert MetricEvent.READER_INITIALIZED == "reader.initialized"
        assert MetricEvent.READER_GET == "reader.get"
        assert MetricEvent.READER_MULTI_GET == "reader.multi_get"
        assert MetricEvent.READER_REFRESHED == "reader.refreshed"
        assert MetricEvent.READER_CLOSED == "reader.closed"
        assert MetricEvent.S3_RETRY == "s3.retry"
        assert MetricEvent.S3_RETRY_EXHAUSTED == "s3.retry_exhausted"
        assert MetricEvent.RATE_LIMITER_THROTTLED == "rate_limiter.throttled"


class TestEmit:
    def test_noop_when_collector_is_none(self):
        """emit() with None collector must not raise."""
        emit(None, MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})

    def test_delegates_to_collector(self):
        mc = ListMetricsCollector()
        emit(mc, MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})
        assert len(mc.events) == 1
        event, payload = mc.events[0]
        assert event is MetricEvent.WRITE_STARTED
        assert payload == {"elapsed_ms": 0}

    def test_payload_is_copied(self):
        """Collector receives a copy — mutating original doesn't affect stored."""
        mc = ListMetricsCollector()
        original = {"elapsed_ms": 100}
        emit(mc, MetricEvent.WRITE_COMPLETED, original)
        original["elapsed_ms"] = 999
        assert mc.events[0][1]["elapsed_ms"] == 100


class TestListMetricsCollector:
    def test_collects_multiple_events(self):
        mc = ListMetricsCollector()
        mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})
        mc.emit(MetricEvent.SHARD_WRITE_COMPLETED, {"elapsed_ms": 50, "row_count": 10})
        mc.emit(MetricEvent.WRITE_COMPLETED, {"elapsed_ms": 100, "rows_written": 10})
        assert len(mc.events) == 3

    def test_events_are_ordered(self):
        mc = ListMetricsCollector()
        mc.emit(MetricEvent.WRITE_STARTED, {})
        mc.emit(MetricEvent.WRITE_COMPLETED, {})
        assert mc.events[0][0] is MetricEvent.WRITE_STARTED
        assert mc.events[1][0] is MetricEvent.WRITE_COMPLETED

    def test_starts_empty(self):
        mc = ListMetricsCollector()
        assert mc.events == []
