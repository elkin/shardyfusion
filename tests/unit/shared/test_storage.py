"""Tests for S3 retry metrics in slatedb_spark_sharded.storage."""

from __future__ import annotations

import pytest

from slatedb_spark_sharded.metrics import MetricEvent
from slatedb_spark_sharded.storage import _retry_s3_operation
from slatedb_spark_sharded.testing import ListMetricsCollector


class _FakeTransientS3Error(Exception):
    """Exception that _is_transient_s3_error() recognises as transient."""

    def __init__(self):
        super().__init__("transient")
        self.response = {"Error": {"Code": "500"}}


class TestRetryS3OperationMetrics:
    def test_s3_retry_emits_metric_on_transient_failure(self, monkeypatch):
        """Transient failure then success → one S3_RETRY event."""
        monkeypatch.setattr("slatedb_spark_sharded.storage.time.sleep", lambda _: None)
        mc = ListMetricsCollector()
        calls = 0

        def operation():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _FakeTransientS3Error()
            return "ok"

        result = _retry_s3_operation(
            operation,
            operation_name="test_op",
            url="s3://bucket/key",
            metrics_collector=mc,
        )

        assert result == "ok"
        retry_events = [(e, p) for e, p in mc.events if e is MetricEvent.S3_RETRY]
        assert len(retry_events) == 1
        assert retry_events[0][1]["attempt"] == 1
        assert retry_events[0][1]["max_retries"] == 3
        assert retry_events[0][1]["delay_s"] == 1.0

    def test_s3_retry_exhausted_emits_metric(self, monkeypatch):
        """All retries exhausted → S3_RETRY per attempt + one S3_RETRY_EXHAUSTED."""
        monkeypatch.setattr("slatedb_spark_sharded.storage.time.sleep", lambda _: None)
        mc = ListMetricsCollector()

        def operation():
            raise _FakeTransientS3Error()

        with pytest.raises(_FakeTransientS3Error):
            _retry_s3_operation(
                operation,
                operation_name="test_op",
                url="s3://bucket/key",
                metrics_collector=mc,
            )

        retry_events = [(e, p) for e, p in mc.events if e is MetricEvent.S3_RETRY]
        exhausted_events = [
            (e, p) for e, p in mc.events if e is MetricEvent.S3_RETRY_EXHAUSTED
        ]
        # 3 retries (attempts 0,1,2 succeed to sleep; attempt 3 exhausts)
        assert len(retry_events) == 3
        assert len(exhausted_events) == 1
        assert exhausted_events[0][1]["attempts"] == 4

    def test_s3_retry_no_metrics_when_collector_is_none(self, monkeypatch):
        """No crash when metrics_collector is None."""
        monkeypatch.setattr("slatedb_spark_sharded.storage.time.sleep", lambda _: None)
        calls = 0

        def operation():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise _FakeTransientS3Error()
            return "ok"

        result = _retry_s3_operation(
            operation,
            operation_name="test_op",
            url="s3://bucket/key",
            metrics_collector=None,
        )

        assert result == "ok"

    def test_s3_retry_no_metric_on_immediate_success(self):
        """Immediate success → no retry metrics emitted."""
        mc = ListMetricsCollector()

        result = _retry_s3_operation(
            lambda: "ok",
            operation_name="test_op",
            url="s3://bucket/key",
            metrics_collector=mc,
        )

        assert result == "ok"
        assert len(mc.events) == 0
