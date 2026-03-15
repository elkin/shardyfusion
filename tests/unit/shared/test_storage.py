"""Tests for S3 retry metrics and utilities in shardyfusion.storage."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from shardyfusion.metrics import MetricEvent
from shardyfusion.storage import _retry_s3_operation, list_prefixes
from shardyfusion.testing import ListMetricsCollector


class _FakeTransientS3Error(Exception):
    """Exception that _is_transient_s3_error() recognises as transient."""

    def __init__(self):
        super().__init__("transient")
        self.response = {"Error": {"Code": "500"}}


class TestRetryS3OperationMetrics:
    def test_s3_retry_emits_metric_on_transient_failure(self, monkeypatch):
        """Transient failure then success → one S3_RETRY event."""
        monkeypatch.setattr("shardyfusion.storage.time.sleep", lambda _: None)
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
        monkeypatch.setattr("shardyfusion.storage.time.sleep", lambda _: None)
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
        monkeypatch.setattr("shardyfusion.storage.time.sleep", lambda _: None)
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


# ---------------------------------------------------------------------------
# list_prefixes
# ---------------------------------------------------------------------------


def _make_listing_client(pages: list[list[str]]) -> MagicMock:
    """Build a mock S3 client whose ``list_objects_v2`` returns paginated CommonPrefixes.

    Each inner list contains S3 key prefixes (e.g. ``["prefix/run_id=a/"]``).
    The mock returns them as consecutive pages with proper ``IsTruncated``
    and ``NextContinuationToken`` fields.
    """
    mock_client = MagicMock()
    responses = []
    for i, page_prefixes in enumerate(pages):
        is_last = i == len(pages) - 1
        response: dict[str, object] = {
            "CommonPrefixes": [{"Prefix": p} for p in page_prefixes],
            "IsTruncated": not is_last,
        }
        if not is_last:
            response["NextContinuationToken"] = f"token-{i + 1}"
        responses.append(response)
    mock_client.list_objects_v2.side_effect = responses
    return mock_client


class TestListPrefixes:
    def test_single_page(self) -> None:
        """Single page of results returns full S3 URLs, sorted."""
        client = _make_listing_client(
            [["prefix/shards/run_id=bbb/", "prefix/shards/run_id=aaa/"]]
        )
        result = list_prefixes("s3://bucket/prefix/shards/", s3_client=client)
        assert result == [
            "s3://bucket/prefix/shards/run_id=aaa/",
            "s3://bucket/prefix/shards/run_id=bbb/",
        ]

    def test_multi_page(self) -> None:
        """Multiple pages are consumed — all results returned."""
        client = _make_listing_client(
            [
                ["prefix/shards/run_id=aaa/"],
                ["prefix/shards/run_id=bbb/"],
                ["prefix/shards/run_id=ccc/"],
            ]
        )
        result = list_prefixes("s3://bucket/prefix/shards/", s3_client=client)
        assert len(result) == 3
        assert result == [
            "s3://bucket/prefix/shards/run_id=aaa/",
            "s3://bucket/prefix/shards/run_id=bbb/",
            "s3://bucket/prefix/shards/run_id=ccc/",
        ]

    def test_empty_prefix(self) -> None:
        """No children returns empty list."""
        client = _make_listing_client([[]])
        result = list_prefixes("s3://bucket/prefix/shards/", s3_client=client)
        assert result == []

    def test_urls_are_full_s3(self) -> None:
        """Returned URLs start with s3://bucket/."""
        client = _make_listing_client([["some/path/"]])
        result = list_prefixes("s3://mybucket/some/", s3_client=client)
        assert all(url.startswith("s3://mybucket/") for url in result)

    def test_trailing_slash_added(self) -> None:
        """Prefix without trailing slash still works."""
        client = _make_listing_client([["prefix/shards/run_id=a/"]])
        result = list_prefixes("s3://bucket/prefix/shards", s3_client=client)
        assert len(result) == 1
        # Verify the Prefix kwarg had a trailing slash appended
        call_kwargs = client.list_objects_v2.call_args.kwargs
        assert call_kwargs["Prefix"].endswith("/")
