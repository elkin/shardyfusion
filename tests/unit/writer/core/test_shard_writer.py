"""Tests for shardyfusion._shard_writer (shared write core + retry)."""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shardyfusion._shard_writer import (
    ShardWriteParams,
    iter_pandas_rows,
    make_shard_local_dir,
    make_shard_url,
    results_pdf_to_attempts,
    write_shard_core,
    write_shard_with_retry,
    write_shard_with_retry_distributed,
)
from shardyfusion.errors import (
    ConfigValidationError,
    ShardWriteError,
)
from shardyfusion.manifest import WriterInfo
from shardyfusion.metrics import MetricEvent
from shardyfusion.serde import ValueSpec, make_key_encoder
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.type_defs import RetryConfig
from tests.helpers.tracking import TrackingFactory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_params(
    tmp_path: Path,
    *,
    db_id: int = 0,
    attempt: int = 0,
    batch_size: int = 100,
    factory: Any | None = None,
    ops_limiter: Any | None = None,
    bytes_limiter: Any | None = None,
    metrics_collector: Any | None = None,
) -> ShardWriteParams:
    return ShardWriteParams(
        db_id=db_id,
        attempt=attempt,
        run_id="test-run",
        db_url=f"s3://bucket/prefix/db={db_id:05d}/attempt={attempt:02d}",
        local_dir=tmp_path / f"db={db_id:05d}" / f"attempt={attempt:02d}",
        factory=factory or TrackingFactory(),
        key_encoder=make_key_encoder(KeyEncoding.U64BE),
        batch_size=batch_size,
        ops_limiter=ops_limiter,
        bytes_limiter=bytes_limiter,
        metrics_collector=metrics_collector,
        started=time.perf_counter(),
    )


def _simple_rows(n: int = 5) -> list[tuple[object, bytes]]:
    """Create n simple (int_key, value_bytes) rows."""
    return [(i, f"value-{i}".encode()) for i in range(n)]


# ---------------------------------------------------------------------------
# write_shard_core
# ---------------------------------------------------------------------------


class TestWriteShardCore:
    def test_basic_write(self, tmp_path: Path) -> None:
        factory = TrackingFactory()
        params = _make_params(tmp_path, factory=factory)
        rows = _simple_rows(3)

        result = write_shard_core(params, iter(rows))

        assert result.db_id == 0
        assert result.attempt == 0
        assert result.row_count == 3
        assert result.min_key == 0
        assert result.max_key == 2
        assert result.checkpoint_id == "fake-checkpoint"
        assert result.db_url == params.db_url

        # Adapter was opened and used
        assert len(factory.adapters) == 1
        adapter = factory.adapters[0]
        assert adapter.flushed
        assert adapter.closed

    def test_batch_flushing(self, tmp_path: Path) -> None:
        """Batches are flushed when they reach batch_size."""
        factory = TrackingFactory()
        params = _make_params(tmp_path, factory=factory, batch_size=3)
        rows = _simple_rows(7)  # 3 + 3 + 1 = three flushes

        write_shard_core(params, iter(rows))

        adapter = factory.adapters[0]
        assert len(adapter.write_calls) == 3
        assert len(adapter.write_calls[0]) == 3
        assert len(adapter.write_calls[1]) == 3
        assert len(adapter.write_calls[2]) == 1

    def test_empty_rows(self, tmp_path: Path) -> None:
        """Writing zero rows produces a result with row_count=0."""
        factory = TrackingFactory()
        params = _make_params(tmp_path, factory=factory)

        result = write_shard_core(params, iter([]))

        assert result.row_count == 0
        assert result.min_key is None
        assert result.max_key is None
        adapter = factory.adapters[0]
        assert len(adapter.write_calls) == 0
        assert adapter.flushed

    def test_min_max_key_tracking(self, tmp_path: Path) -> None:
        factory = TrackingFactory()
        params = _make_params(tmp_path, factory=factory)
        rows = [(10, b"a"), (3, b"b"), (99, b"c"), (7, b"d")]

        result = write_shard_core(params, iter(rows))

        assert result.min_key == 3
        assert result.max_key == 99

    def test_ops_limiter_called(self, tmp_path: Path) -> None:
        limiter = MagicMock()
        factory = TrackingFactory()
        params = _make_params(
            tmp_path,
            factory=factory,
            batch_size=2,
            ops_limiter=limiter,
        )

        write_shard_core(params, iter(_simple_rows(5)))

        # 2 + 2 + 1 = three flushes, limiter.acquire called each time
        assert limiter.acquire.call_count == 3
        limiter.acquire.assert_any_call(2)
        limiter.acquire.assert_any_call(1)

    def test_bytes_limiter_called(self, tmp_path: Path) -> None:
        limiter = MagicMock()
        factory = TrackingFactory()
        params = _make_params(
            tmp_path,
            factory=factory,
            batch_size=100,
            bytes_limiter=limiter,
        )

        write_shard_core(params, iter(_simple_rows(2)))

        assert limiter.acquire.call_count == 1

    def test_metrics_emitted(self, tmp_path: Path) -> None:
        mc = MagicMock()
        factory = TrackingFactory()
        params = _make_params(
            tmp_path,
            factory=factory,
            batch_size=2,
            metrics_collector=mc,
        )

        write_shard_core(params, iter(_simple_rows(3)))

        events = [call.args[0] for call in mc.emit.call_args_list]
        assert MetricEvent.SHARD_WRITE_STARTED in events
        assert MetricEvent.BATCH_WRITTEN in events
        assert MetricEvent.SHARD_WRITE_COMPLETED in events

    def test_unknown_exception_wrapped_as_shard_write_error(
        self,
        tmp_path: Path,
    ) -> None:
        """Non-ShardyfusionError from adapter is wrapped as ShardWriteError."""

        class BrokenFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> Any:
                adapter = MagicMock()
                adapter.__enter__ = MagicMock(return_value=adapter)
                adapter.__exit__ = MagicMock(return_value=False)
                adapter.write_batch.side_effect = OSError("disk full")
                return adapter

        params = _make_params(tmp_path, factory=BrokenFactory())

        with pytest.raises(ShardWriteError, match="disk full") as exc_info:
            write_shard_core(params, iter(_simple_rows(1)))

        assert exc_info.value.retryable is True
        assert isinstance(exc_info.value.__cause__, OSError)

    def test_shardyfusion_error_not_wrapped(self, tmp_path: Path) -> None:
        """ShardyfusionError from adapter passes through without wrapping."""

        class ErrorFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> Any:
                adapter = MagicMock()
                adapter.__enter__ = MagicMock(return_value=adapter)
                adapter.__exit__ = MagicMock(return_value=False)
                adapter.write_batch.side_effect = ConfigValidationError("bad key")
                return adapter

        params = _make_params(tmp_path, factory=ErrorFactory())

        with pytest.raises(ConfigValidationError, match="bad key"):
            write_shard_core(params, iter(_simple_rows(1)))

    def test_writer_info_base_fields_preserved(self, tmp_path: Path) -> None:
        factory = TrackingFactory()
        params = _make_params(tmp_path, factory=factory)
        base = WriterInfo(stage_id=42, task_attempt_id=7)

        result = write_shard_core(
            params,
            iter(_simple_rows(1)),
            writer_info_base=base,
        )

        assert result.writer_info.stage_id == 42
        assert result.writer_info.task_attempt_id == 7
        assert result.writer_info.attempt == 0

    def test_local_dir_created(self, tmp_path: Path) -> None:
        factory = TrackingFactory()
        local_dir = tmp_path / "deep" / "nested" / "dir"
        params = ShardWriteParams(
            db_id=0,
            attempt=0,
            run_id="r",
            db_url="s3://b/p",
            local_dir=local_dir,
            factory=factory,
            key_encoder=make_key_encoder(KeyEncoding.U64BE),
            batch_size=100,
            ops_limiter=None,
            bytes_limiter=None,
            metrics_collector=None,
            started=time.perf_counter(),
        )

        write_shard_core(params, iter([]))

        assert local_dir.is_dir()


# ---------------------------------------------------------------------------
# write_shard_with_retry
# ---------------------------------------------------------------------------


class TestWriteShardWithRetry:
    def _common_kwargs(
        self,
        tmp_path: Path,
        factory: Any | None = None,
    ) -> dict[str, Any]:
        return dict(
            db_id=0,
            run_id="test-run",
            s3_prefix="s3://bucket/prefix",
            shard_prefix="shards",
            db_path_template="db={db_id:05d}",
            local_root=str(tmp_path),
            key_encoder=make_key_encoder(KeyEncoding.U64BE),
            batch_size=100,
            factory=factory or TrackingFactory(),
            max_writes_per_second=None,
            max_write_bytes_per_second=None,
            metrics_collector=None,
            started=time.perf_counter(),
        )

    def test_no_retry_config_single_attempt(self, tmp_path: Path) -> None:
        """retry_config=None produces a single attempt."""
        factory = TrackingFactory()
        result = write_shard_with_retry(
            **self._common_kwargs(tmp_path, factory=factory),
            rows_fn=lambda: iter(_simple_rows(3)),
            retry_config=None,
        )

        assert result.attempt == 0
        assert result.row_count == 3
        assert len(factory.adapters) == 1

    @patch("shardyfusion._shard_writer.time.sleep")
    def test_retry_on_retryable_error(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Retryable error triggers retry with backoff."""
        call_count = 0

        class FailOnceFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> Any:
                nonlocal call_count
                call_count += 1
                adapter = MagicMock()
                adapter.__enter__ = MagicMock(return_value=adapter)
                adapter.__exit__ = MagicMock(return_value=False)
                if call_count == 1:
                    adapter.write_batch.side_effect = OSError("transient S3")
                else:
                    adapter.write_batch.return_value = None
                    adapter.flush.return_value = None
                    adapter.checkpoint.return_value = "ckpt-2"
                    adapter.db_bytes.return_value = 0
                return adapter

        result = write_shard_with_retry(
            **self._common_kwargs(tmp_path, factory=FailOnceFactory()),
            rows_fn=lambda: iter(_simple_rows(2)),
            retry_config=RetryConfig(
                max_retries=2, initial_backoff=timedelta(seconds=0.5)
            ),
        )

        assert result.attempt == 1
        assert result.row_count == 2
        assert result.all_attempt_urls == (
            "s3://bucket/prefix/shards/run_id=test-run/db=00000/attempt=00",
            "s3://bucket/prefix/shards/run_id=test-run/db=00000/attempt=01",
        )
        assert call_count == 2
        mock_sleep.assert_called_once_with(0.5)

    def test_non_retryable_error_no_retry(self, tmp_path: Path) -> None:
        """Non-retryable error raises immediately without retry."""

        class FailFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> Any:
                adapter = MagicMock()
                adapter.__enter__ = MagicMock(return_value=adapter)
                adapter.__exit__ = MagicMock(return_value=False)
                adapter.write_batch.side_effect = ConfigValidationError("bad")
                return adapter

        with pytest.raises(ConfigValidationError, match="bad"):
            write_shard_with_retry(
                **self._common_kwargs(tmp_path, factory=FailFactory()),
                rows_fn=lambda: iter(_simple_rows(1)),
                retry_config=RetryConfig(max_retries=3),
            )

    @patch("shardyfusion._shard_writer.time.sleep")
    def test_all_retries_exhausted(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """All attempts fail -> final error raised."""

        class AlwaysFailFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> Any:
                adapter = MagicMock()
                adapter.__enter__ = MagicMock(return_value=adapter)
                adapter.__exit__ = MagicMock(return_value=False)
                adapter.write_batch.side_effect = OSError("persistent failure")
                return adapter

        with pytest.raises(ShardWriteError, match="persistent failure"):
            write_shard_with_retry(
                **self._common_kwargs(tmp_path, factory=AlwaysFailFactory()),
                rows_fn=lambda: iter(_simple_rows(1)),
                retry_config=RetryConfig(
                    max_retries=2, initial_backoff=timedelta(seconds=0.1)
                ),
            )

        # 2 retries = 2 sleeps (attempt 0 fails -> sleep -> attempt 1 fails -> sleep -> attempt 2 fails -> raise)
        assert mock_sleep.call_count == 2

    @patch("shardyfusion._shard_writer.time.sleep")
    def test_each_attempt_gets_new_url(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Each retry attempt writes to a different S3 path."""
        urls_seen: list[str] = []

        class TrackUrlFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> Any:
                urls_seen.append(db_url)
                adapter = MagicMock()
                adapter.__enter__ = MagicMock(return_value=adapter)
                adapter.__exit__ = MagicMock(return_value=False)
                if len(urls_seen) < 3:
                    adapter.write_batch.side_effect = OSError("fail")
                else:
                    adapter.write_batch.return_value = None
                    adapter.flush.return_value = None
                    adapter.checkpoint.return_value = "ckpt"
                    adapter.db_bytes.return_value = 0
                return adapter

        write_shard_with_retry(
            **self._common_kwargs(tmp_path, factory=TrackUrlFactory()),
            rows_fn=lambda: iter(_simple_rows(1)),
            retry_config=RetryConfig(max_retries=3),
        )

        assert len(urls_seen) == 3
        assert "attempt=00" in urls_seen[0]
        assert "attempt=01" in urls_seen[1]
        assert "attempt=02" in urls_seen[2]

    @patch("shardyfusion._shard_writer.time.sleep")
    def test_retry_metrics_emitted(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        mc = MagicMock()
        call_count = 0

        class FailOnceFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> Any:
                nonlocal call_count
                call_count += 1
                adapter = MagicMock()
                adapter.__enter__ = MagicMock(return_value=adapter)
                adapter.__exit__ = MagicMock(return_value=False)
                if call_count == 1:
                    adapter.write_batch.side_effect = OSError("transient")
                else:
                    adapter.write_batch.return_value = None
                    adapter.flush.return_value = None
                    adapter.checkpoint.return_value = "ckpt"
                    adapter.db_bytes.return_value = 0
                return adapter

        write_shard_with_retry(
            **{
                **self._common_kwargs(tmp_path, factory=FailOnceFactory()),
                "metrics_collector": mc,
            },
            rows_fn=lambda: iter(_simple_rows(1)),
            retry_config=RetryConfig(max_retries=2),
        )

        events = [call.args[0] for call in mc.emit.call_args_list]
        assert MetricEvent.SHARD_WRITE_RETRIED in events

    @patch("shardyfusion._shard_writer.time.sleep")
    def test_exhausted_metric_emitted(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        mc = MagicMock()

        class AlwaysFailFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> Any:
                adapter = MagicMock()
                adapter.__enter__ = MagicMock(return_value=adapter)
                adapter.__exit__ = MagicMock(return_value=False)
                adapter.write_batch.side_effect = OSError("fail")
                return adapter

        with pytest.raises(ShardWriteError):
            write_shard_with_retry(
                **{
                    **self._common_kwargs(tmp_path, factory=AlwaysFailFactory()),
                    "metrics_collector": mc,
                },
                rows_fn=lambda: iter(_simple_rows(1)),
                retry_config=RetryConfig(max_retries=1),
            )

        events = [call.args[0] for call in mc.emit.call_args_list]
        assert MetricEvent.SHARD_WRITE_RETRY_EXHAUSTED in events

    @patch("shardyfusion._shard_writer.time.sleep")
    def test_exponential_backoff(
        self,
        mock_sleep: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Backoff increases exponentially."""

        class AlwaysFailFactory:
            def __call__(self, *, db_url: str, local_dir: Path) -> Any:
                adapter = MagicMock()
                adapter.__enter__ = MagicMock(return_value=adapter)
                adapter.__exit__ = MagicMock(return_value=False)
                adapter.write_batch.side_effect = OSError("fail")
                return adapter

        with pytest.raises(ShardWriteError):
            write_shard_with_retry(
                **self._common_kwargs(tmp_path, factory=AlwaysFailFactory()),
                rows_fn=lambda: iter(_simple_rows(1)),
                retry_config=RetryConfig(
                    max_retries=3,
                    initial_backoff=timedelta(seconds=1.0),
                    backoff_multiplier=2.0,
                ),
            )

        sleeps = [call.args[0] for call in mock_sleep.call_args_list]
        assert sleeps == [1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# URL / dir helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_make_shard_url(self) -> None:
        url = make_shard_url(
            "s3://bucket/prefix",
            "shards",
            "run-123",
            "db={db_id:05d}",
            7,
            2,
        )
        assert url == "s3://bucket/prefix/shards/run_id=run-123/db=00007/attempt=02"

    def test_make_shard_local_dir(self) -> None:
        d = make_shard_local_dir("/tmp/sf", "run-1", 3, 1)
        assert d == Path("/tmp/sf/run_id=run-1/db=00003/attempt=01")


class TestPandasInteropHelpers:
    def test_iter_pandas_rows_coerces_numpy_scalar_keys(self) -> None:
        pd = pytest.importorskip("pandas")
        np = pytest.importorskip("numpy")

        pdf = pd.DataFrame({"id": [np.int64(7)], "value": ["a"]})
        value_spec = ValueSpec.callable_encoder(lambda row: str(row["value"]).encode())

        rows = list(iter_pandas_rows(pdf, "id", value_spec))
        assert rows == [(7, b"a")]

    def test_results_pdf_to_attempts_normalizes_nan_and_attempt_urls(self) -> None:
        pd = pytest.importorskip("pandas")

        writer_info = WriterInfo(attempt=0, duration_ms=3)
        pdf = pd.DataFrame(
            [
                {
                    "db_id": 1,
                    "db_url": "s3://bucket/db=00001/attempt=00",
                    "attempt": 0,
                    "row_count": 2,
                    "min_key": 1.0,
                    "max_key": float("nan"),
                    "checkpoint_id": float("nan"),
                    "writer_info": writer_info,
                    "all_attempt_urls": [
                        "s3://bucket/db=00001/attempt=00",
                        "s3://bucket/db=00001/attempt=01",
                    ],
                }
            ]
        )

        attempts = results_pdf_to_attempts(pdf)
        assert len(attempts) == 1
        attempt = attempts[0]
        assert attempt.db_id == 1
        assert attempt.min_key == 1
        assert attempt.max_key is None
        assert attempt.checkpoint_id is None
        assert attempt.all_attempt_urls == (
            "s3://bucket/db=00001/attempt=00",
            "s3://bucket/db=00001/attempt=01",
        )

    def test_results_pdf_to_attempts_empty_returns_empty(self) -> None:
        pd = pytest.importorskip("pandas")
        assert results_pdf_to_attempts(pd.DataFrame()) == []


class TestWriteShardWithRetryDistributed:
    def test_distributed_retry_creates_rate_limiters(self, tmp_path: Path) -> None:
        bucket_calls: list[tuple[float, str]] = []

        class _TokenBucketStub:
            def __init__(
                self,
                rate: float,
                *,
                metrics_collector: Any | None = None,
                limiter_type: str = "ops",
            ) -> None:
                del metrics_collector
                bucket_calls.append((rate, limiter_type))

            def acquire(self, tokens: int) -> None:
                del tokens

        class _VectorAdapter:
            def __enter__(self) -> _VectorAdapter:
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

            def write_batch(self, batch: list[tuple[bytes, bytes]]) -> None:
                del batch

            def write_vector_batch(
                self,
                ids: Any,
                vectors: Any,
                payloads: list[dict[str, Any] | None] | None = None,
            ) -> None:
                del ids, vectors, payloads

            def flush(self) -> None:
                pass

            def checkpoint(self) -> str:
                return "ckpt"

            def db_bytes(self) -> int:
                return 0

        with patch("shardyfusion._shard_writer.TokenBucket", _TokenBucketStub):
            result = write_shard_with_retry_distributed(
                db_id=0,
                rows_fn=lambda: iter([(1, b"v1", ("id-1", [0.1, 0.2], None))]),
                run_id="run-1",
                s3_prefix="s3://bucket/prefix",
                shard_prefix="shards",
                db_path_template="db={db_id:05d}",
                local_root=str(tmp_path),
                key_encoder=make_key_encoder(KeyEncoding.U64BE),
                batch_size=1,
                factory=lambda *, db_url, local_dir: _VectorAdapter(),
                max_writes_per_second=100.0,
                max_write_bytes_per_second=1000.0,
                metrics_collector=None,
                started=time.perf_counter(),
                retry_config=None,
            )

        assert result.attempt == 0
        assert bucket_calls == [(100.0, "ops"), (1000.0, "bytes")]


# ---------------------------------------------------------------------------
# _require_db_bytes — strict validation for manifest v4
# ---------------------------------------------------------------------------


class TestRequireDbBytes:
    """Adapters MUST implement db_bytes() returning a non-negative int.

    Manifest v4 makes db_bytes mandatory in RequiredShardMeta, so the shard
    writer no longer tolerates missing/invalid implementations.
    """

    def _factory(self, adapter: Any) -> Any:
        def _make(*, db_url: str, local_dir: Path) -> Any:
            return adapter

        return _make

    def _adapter_base(self) -> MagicMock:
        adapter = MagicMock()
        adapter.__enter__ = MagicMock(return_value=adapter)
        adapter.__exit__ = MagicMock(return_value=False)
        adapter.write_batch.return_value = None
        adapter.flush.return_value = None
        adapter.checkpoint.return_value = "ckpt"
        return adapter

    def test_missing_db_bytes_attribute_raises(self, tmp_path: Path) -> None:
        adapter = self._adapter_base()
        # Force missing attribute by spec'ing it away
        del adapter.db_bytes
        params = _make_params(tmp_path, factory=self._factory(adapter))

        with pytest.raises(ShardWriteError, match="does not implement db_bytes"):
            write_shard_core(params, iter(_simple_rows(1)))

    def test_db_bytes_not_callable_raises(self, tmp_path: Path) -> None:
        adapter = self._adapter_base()
        adapter.db_bytes = 123  # int, not callable
        params = _make_params(tmp_path, factory=self._factory(adapter))

        with pytest.raises(ShardWriteError, match="does not implement db_bytes"):
            write_shard_core(params, iter(_simple_rows(1)))

    def test_db_bytes_returns_non_int_raises(self, tmp_path: Path) -> None:
        adapter = self._adapter_base()
        # MagicMock auto-returns a MagicMock from db_bytes() unless configured.
        params = _make_params(tmp_path, factory=self._factory(adapter))

        with pytest.raises(ShardWriteError, match="invalid db_bytes"):
            write_shard_core(params, iter(_simple_rows(1)))

    def test_db_bytes_negative_raises(self, tmp_path: Path) -> None:
        adapter = self._adapter_base()
        adapter.db_bytes.return_value = -1
        params = _make_params(tmp_path, factory=self._factory(adapter))

        with pytest.raises(ShardWriteError, match="invalid db_bytes"):
            write_shard_core(params, iter(_simple_rows(1)))

    def test_db_bytes_zero_is_accepted(self, tmp_path: Path) -> None:
        """db_bytes=0 is the canonical SlateDB value; must succeed."""
        adapter = self._adapter_base()
        adapter.db_bytes.return_value = 0
        params = _make_params(tmp_path, factory=self._factory(adapter))

        result = write_shard_core(params, iter(_simple_rows(1)))
        assert result.row_count == 1

