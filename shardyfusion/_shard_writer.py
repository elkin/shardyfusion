"""Shared shard-write logic used by Dask, Ray, and Spark writers.

This module extracts the duplicated inner loop from each framework's
``_write_one_shard()`` function into a single ``write_shard_core()``
function, and adds an optional retry wrapper via ``write_shard_with_retry()``.

Spark calls ``write_shard_core()`` directly (Spark handles retries via
speculative execution).  Dask and Ray call ``write_shard_with_retry()``
which adds configurable retry with exponential backoff.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from ._rate_limiter import RateLimiter, TokenBucket
from ._writer_core import ShardAttemptResult, update_min_max
from .errors import ShardWriteError, ShardyfusionError
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .manifest import WriterInfo
from .metrics import MetricEvent, MetricsCollector
from .serde import KeyEncoder
from .slatedb_adapter import DbAdapterFactory
from .storage import join_s3
from .type_defs import KeyInput, RetryConfig

_logger = get_logger(__name__)

VectorWriteTuple = tuple[int | str, Any, dict[str, Any] | None]


def _require_db_bytes(adapter: Any, db_id: int) -> int:
    """Return ``adapter.db_bytes()``, raising on missing/invalid implementations.

    Manifest v4 makes ``db_bytes`` mandatory in :class:`RequiredShardMeta` and
    the adaptive SQLite reader uses these values to decide between download
    and range-read modes.  Silently substituting ``0`` for adapters that omit
    the method would underestimate snapshot size and may push the policy to
    pick ``download`` for very large shards (causing OOM at read time), so
    we surface the misconfiguration immediately instead.
    """
    db_bytes_attr = getattr(adapter, "db_bytes", None)
    if db_bytes_attr is None or not callable(db_bytes_attr):
        raise ShardWriteError(
            f"Adapter for db_id={db_id} does not implement db_bytes(); "
            f"all DbAdapter implementations must report shard size for "
            f"manifest v4. Adapter type: {type(adapter).__name__}"
        )
    value = db_bytes_attr()
    if not isinstance(value, int) or value < 0:
        raise ShardWriteError(
            f"Adapter for db_id={db_id} returned invalid db_bytes={value!r}; "
            f"expected a non-negative int."
        )
    return value


class _PandasRowLike(Protocol):
    def __getitem__(self, key: str) -> object: ...


class _PandasFrameLike(Protocol):
    def iterrows(self) -> Iterable[tuple[object, _PandasRowLike]]: ...


# ---------------------------------------------------------------------------
# Parameters dataclass
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ShardWriteParams:
    """Parameters for a single shard write attempt."""

    db_id: int
    attempt: int
    run_id: str
    db_url: str
    local_dir: Path
    factory: DbAdapterFactory
    key_encoder: KeyEncoder
    batch_size: int
    ops_limiter: RateLimiter | None
    bytes_limiter: RateLimiter | None
    metrics_collector: MetricsCollector | None
    started: float


@dataclass(slots=True, frozen=True)
class _RetryAttemptContext:
    """Materialized URL/dir context for one retryable attempt."""

    attempt: int
    db_url: str
    local_dir: Path
    prior_attempt_urls: tuple[str, ...]


# ---------------------------------------------------------------------------
# URL / directory helpers
# ---------------------------------------------------------------------------


def make_shard_url(
    s3_prefix: str,
    shard_prefix: str,
    run_id: str,
    db_path_template: str,
    db_id: int,
    attempt: int,
) -> str:
    """Build the S3 URL for one shard attempt."""
    db_rel_path = db_path_template.format(db_id=db_id)
    return join_s3(
        s3_prefix,
        shard_prefix,
        f"run_id={run_id}",
        db_rel_path,
        f"attempt={attempt:02d}",
    )


def make_shard_local_dir(
    local_root: str,
    run_id: str,
    db_id: int,
    attempt: int,
) -> Path:
    """Build the local directory for one shard attempt."""
    return (
        Path(local_root)
        / f"run_id={run_id}"
        / f"db={db_id:05d}"
        / f"attempt={attempt:02d}"
    )


# ---------------------------------------------------------------------------
# Pandas row iterator (shared by Dask and Ray)
# ---------------------------------------------------------------------------


def iter_pandas_rows(
    pdf: object,
    key_col: str,
    value_spec: object,
) -> Iterable[tuple[object, bytes]]:
    """Yield ``(raw_key, encoded_value)`` pairs from a pandas DataFrame.

    Handles numpy scalar coercion for downstream compatibility.
    Accepts ``object`` types to avoid a hard pandas/serde import at module
    level (this module is imported by framework-specific writers that
    already have these dependencies).
    """
    from .serde import ValueSpec as _ValueSpec

    frame = cast(_PandasFrameLike, pdf)
    spec: _ValueSpec = value_spec  # type: ignore[assignment]
    for _, row in frame.iterrows():
        key_value = row[key_col]
        item = getattr(key_value, "item", None)
        if callable(item):
            key_value = item()
        yield key_value, spec.encode(row)


# ---------------------------------------------------------------------------
# Core shard writer (single attempt, no retry)
# ---------------------------------------------------------------------------


def write_shard_core(
    params: ShardWriteParams,
    rows: Iterable[tuple[object, bytes]],
    *,
    writer_info_base: WriterInfo | None = None,
    prior_attempt_urls: tuple[str, ...] = (),
) -> ShardAttemptResult:
    """Write one shard attempt.

    This function contains the shared inner loop previously duplicated
    across the Spark, Dask, and Ray writers:

    1. Create local directory
    2. Open adapter via factory
    3. Iterate rows — encode key, accumulate batch, track min/max
    4. Flush batch when full (with rate limiting)
    5. Flush remaining + checkpoint
    6. Build and return ``ShardAttemptResult``

    Args:
        params: Immutable parameters for this attempt.
        rows: Iterable of ``(raw_key, encoded_value_bytes)`` pairs.
            The raw key is passed through ``params.key_encoder`` and
            used for min/max tracking.
        writer_info_base: Optional ``WriterInfo`` with framework-specific
            fields pre-populated (e.g. Spark ``stage_id``, ``task_attempt_id``).
            Missing fields are filled in by this function.
    """
    mc = params.metrics_collector

    params.local_dir.mkdir(parents=True, exist_ok=True)

    log_event(
        "shard_write_started",
        logger=_logger,
        run_id=params.run_id,
        db_id=params.db_id,
        attempt=params.attempt,
        db_url=params.db_url,
    )
    if mc is not None:
        mc.emit(
            MetricEvent.SHARD_WRITE_STARTED,
            {"elapsed_ms": int((time.perf_counter() - params.started) * 1000)},
        )

    partition_started = time.perf_counter()
    row_count = 0
    min_key: KeyInput | None = None
    max_key: KeyInput | None = None
    checkpoint_id: str | None = None
    batch: list[tuple[bytes, bytes]] = []

    try:
        with params.factory(
            db_url=params.db_url, local_dir=params.local_dir
        ) as adapter:
            for key_value, value_bytes in rows:
                key_bytes = params.key_encoder(key_value)
                batch.append((key_bytes, value_bytes))
                row_count += 1
                min_key, max_key = update_min_max(min_key, max_key, key_value)  # type: ignore[arg-type]  # object is KeyInput at runtime

                if len(batch) >= params.batch_size:
                    _flush_batch(batch, adapter, params, mc)

            if batch:
                _flush_batch(batch, adapter, params, mc)

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
            db_bytes = _require_db_bytes(adapter, params.db_id)
    except ShardyfusionError:
        raise
    except Exception as exc:
        log_failure(
            "shard_write_failed",
            severity=FailureSeverity.ERROR,
            logger=_logger,
            error=exc,
            run_id=params.run_id,
            db_id=params.db_id,
            attempt=params.attempt,
            db_url=params.db_url,
            rows_written=row_count,
            include_traceback=True,
        )
        raise ShardWriteError(
            f"Shard write failed for db_id={params.db_id}, "
            f"attempt={params.attempt}: {exc}"
        ) from exc

    duration_ms = int((time.perf_counter() - partition_started) * 1000)
    log_event(
        "shard_write_completed",
        logger=_logger,
        run_id=params.run_id,
        db_id=params.db_id,
        attempt=params.attempt,
        row_count=row_count,
        duration_ms=duration_ms,
    )
    if mc is not None:
        mc.emit(
            MetricEvent.SHARD_WRITE_COMPLETED,
            {
                "elapsed_ms": int((time.perf_counter() - params.started) * 1000),
                "duration_ms": duration_ms,
                "row_count": row_count,
            },
        )

    info = WriterInfo(
        attempt=params.attempt,
        duration_ms=duration_ms,
        stage_id=writer_info_base.stage_id if writer_info_base else None,
        task_attempt_id=writer_info_base.task_attempt_id if writer_info_base else None,
    )

    return ShardAttemptResult(
        db_id=params.db_id,
        db_url=params.db_url,
        attempt=params.attempt,
        row_count=row_count,
        min_key=min_key,
        max_key=max_key,
        checkpoint_id=checkpoint_id,
        writer_info=info,
        db_bytes=db_bytes,
        all_attempt_urls=(*prior_attempt_urls, params.db_url),
    )


def _flush_batch(
    batch: list[tuple[bytes, bytes]],
    adapter: object,
    params: ShardWriteParams,
    mc: MetricsCollector | None,
) -> None:
    """Flush one accumulated batch through rate limiters and adapter."""
    if params.ops_limiter is not None:
        params.ops_limiter.acquire(len(batch))
    if params.bytes_limiter is not None:
        params.bytes_limiter.acquire(sum(len(k) + len(v) for k, v in batch))
    adapter.write_batch(batch)  # type: ignore[union-attr]
    if mc is not None:
        mc.emit(
            MetricEvent.BATCH_WRITTEN,
            {
                "elapsed_ms": int((time.perf_counter() - params.started) * 1000),
                "batch_size": len(batch),
            },
        )
    batch.clear()


def _flush_vector_batch(
    vector_ids: list[int | str],
    vector_data: list[Any],
    vector_payloads: list[dict[str, Any] | None],
    adapter: object,
) -> None:
    if not vector_ids:
        return
    vector_writer = getattr(adapter, "write_vector_batch", None)
    if not callable(vector_writer):
        raise ShardWriteError("Adapter does not support vector writes in unified mode")
    import numpy as np

    ids_arr = np.array(vector_ids)
    vectors_arr = np.array(vector_data, dtype=np.float32)
    vector_writer(ids_arr, vectors_arr, vector_payloads)
    vector_ids.clear()
    vector_data.clear()
    vector_payloads.clear()


def write_shard_core_distributed(
    params: ShardWriteParams,
    rows: Iterable[tuple[object, bytes, VectorWriteTuple | None]],
    *,
    writer_info_base: WriterInfo | None = None,
    prior_attempt_urls: tuple[str, ...] = (),
) -> ShardAttemptResult:
    """Distributed shard writer supporting unified KV+vector batches."""
    mc = params.metrics_collector
    params.local_dir.mkdir(parents=True, exist_ok=True)
    partition_started = time.perf_counter()
    row_count = 0
    min_key: KeyInput | None = None
    max_key: KeyInput | None = None
    checkpoint_id: str | None = None
    batch: list[tuple[bytes, bytes]] = []
    vector_ids: list[int | str] = []
    vector_data: list[Any] = []
    vector_payloads: list[dict[str, Any] | None] = []

    try:
        with params.factory(
            db_url=params.db_url, local_dir=params.local_dir
        ) as adapter:
            for key_value, value_bytes, vector_item in rows:
                key_bytes = params.key_encoder(key_value)
                batch.append((key_bytes, value_bytes))
                row_count += 1
                min_key, max_key = update_min_max(min_key, max_key, key_value)  # type: ignore[arg-type]
                if vector_item is not None:
                    vec_id, vec_data, vec_payload = vector_item
                    vector_ids.append(vec_id)
                    vector_data.append(vec_data)
                    vector_payloads.append(vec_payload)
                if len(batch) >= params.batch_size:
                    _flush_batch(batch, adapter, params, mc)
                    _flush_vector_batch(
                        vector_ids, vector_data, vector_payloads, adapter
                    )

            if batch:
                _flush_batch(batch, adapter, params, mc)
            _flush_vector_batch(vector_ids, vector_data, vector_payloads, adapter)
            adapter.flush()
            checkpoint_id = adapter.checkpoint()
            db_bytes = _require_db_bytes(adapter, params.db_id)
    except ShardyfusionError:
        raise
    except Exception as exc:
        raise ShardWriteError(
            f"Shard write failed for db_id={params.db_id}, attempt={params.attempt}: {exc}"
        ) from exc

    duration_ms = int((time.perf_counter() - partition_started) * 1000)
    info = WriterInfo(
        attempt=params.attempt,
        duration_ms=duration_ms,
        stage_id=writer_info_base.stage_id if writer_info_base else None,
        task_attempt_id=writer_info_base.task_attempt_id if writer_info_base else None,
    )
    return ShardAttemptResult(
        db_id=params.db_id,
        db_url=params.db_url,
        attempt=params.attempt,
        row_count=row_count,
        min_key=min_key,
        max_key=max_key,
        checkpoint_id=checkpoint_id,
        writer_info=info,
        db_bytes=db_bytes,
        all_attempt_urls=(*prior_attempt_urls, params.db_url),
    )


# ---------------------------------------------------------------------------
# Retry wrapper (used by Dask and Ray, NOT Spark)
# ---------------------------------------------------------------------------


def _run_attempts_with_retry(
    *,
    db_id: int,
    run_id: str,
    s3_prefix: str,
    shard_prefix: str,
    db_path_template: str,
    local_root: str,
    retry_config: RetryConfig | None,
    metrics_collector: MetricsCollector | None,
    started: float,
    attempt_fn: Callable[[_RetryAttemptContext], ShardAttemptResult],
) -> ShardAttemptResult:
    """Run ``attempt_fn`` with retry-specific attempt IDs and metadata."""

    max_attempts = 1 + (retry_config.max_retries if retry_config else 0)
    delay = retry_config.initial_backoff.total_seconds() if retry_config else 0.0
    multiplier = retry_config.backoff_multiplier if retry_config else 1.0

    last_exc: BaseException | None = None
    failed_attempt_urls: list[str] = []

    for attempt in range(max_attempts):
        db_url = make_shard_url(
            s3_prefix,
            shard_prefix,
            run_id,
            db_path_template,
            db_id,
            attempt,
        )
        local_dir = make_shard_local_dir(local_root, run_id, db_id, attempt)
        ctx = _RetryAttemptContext(
            attempt=attempt,
            db_url=db_url,
            local_dir=local_dir,
            prior_attempt_urls=tuple(failed_attempt_urls),
        )

        try:
            return attempt_fn(ctx)
        except ShardyfusionError as exc:
            last_exc = exc
            if not exc.retryable or attempt >= max_attempts - 1:
                if attempt > 0 and metrics_collector is not None:
                    metrics_collector.emit(
                        MetricEvent.SHARD_WRITE_RETRY_EXHAUSTED,
                        {
                            "elapsed_ms": int((time.perf_counter() - started) * 1000),
                            "db_id": db_id,
                            "attempts": attempt + 1,
                        },
                    )
                raise

            failed_attempt_urls.append(db_url)
            log_failure(
                "shard_write_retrying",
                severity=FailureSeverity.TRANSIENT,
                logger=_logger,
                error=exc,
                run_id=run_id,
                db_id=db_id,
                attempt=attempt,
                next_attempt=attempt + 1,
                backoff_s=delay,
            )
            if metrics_collector is not None:
                metrics_collector.emit(
                    MetricEvent.SHARD_WRITE_RETRIED,
                    {
                        "elapsed_ms": int((time.perf_counter() - started) * 1000),
                        "db_id": db_id,
                        "attempt": attempt,
                        "backoff_s": delay,
                    },
                )
            time.sleep(delay)
            delay *= multiplier

    raise last_exc  # type: ignore[misc]


def write_shard_with_retry(
    *,
    db_id: int,
    rows_fn: Callable[[], Iterable[tuple[object, bytes]]],
    run_id: str,
    s3_prefix: str,
    shard_prefix: str,
    db_path_template: str,
    local_root: str,
    key_encoder: KeyEncoder,
    batch_size: int,
    factory: DbAdapterFactory,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    metrics_collector: MetricsCollector | None,
    started: float,
    retry_config: RetryConfig | None,
    writer_info_base: WriterInfo | None = None,
) -> ShardAttemptResult:
    """Write one shard with optional retry on transient failures.

    Each retry attempt writes to a new S3 path (``attempt=00``,
    ``attempt=01``, ...) with a fresh adapter and rate limiters.

    Args:
        rows_fn: Callable returning a fresh iterable of ``(key, value)``
            pairs on each invocation.  Must be re-callable for retries.
        retry_config: Retry parameters.  ``None`` means single attempt
            (no retry), preserving current behavior.

    Returns:
        ``ShardAttemptResult`` from the first successful attempt.

    Raises:
        ShardWriteError: If all attempts fail with retryable errors.
        ShardyfusionError: Immediately on non-retryable errors.
    """

    def _attempt_once(ctx: _RetryAttemptContext) -> ShardAttemptResult:
        ops_limiter: RateLimiter | None = None
        if max_writes_per_second is not None:
            ops_limiter = TokenBucket(
                max_writes_per_second,
                metrics_collector=metrics_collector,
            )
        bytes_limiter: RateLimiter | None = None
        if max_write_bytes_per_second is not None:
            bytes_limiter = TokenBucket(
                max_write_bytes_per_second,
                metrics_collector=metrics_collector,
                limiter_type="bytes",
            )

        params = ShardWriteParams(
            db_id=db_id,
            attempt=ctx.attempt,
            run_id=run_id,
            db_url=ctx.db_url,
            local_dir=ctx.local_dir,
            factory=factory,
            key_encoder=key_encoder,
            batch_size=batch_size,
            ops_limiter=ops_limiter,
            bytes_limiter=bytes_limiter,
            metrics_collector=metrics_collector,
            started=started,
        )

        return write_shard_core(
            params,
            rows_fn(),
            writer_info_base=writer_info_base,
            prior_attempt_urls=ctx.prior_attempt_urls,
        )

    return _run_attempts_with_retry(
        db_id=db_id,
        run_id=run_id,
        s3_prefix=s3_prefix,
        shard_prefix=shard_prefix,
        db_path_template=db_path_template,
        local_root=local_root,
        retry_config=retry_config,
        metrics_collector=metrics_collector,
        started=started,
        attempt_fn=_attempt_once,
    )


def write_shard_with_retry_distributed(
    *,
    db_id: int,
    rows_fn: Callable[[], Iterable[tuple[object, bytes, VectorWriteTuple | None]]],
    run_id: str,
    s3_prefix: str,
    shard_prefix: str,
    db_path_template: str,
    local_root: str,
    key_encoder: KeyEncoder,
    batch_size: int,
    factory: DbAdapterFactory,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    metrics_collector: MetricsCollector | None,
    started: float,
    retry_config: RetryConfig | None,
    writer_info_base: WriterInfo | None = None,
) -> ShardAttemptResult:
    """Retry-enabled distributed shard write with optional vector tuples."""

    def _attempt_once(ctx: _RetryAttemptContext) -> ShardAttemptResult:
        ops_limiter: RateLimiter | None = None
        if max_writes_per_second is not None:
            ops_limiter = TokenBucket(
                max_writes_per_second, metrics_collector=metrics_collector
            )
        bytes_limiter: RateLimiter | None = None
        if max_write_bytes_per_second is not None:
            bytes_limiter = TokenBucket(
                max_write_bytes_per_second,
                metrics_collector=metrics_collector,
                limiter_type="bytes",
            )
        params = ShardWriteParams(
            db_id=db_id,
            attempt=ctx.attempt,
            run_id=run_id,
            db_url=ctx.db_url,
            local_dir=ctx.local_dir,
            factory=factory,
            key_encoder=key_encoder,
            batch_size=batch_size,
            ops_limiter=ops_limiter,
            bytes_limiter=bytes_limiter,
            metrics_collector=metrics_collector,
            started=started,
        )
        return write_shard_core_distributed(
            params,
            rows_fn(),
            writer_info_base=writer_info_base,
            prior_attempt_urls=ctx.prior_attempt_urls,
        )

    return _run_attempts_with_retry(
        db_id=db_id,
        run_id=run_id,
        s3_prefix=s3_prefix,
        shard_prefix=shard_prefix,
        db_path_template=db_path_template,
        local_root=local_root,
        retry_config=retry_config,
        metrics_collector=metrics_collector,
        started=started,
        attempt_fn=_attempt_once,
    )


# ---------------------------------------------------------------------------
# Result DataFrame → ShardAttemptResult (shared by Dask and Ray)
# ---------------------------------------------------------------------------


def results_pdf_to_attempts(
    results_pdf: object,
) -> list[ShardAttemptResult]:
    """Convert a result pandas DataFrame to a list of ShardAttemptResult.

    Handles pandas NaN coercion (int + None columns become float64) and
    normalises ``all_attempt_urls`` from the serialised DataFrame form.
    Accepts ``object`` to avoid a hard pandas import at module level.
    """
    import pandas as pd

    pdf: pd.DataFrame = results_pdf  # type: ignore[assignment]

    if pdf.empty:
        return []

    attempts: list[ShardAttemptResult] = []
    for _, row in pdf.iterrows():
        # Extract pandas row values as Any to satisfy both ty and pyright
        r: Any = row
        min_key = r["min_key"]
        max_key = r["max_key"]
        checkpoint_id = r["checkpoint_id"]
        all_attempt_urls = r["all_attempt_urls"]

        # Handle pandas NaN for None values (int + None columns become float64)
        if isinstance(min_key, float) and pd.isna(min_key):
            min_key = None
        elif isinstance(min_key, float):
            min_key = int(min_key)
        if isinstance(max_key, float) and pd.isna(max_key):
            max_key = None
        elif isinstance(max_key, float):
            max_key = int(max_key)
        if isinstance(checkpoint_id, float) and pd.isna(checkpoint_id):
            checkpoint_id = None

        normalized_attempt_urls: tuple[str, ...]
        if isinstance(all_attempt_urls, (tuple, list)):
            normalized_attempt_urls = tuple(str(url) for url in all_attempt_urls)
        else:
            normalized_attempt_urls = ()

        attempts.append(
            ShardAttemptResult(
                db_id=int(r["db_id"]),
                db_url=str(r["db_url"]),
                attempt=int(r["attempt"]),
                row_count=int(r["row_count"]),
                min_key=min_key,
                max_key=max_key,
                checkpoint_id=None if checkpoint_id is None else str(checkpoint_id),
                writer_info=r["writer_info"],
                db_bytes=int(r["db_bytes"]) if "db_bytes" in r.index else 0,
                all_attempt_urls=normalized_attempt_urls,
            )
        )

    return attempts
