"""Single-database writer using Dask for distributed sorting."""

import math
import time
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

import dask.dataframe as dd
import pandas as pd

from shardyfusion._rate_limiter import RateLimiter, TokenBucket
from shardyfusion._shard_writer import _RetryAttemptContext, _run_attempts_with_retry
from shardyfusion._writer_core import (
    ShardAttemptResult,
    assemble_build_result,
    cleanup_losers,
    publish_to_store,
    select_winners,
    update_min_max,
)
from shardyfusion.config import (
    ColumnWriteInput,
    SingleDbWriteConfig,
    SingleDbWriteOptions,
)
from shardyfusion.errors import (
    ConfigValidationError,
    ShardWriteError,
    ShardyfusionError,
)
from shardyfusion.logging import FailureSeverity, log_failure
from shardyfusion.manifest import BuildResult, WriterInfo
from shardyfusion.run_registry import RunRecordLifecycle
from shardyfusion.serde import KeyEncoder, ValueSpec, make_key_encoder
from shardyfusion.sharding_types import HashShardingSpec, ShardHashAlgorithm
from shardyfusion.slatedb_adapter import (
    DbAdapter,
    DbAdapterFactory,
    SlateDbFactory,
)
from shardyfusion.type_defs import KeyInput


class DaskCacheContext:
    """Persist a Dask DataFrame for the lifetime of the context."""

    def __init__(self, ddf: dd.DataFrame, *, enabled: bool = True) -> None:
        self._ddf = ddf
        self._enabled = enabled
        self._persisted: dd.DataFrame | None = None

    def __enter__(self) -> dd.DataFrame:
        if not self._enabled:
            return self._ddf
        persisted: dd.DataFrame = self._ddf.persist()  # type: ignore[assignment]
        self._persisted = persisted
        return persisted

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        # Release reference to persisted futures; Dask GC handles cleanup
        self._persisted = None


def write_single_db(
    ddf: dd.DataFrame,
    config: SingleDbWriteConfig,
    input: ColumnWriteInput,
    options: SingleDbWriteOptions | None = None,
) -> BuildResult:
    """Write a Dask DataFrame into a single shard database, optionally sorting by key first.

    Uses Dask for the expensive global sort, then streams sorted partitions to the
    local process where they are written into one database instance.

    ``num_partitions`` controls how many partitions the sorted data is split
    into before streaming.  When *None* (the default) the function calls
    ``len(ddf)`` and derives ``ceil(rows / batch_size)`` partitions.  Pass an
    explicit value to skip that count — useful when the DataFrame is large
    and not cached.
    """

    options = options or SingleDbWriteOptions()
    config.validate()
    input.validate()
    options.validate()

    if input.key_col not in ddf.columns:
        raise ConfigValidationError(
            f"Key column `{input.key_col}` not found in DataFrame columns: "
            f"{list(ddf.columns)}"
        )

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    with DaskCacheContext(ddf, enabled=options.cache_input) as cached_ddf:
        return _write_single_db_impl(
            ddf=cached_ddf,
            config=config,
            run_id=run_id,
            started=started,
            key_col=input.key_col,
            value_spec=input.value_spec,
            sort_keys=options.sort_keys,
            num_partitions=options.num_partitions,
            prefetch_partitions=options.prefetch_partitions,
            max_writes_per_second=config.rate_limits.max_writes_per_second,
            max_write_bytes_per_second=config.rate_limits.max_write_bytes_per_second,
        )


def _write_single_db_impl(
    *,
    ddf: dd.DataFrame,
    config: SingleDbWriteConfig,
    run_id: str,
    started: float,
    key_col: str,
    value_spec: ValueSpec,
    sort_keys: bool,
    num_partitions: int | None,
    prefetch_partitions: bool,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
) -> BuildResult:
    """Implementation: sort, stream partitions, write, publish."""
    with RunRecordLifecycle.start(
        config=config,
        run_id=run_id,
        writer_type="dask",
    ) as run_record:
        # Compute partition count (triggers a compute when num_partitions is None).
        if num_partitions is None:
            total_rows = len(ddf)
            num_partitions = max(1, math.ceil(total_rows / config.batch_size))

        # Sort and repartition.
        if sort_keys:
            ddf_prepared = ddf.sort_values(key_col).repartition(
                npartitions=num_partitions
            )
        else:
            ddf_prepared = ddf.repartition(npartitions=num_partitions)

        # No sharding happens in single-db mode.  The sort task graph is lazy so
        # its actual execution cost is included in write_duration_ms below.
        shard_duration_ms = 0

        # Phase 3: Stream partitions to single writer
        write_started = time.perf_counter()
        attempt_result = _run_attempts_with_retry(
            db_id=0,
            run_id=run_id,
            s3_prefix=config.s3_prefix,
            shard_prefix=config.output.shard_prefix,
            db_path_template=config.output.db_path_template,
            local_root=config.output.local_root,
            retry_config=config.shard_retry,
            metrics_collector=config.metrics_collector,
            started=started,
            attempt_fn=lambda ctx: _stream_to_single_db(
                ddf=ddf_prepared,
                config=config,
                run_id=run_id,
                key_col=key_col,
                value_spec=value_spec,
                prefetch_partitions=prefetch_partitions,
                max_writes_per_second=max_writes_per_second,
                max_write_bytes_per_second=max_write_bytes_per_second,
                attempt_ctx=ctx,
            ),
        )
        write_duration_ms = int((time.perf_counter() - write_started) * 1000)

        # Phase 4: Publish
        attempts = [attempt_result]
        winners, num_attempts, all_attempt_urls = select_winners(attempts, num_dbs=1)

        resolved_sharding = HashShardingSpec(
            hash_algorithm=ShardHashAlgorithm.XXH3_64,
        )

        manifest_started = time.perf_counter()
        manifest_ref = publish_to_store(
            config=config,
            run_id=run_id,
            resolved_sharding=resolved_sharding,
            winners=winners,
            key_col=key_col,
            started=started,
            num_dbs=1,
        )
        run_record.set_manifest_ref(manifest_ref)
        manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

        cleanup_losers(
            all_attempt_urls, winners, metrics_collector=config.metrics_collector
        )

        result = assemble_build_result(
            run_id=run_id,
            winners=winners,
            manifest_ref=manifest_ref,
            run_record_ref=run_record.run_record_ref,
            num_attempts=num_attempts,
            shard_duration_ms=shard_duration_ms,
            write_duration_ms=write_duration_ms,
            manifest_duration_ms=manifest_duration_ms,
            started=started,
        )
        run_record.mark_succeeded()
        return result


def _stream_to_single_db(
    *,
    ddf: dd.DataFrame,
    config: SingleDbWriteConfig,
    run_id: str,
    key_col: str,
    value_spec: ValueSpec,
    prefetch_partitions: bool,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    attempt_ctx: _RetryAttemptContext,
) -> ShardAttemptResult:
    """Stream Dask partitions to a single shard adapter."""

    db_id = 0
    attempt = attempt_ctx.attempt
    key_encoder = make_key_encoder(config.key_encoding)
    db_url = attempt_ctx.db_url
    local_dir = attempt_ctx.local_dir
    local_dir.mkdir(parents=True, exist_ok=True)

    factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory(
        credential_provider=config.credential_provider
    )

    bucket: RateLimiter | None = None
    if max_writes_per_second is not None:
        bucket = TokenBucket(max_writes_per_second)
    bytes_bucket: RateLimiter | None = None
    if max_write_bytes_per_second is not None:
        bytes_bucket = TokenBucket(max_write_bytes_per_second)

    row_count = 0
    min_key: KeyInput | None = None
    max_key: KeyInput | None = None
    checkpoint_id: str | None = None
    db_bytes = 0

    write_started = time.perf_counter()

    try:
        with factory(db_url=db_url, local_dir=local_dir) as adapter:
            partitions = ddf.to_delayed()

            if prefetch_partitions and len(partitions) > 1:
                row_count, min_key, max_key = _write_with_prefetch(
                    partitions=partitions,
                    adapter=adapter,
                    key_col=key_col,
                    key_encoder=key_encoder,
                    value_spec=value_spec,
                    batch_size=config.batch_size,
                    bucket=bucket,
                    bytes_bucket=bytes_bucket,
                )
            else:
                row_count, min_key, max_key = _write_sequential(
                    partitions=partitions,
                    adapter=adapter,
                    key_col=key_col,
                    key_encoder=key_encoder,
                    value_spec=value_spec,
                    batch_size=config.batch_size,
                    bucket=bucket,
                    bytes_bucket=bytes_bucket,
                )

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
            db_bytes = adapter.db_bytes()
    except ShardyfusionError:
        raise
    except Exception as exc:
        log_failure(
            "single_db_dask_write_failed",
            severity=FailureSeverity.ERROR,
            error=exc,
            db_id=db_id,
            attempt=attempt,
            db_url=db_url,
            rows_written=row_count,
            include_traceback=True,
        )
        raise ShardWriteError(
            f"Single-db write failed for db_id={db_id}, attempt={attempt}: {exc}"
        ) from exc

    return ShardAttemptResult(
        db_id=db_id,
        db_url=db_url,
        attempt=attempt,
        row_count=row_count,
        min_key=min_key,
        max_key=max_key,
        checkpoint_id=checkpoint_id,
        writer_info=WriterInfo(
            attempt=attempt,
            duration_ms=int((time.perf_counter() - write_started) * 1000),
        ),
        db_bytes=db_bytes,
        all_attempt_urls=(*attempt_ctx.prior_attempt_urls, db_url),
    )


def _write_pdf_rows(
    pdf: pd.DataFrame,
    adapter: DbAdapter,
    key_col: str,
    key_encoder: KeyEncoder,
    value_spec: ValueSpec,
    batch_size: int,
    bucket: RateLimiter | None,
    row_count: int,
    min_key: KeyInput | None,
    max_key: KeyInput | None,
    bytes_bucket: RateLimiter | None = None,
) -> tuple[int, KeyInput | None, KeyInput | None]:
    """Write rows from a pandas DataFrame to the adapter."""

    batch: list[tuple[bytes, bytes]] = []

    for _, row in pdf.iterrows():
        key_value: Any = row[key_col]
        # Convert numpy scalars to Python types
        if hasattr(key_value, "item"):
            key_value = key_value.item()

        key_bytes = key_encoder(key_value)
        value_bytes = value_spec.encode(row)

        batch.append((key_bytes, value_bytes))
        row_count += 1
        min_key, max_key = update_min_max(min_key, max_key, key_value)

        if len(batch) >= batch_size:
            if bucket is not None:
                bucket.acquire(len(batch))
            if bytes_bucket is not None:
                batch_bytes = sum(len(k) + len(v) for k, v in batch)
                bytes_bucket.acquire(batch_bytes)
            adapter.write_batch(batch)
            batch.clear()

    if batch:
        if bucket is not None:
            bucket.acquire(len(batch))
        if bytes_bucket is not None:
            batch_bytes = sum(len(k) + len(v) for k, v in batch)
            bytes_bucket.acquire(batch_bytes)
        adapter.write_batch(batch)

    return row_count, min_key, max_key


def _write_with_prefetch(
    *,
    partitions: list[Any],
    adapter: DbAdapter,
    key_col: str,
    key_encoder: KeyEncoder,
    value_spec: ValueSpec,
    batch_size: int,
    bucket: RateLimiter | None,
    bytes_bucket: RateLimiter | None = None,
) -> tuple[int, KeyInput | None, KeyInput | None]:
    """Write partitions with background prefetching of the next partition."""

    row_count = 0
    min_key: KeyInput | None = None
    max_key: KeyInput | None = None

    with ThreadPoolExecutor(max_workers=1) as prefetcher:
        next_future = prefetcher.submit(partitions[0].compute)
        for i in range(len(partitions)):
            current_pdf: pd.DataFrame = next_future.result()
            if i + 1 < len(partitions):
                next_future = prefetcher.submit(partitions[i + 1].compute)

            if not current_pdf.empty:
                row_count, min_key, max_key = _write_pdf_rows(
                    current_pdf,
                    adapter,
                    key_col,
                    key_encoder,
                    value_spec,
                    batch_size,
                    bucket,
                    row_count,
                    min_key,
                    max_key,
                    bytes_bucket=bytes_bucket,
                )

    return row_count, min_key, max_key


def _write_sequential(
    *,
    partitions: list[Any],
    adapter: DbAdapter,
    key_col: str,
    key_encoder: KeyEncoder,
    value_spec: ValueSpec,
    batch_size: int,
    bucket: RateLimiter | None,
    bytes_bucket: RateLimiter | None = None,
) -> tuple[int, KeyInput | None, KeyInput | None]:
    """Write partitions sequentially without prefetching."""

    row_count = 0
    min_key: KeyInput | None = None
    max_key: KeyInput | None = None

    for partition in partitions:
        pdf: pd.DataFrame = partition.compute()
        if not pdf.empty:
            row_count, min_key, max_key = _write_pdf_rows(
                pdf,
                adapter,
                key_col,
                key_encoder,
                value_spec,
                batch_size,
                bucket,
                row_count,
                min_key,
                max_key,
                bytes_bucket=bytes_bucket,
            )

    return row_count, min_key, max_key
