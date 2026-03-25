"""Single-database writer using Ray Data for distributed sorting."""

import os
import time
import types
from typing import Any
from uuid import uuid4

import ray.data

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
from shardyfusion.config import WriteConfig
from shardyfusion.errors import (
    ConfigValidationError,
    ShardWriteError,
    ShardyfusionError,
)
from shardyfusion.logging import FailureSeverity, log_failure
from shardyfusion.manifest import BuildResult, WriterInfo
from shardyfusion.run_registry import managed_run_record
from shardyfusion.serde import ValueSpec, make_key_encoder
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
from shardyfusion.slatedb_adapter import (
    DbAdapterFactory,
    SlateDbFactory,
)
from shardyfusion.type_defs import KeyInput


class RayCacheContext:
    """Materialize a Ray Dataset for the lifetime of the context.

    Calling ``materialize()`` pins the dataset in the Ray object store,
    preventing re-computation on subsequent reads.
    """

    def __init__(self, ds: ray.data.Dataset, *, enabled: bool = True) -> None:
        self._ds = ds
        self._enabled = enabled
        self._materialized: ray.data.Dataset | None = None

    def __enter__(self) -> ray.data.Dataset:
        if not self._enabled:
            return self._ds
        materialized: ray.data.Dataset = self._ds.materialize()
        self._materialized = materialized
        return materialized

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        # Release reference to materialized dataset; Ray GC handles cleanup
        self._materialized = None


def write_single_db(
    ds: ray.data.Dataset,
    config: WriteConfig,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_keys: bool = True,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
) -> BuildResult:
    """Write a Ray Dataset into a single shard database, optionally sorting by key first.

    Uses Ray Data for the expensive global sort, then streams sorted batches to the
    local process where they are written into one database instance.
    """

    if config.num_dbs != 1:
        raise ConfigValidationError(
            f"write_single_db requires num_dbs=1, got {config.num_dbs}"
        )

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    return _write_single_db_impl(
        ds=ds,
        config=config,
        run_id=run_id,
        started=started,
        key_col=key_col,
        value_spec=value_spec,
        sort_keys=sort_keys,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )


def _write_single_db_impl(
    *,
    ds: ray.data.Dataset,
    config: WriteConfig,
    run_id: str,
    started: float,
    key_col: str,
    value_spec: ValueSpec,
    sort_keys: bool,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
) -> BuildResult:
    """Implementation: sort, stream batches, write, publish."""
    with managed_run_record(
        config=config,
        run_id=run_id,
        writer_type="ray",
    ) as run_record:
        # Sort if requested (Ray's sort is a global distributed sort).
        if sort_keys:
            ds_prepared = ds.sort(key_col)
        else:
            ds_prepared = ds

        # No sharding happens in single-db mode.
        shard_duration_ms = 0

        # Stream batches to single writer
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
                ds=ds_prepared,
                config=config,
                run_id=run_id,
                key_col=key_col,
                value_spec=value_spec,
                max_writes_per_second=max_writes_per_second,
                max_write_bytes_per_second=max_write_bytes_per_second,
                attempt_ctx=ctx,
            ),
        )
        write_duration_ms = int((time.perf_counter() - write_started) * 1000)

        # Publish
        attempts = [attempt_result]
        winners, num_attempts, all_attempt_urls = select_winners(attempts, num_dbs=1)

        resolved_sharding = ShardingSpec(strategy=ShardingStrategy.HASH)

        manifest_started = time.perf_counter()
        manifest_ref = publish_to_store(
            config=config,
            run_id=run_id,
            resolved_sharding=resolved_sharding,
            winners=winners,
            key_col=key_col,
            started=started,
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
    ds: ray.data.Dataset,
    config: WriteConfig,
    run_id: str,
    key_col: str,
    value_spec: ValueSpec,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    attempt_ctx: _RetryAttemptContext,
) -> ShardAttemptResult:
    """Stream Ray Dataset batches to a single shard adapter."""

    db_id = 0
    attempt = attempt_ctx.attempt
    key_encoder = make_key_encoder(config.key_encoding)
    db_url = attempt_ctx.db_url
    local_dir = str(attempt_ctx.local_dir)
    os.makedirs(local_dir, exist_ok=True)

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

    write_started = time.perf_counter()

    try:
        with factory(db_url=db_url, local_dir=attempt_ctx.local_dir) as adapter:
            batch: list[tuple[bytes, bytes]] = []

            # Stream batches using iter_batches — Ray handles prefetching internally.
            for batch_pdf in ds.iter_batches(batch_format="pandas"):  # type: ignore[union-attr]
                for _, row in batch_pdf.iterrows():  # type: ignore[union-attr]
                    key_value: Any = row[key_col]
                    # Convert numpy scalars to Python types
                    if hasattr(key_value, "item"):
                        key_value = key_value.item()

                    key_bytes = key_encoder(key_value)
                    value_bytes = value_spec.encode(row)

                    batch.append((key_bytes, value_bytes))
                    row_count += 1
                    min_key, max_key = update_min_max(min_key, max_key, key_value)

                    if len(batch) >= config.batch_size:
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

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
    except ShardyfusionError:
        raise
    except Exception as exc:
        log_failure(
            "single_db_ray_write_failed",
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
        all_attempt_urls=(*attempt_ctx.prior_attempt_urls, db_url),
    )
