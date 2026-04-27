"""Single-database writer using Spark for distributed sorting."""

import math
import time
from uuid import uuid4

from pyspark import StorageLevel
from pyspark.sql import DataFrame

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

from .util import DataFrameCacheContext, SparkConfOverrideContext


def write_single_db(
    df: DataFrame,
    config: WriteConfig,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_keys: bool = True,
    num_partitions: int | None = None,
    prefetch_partitions: bool = True,
    cache_input: bool = True,
    storage_level: StorageLevel | None = None,
    spark_conf_overrides: dict[str, str] | None = None,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
) -> BuildResult:
    """Write a DataFrame into a single shard database, optionally sorting by key first.

    Uses Spark for the expensive global sort, then streams sorted partitions to the
    driver where they are written into one database instance. This produces an optimal
    write pattern for the LSM tree (sorted keys → clean sorted runs → minimal
    compaction).

    ``num_partitions`` controls how many partitions the sorted data is split
    into before streaming.  When *None* (the default) the function calls
    ``df.count()`` and derives ``ceil(rows / batch_size)`` partitions.
    Pass an explicit value to skip that count — useful when the DataFrame is
    large and not cached.
    """

    if config.num_dbs != 1:
        raise ConfigValidationError(
            f"write_single_db requires num_dbs=1, got {config.num_dbs}"
        )

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex
    spark = df.sparkSession

    with SparkConfOverrideContext(spark, spark_conf_overrides):
        with DataFrameCacheContext(
            df, storage_level=storage_level, enabled=cache_input
        ) as cached_df:
            return _write_single_db_impl(
                df=cached_df,
                config=config,
                run_id=run_id,
                started=started,
                key_col=key_col,
                value_spec=value_spec,
                sort_keys=sort_keys,
                num_partitions=num_partitions,
                prefetch_partitions=prefetch_partitions,
                max_writes_per_second=max_writes_per_second,
                max_write_bytes_per_second=max_write_bytes_per_second,
            )


def _write_single_db_impl(
    *,
    df: DataFrame,
    config: WriteConfig,
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
    """Implementation: sort, stream, write, publish."""
    with managed_run_record(
        config=config,
        run_id=run_id,
        writer_type="spark",
    ) as run_record:
        # Compute partition count (triggers an action when num_partitions is None).
        if num_partitions is None:
            total_rows = df.count()
            num_partitions = max(1, math.ceil(total_rows / config.batch_size))

        # Sort and resize partitions.
        # IMPORTANT: Use coalesce() after sort() — repartition() does a hash shuffle
        # that destroys global ordering.  coalesce() merges adjacent partitions
        # without shuffling, preserving the sorted order across partitions.
        if sort_keys:
            df_prepared = df.sort(key_col).coalesce(num_partitions)
        else:
            df_prepared = df.coalesce(num_partitions)

        # No sharding happens in single-db mode.  The sort DAG is lazy so its
        # actual execution cost is included in write_duration_ms below.
        shard_duration_ms = 0

        # Phase 3: Stream to single writer
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
                df=df_prepared,
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

        resolved_sharding = ShardingSpec(
            strategy=ShardingStrategy.HASH,
            hash_algorithm=config.sharding.hash_algorithm,
        )

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
    df: DataFrame,
    config: WriteConfig,
    run_id: str,
    key_col: str,
    value_spec: ValueSpec,
    prefetch_partitions: bool,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    attempt_ctx: _RetryAttemptContext,
) -> ShardAttemptResult:
    """Stream DataFrame rows to a single shard adapter on the driver."""

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
    batch: list[tuple[bytes, bytes]] = []

    write_started = time.perf_counter()

    try:
        with factory(db_url=db_url, local_dir=local_dir) as adapter:
            for row in df.toLocalIterator(prefetchPartitions=prefetch_partitions):
                key_value = row[key_col]
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
                batch.clear()

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
            db_bytes = adapter.db_bytes() if hasattr(adapter, "db_bytes") else 0
    except ShardyfusionError:
        raise
    except Exception as exc:
        log_failure(
            "single_db_write_failed",
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
