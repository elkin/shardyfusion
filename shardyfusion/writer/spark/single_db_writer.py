"""Single-database writer using Spark for distributed sorting."""

import math
import time
from pathlib import Path
from uuid import uuid4

from pyspark import StorageLevel
from pyspark.sql import DataFrame

from shardyfusion._rate_limiter import RateLimiter, TokenBucket
from shardyfusion._writer_core import (
    ShardAttemptResult,
    assemble_build_result,
    publish_to_store,
    select_winners,
    update_min_max,
)
from shardyfusion.config import WriteConfig
from shardyfusion.errors import ConfigValidationError, ShardyfusionError
from shardyfusion.logging import FailureSeverity, log_failure
from shardyfusion.manifest import BuildResult, WriterInfo
from shardyfusion.serde import ValueSpec, make_key_encoder
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
from shardyfusion.slatedb_adapter import (
    DbAdapterFactory,
    SlateDbFactory,
)
from shardyfusion.storage import join_s3
from shardyfusion.type_defs import KeyLike

from .sharding import validate_key_col_type
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
) -> BuildResult:
    """Write a DataFrame into a single SlateDB database, optionally sorting by key first.

    Uses Spark for the expensive global sort, then streams sorted partitions to the
    driver where they are written into one SlateDB instance. This produces an optimal
    write pattern for SlateDB's LSM tree (sorted keys → clean sorted runs → minimal
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

    validate_key_col_type(df=df, key_col=key_col, strategy=ShardingStrategy.HASH)

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
) -> BuildResult:
    """Implementation: sort, stream, write, publish."""

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
    attempt_result = _stream_to_single_db(
        df=df_prepared,
        config=config,
        run_id=run_id,
        key_col=key_col,
        value_spec=value_spec,
        prefetch_partitions=prefetch_partitions,
        max_writes_per_second=max_writes_per_second,
    )
    write_duration_ms = int((time.perf_counter() - write_started) * 1000)

    # Phase 4: Publish
    attempts = [attempt_result]
    winners, num_attempts, _attempt_urls = select_winners(attempts, num_dbs=1)

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
    manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

    return assemble_build_result(
        run_id=run_id,
        winners=winners,
        manifest_ref=manifest_ref,
        num_attempts=num_attempts,
        shard_duration_ms=shard_duration_ms,
        write_duration_ms=write_duration_ms,
        manifest_duration_ms=manifest_duration_ms,
        started=started,
    )


def _stream_to_single_db(
    *,
    df: DataFrame,
    config: WriteConfig,
    run_id: str,
    key_col: str,
    value_spec: ValueSpec,
    prefetch_partitions: bool,
    max_writes_per_second: float | None,
) -> ShardAttemptResult:
    """Stream DataFrame rows to a single SlateDB adapter on the driver."""

    db_id = 0
    attempt = 0
    key_encoder = make_key_encoder(config.key_encoding)

    db_rel_path = config.output.db_path_template.format(db_id=db_id)
    db_url = join_s3(
        config.s3_prefix,
        config.output.tmp_prefix,
        f"run_id={run_id}",
        db_rel_path,
        f"attempt={attempt:02d}",
    )
    local_dir = (
        Path(config.output.local_root)
        / f"run_id={run_id}"
        / f"db={db_id:05d}"
        / f"attempt={attempt:02d}"
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory(
        credential_provider=config.credential_provider
    )

    bucket: RateLimiter | None = None
    if max_writes_per_second is not None:
        bucket = TokenBucket(max_writes_per_second)

    row_count = 0
    min_key: KeyLike | None = None
    max_key: KeyLike | None = None
    checkpoint_id: str | None = None
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
                    adapter.write_batch(batch)
                    batch.clear()

            if batch:
                if bucket is not None:
                    bucket.acquire(len(batch))
                adapter.write_batch(batch)
                batch.clear()

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
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
        raise ShardyfusionError(
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
    )
