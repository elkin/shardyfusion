"""Single-database writer using Spark for distributed sorting."""

import math
import os
import time
from uuid import uuid4

from pyspark import StorageLevel
from pyspark.sql import DataFrame

from slatedb_spark_sharded._rate_limiter import TokenBucket
from slatedb_spark_sharded._writer_core import (
    ShardAttemptResult,
    assemble_build_result,
    build_manifest_artifact,
    publish_manifest_and_current,
    select_winners,
    update_min_max,
)
from slatedb_spark_sharded.config import WriteConfig
from slatedb_spark_sharded.errors import ConfigValidationError, SlatedbSparkShardedError
from slatedb_spark_sharded.logging import FailureSeverity, log_failure
from slatedb_spark_sharded.manifest import BuildResult
from slatedb_spark_sharded.serde import ValueSpec, make_key_encoder
from slatedb_spark_sharded.sharding_types import ShardingSpec, ShardingStrategy
from slatedb_spark_sharded.slatedb_adapter import (
    DbAdapterFactory,
    SlateDbFactory,
)
from slatedb_spark_sharded.storage import join_s3
from slatedb_spark_sharded.type_defs import JsonObject, KeyLike

from .sharding import _validate_key_col_type
from .writer import DataFrameCacheContext, SparkConfOverrideContext


def write_single_db_spark(
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
            f"write_single_db_spark requires num_dbs=1, got {config.num_dbs}"
        )

    _validate_key_col_type(df=df, key_col=key_col, strategy=ShardingStrategy.HASH)

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
    winners = select_winners(attempts, num_dbs=1)

    resolved_sharding = ShardingSpec(strategy=ShardingStrategy.HASH)

    manifest_started = time.perf_counter()
    artifact = build_manifest_artifact(
        config=config,
        run_id=run_id,
        resolved_sharding=resolved_sharding,
        winners=winners,
        key_col=key_col,
    )
    publish_result = publish_manifest_and_current(
        config=config,
        run_id=run_id,
        artifact=artifact,
    )
    manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

    return assemble_build_result(
        run_id=run_id,
        winners=winners,
        artifact=artifact,
        manifest_ref=publish_result.manifest_ref,
        current_ref=publish_result.current_ref,
        attempts=attempts,
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
    local_dir = os.path.join(
        config.output.local_root,
        f"run_id={run_id}",
        f"db={db_id:05d}",
        f"attempt={attempt:02d}",
    )
    os.makedirs(local_dir, exist_ok=True)

    factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory()

    bucket: TokenBucket | None = None
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
                        bucket.acquire(1)
                    adapter.write_batch(batch)
                    batch.clear()

            if batch:
                if bucket is not None:
                    bucket.acquire(1)
                adapter.write_batch(batch)
                batch.clear()

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
    except SlatedbSparkShardedError:
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
        raise SlatedbSparkShardedError(
            f"Single-db write failed for db_id={db_id}, attempt={attempt}: {exc}"
        ) from exc

    writer_info: JsonObject = {
        "stage_id": None,
        "task_attempt_id": None,
        "attempt": attempt,
        "duration_ms": int((time.perf_counter() - write_started) * 1000),
    }

    return ShardAttemptResult(
        db_id=db_id,
        db_url=db_url,
        attempt=attempt,
        row_count=row_count,
        min_key=min_key,
        max_key=max_key,
        checkpoint_id=checkpoint_id,
        writer_info=writer_info,
    )
