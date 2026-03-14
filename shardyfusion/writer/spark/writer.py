"""Public sharded snapshot writer entrypoint."""

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from pyspark import RDD, StorageLevel, TaskContext
from pyspark.sql import DataFrame, Row

from shardyfusion._rate_limiter import RateLimiter, TokenBucket
from shardyfusion._writer_core import (
    PartitionWriteOutcome,
    ShardAttemptResult,
    assemble_build_result,
    publish_to_store,
    select_winners,
    update_min_max,
)
from shardyfusion.config import WriteConfig
from shardyfusion.errors import ShardAssignmentError, ShardyfusionError
from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_event,
    log_failure,
)
from shardyfusion.manifest import BuildResult
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.serde import KeyEncoder, ValueSpec, make_key_encoder
from shardyfusion.sharding_types import DB_ID_COL, KeyEncoding
from shardyfusion.slatedb_adapter import (
    DbAdapterFactory,
    default_adapter_factory,
)
from shardyfusion.storage import join_s3
from shardyfusion.type_defs import JsonObject, KeyLike
from shardyfusion.writer.spark.util import (
    DataFrameCacheContext,
    SparkConfOverrideContext,
)

from .sharding import ShardingSpec, add_db_id_column, prepare_partitioned_rdd

_logger = get_logger(__name__)


@dataclass(slots=True)
class PartitionWriteConfig:
    run_id: str
    s3_prefix: str
    tmp_prefix: str
    db_path_template: str
    local_root: str
    key_col: str
    key_encoding: KeyEncoding
    key_encoder: KeyEncoder
    value_spec: ValueSpec
    batch_size: int
    adapter_factory: DbAdapterFactory | None
    max_writes_per_second: float | None
    max_write_bytes_per_second: float | None = None
    metrics_collector: MetricsCollector | None = None  # must be picklable
    started: float = 0.0


@dataclass(slots=True)
class _PreparedPartitionRows:
    partitioned_rdd: RDD[tuple[int, Row]]
    resolved_sharding: ShardingSpec
    shard_duration_ms: int


def write_sharded(
    df: DataFrame,
    config: WriteConfig,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool = False,
    spark_conf_overrides: dict[str, str] | None = None,
    cache_input: bool = False,
    storage_level: StorageLevel | None = None,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    verify_routing: bool = True,
) -> BuildResult:
    """Write a DataFrame into N independent sharded databases and publish manifest metadata.

    Args:
        df: PySpark DataFrame containing at least the key column and value column(s).
        config: Write configuration (num_dbs, s3_prefix, sharding strategy, etc.).
        key_col: Name of the integer key column used for shard routing.
        value_spec: Specifies how DataFrame rows are serialized to bytes
            (binary_col, json_cols, or a callable encoder).
        sort_within_partitions: If True, sort rows by key within each partition
            before writing. Useful for range-scan workloads.
        spark_conf_overrides: Optional Spark config overrides applied for the
            duration of the write (restored after completion).
        cache_input: If True, cache the input DataFrame before processing.
        storage_level: Spark storage level for caching (default: MEMORY_AND_DISK).
        max_writes_per_second: Optional rate limit (token-bucket) for write ops/sec.
        max_write_bytes_per_second: Optional rate limit (token-bucket) for write bytes/sec.
        verify_routing: If True (default), spot-check that Spark-assigned shard IDs
            match Python routing on a sample of written rows.

    Returns:
        BuildResult with manifest reference, shard metadata, and build statistics.

    Raises:
        ConfigValidationError: If configuration is invalid.
        ShardAssignmentError: If rows cannot be assigned to valid shard IDs.
        ShardCoverageError: If partition results don't cover all expected shards.
        PublishManifestError: If manifest upload to S3 fails.
        PublishCurrentError: If CURRENT pointer upload fails (manifest already published).
    """

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex
    spark = df.sparkSession
    with SparkConfOverrideContext(spark, spark_conf_overrides):
        with DataFrameCacheContext(
            df, storage_level=storage_level, enabled=cache_input
        ) as cached_df:
            return _write_sharded_impl(
                df=cached_df,
                config=config,
                run_id=run_id,
                started=started,
                key_col=key_col,
                value_spec=value_spec,
                sort_within_partitions=sort_within_partitions,
                max_writes_per_second=max_writes_per_second,
                max_write_bytes_per_second=max_write_bytes_per_second,
                verify_routing=verify_routing,
            )


def _write_sharded_impl(
    *,
    df: DataFrame,
    config: WriteConfig,
    run_id: str,
    started: float,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    verify_routing: bool,
) -> BuildResult:
    """Implementation assuming Spark conf already prepared."""
    from shardyfusion.logging import LogContext

    mc = config.metrics_collector

    with LogContext(run_id=run_id):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=config.num_dbs,
            s3_prefix=config.s3_prefix,
            strategy=config.sharding.strategy,
            key_encoding=config.key_encoding,
            writer_type="spark",
        )
        if mc is not None:
            mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})

        prepared_rows = _prepare_partitioned_rows(
            df=df,
            config=config,
            key_col=key_col,
            sort_within_partitions=sort_within_partitions,
            verify_routing=verify_routing,
        )
        log_event(
            "sharding_completed",
            logger=_logger,
            duration_ms=prepared_rows.shard_duration_ms,
        )
        if mc is not None:
            mc.emit(
                MetricEvent.SHARDING_COMPLETED,
                {
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "duration_ms": prepared_rows.shard_duration_ms,
                },
            )

        runtime = _build_partition_write_runtime(
            config=config,
            run_id=run_id,
            key_col=key_col,
            value_spec=value_spec,
            max_writes_per_second=max_writes_per_second,
            max_write_bytes_per_second=max_write_bytes_per_second,
            started=started,
        )
        write_outcome = _run_partition_writes(
            partitioned_rdd=prepared_rows.partitioned_rdd,
            runtime=runtime,
            num_dbs=config.num_dbs,
        )

        rows_written = sum(w.row_count for w in write_outcome.winners)
        log_event(
            "shard_writes_completed",
            logger=_logger,
            num_winners=len(write_outcome.winners),
            rows_written=rows_written,
            duration_ms=write_outcome.write_duration_ms,
        )
        if mc is not None:
            mc.emit(
                MetricEvent.SHARD_WRITES_COMPLETED,
                {
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "duration_ms": write_outcome.write_duration_ms,
                    "rows_written": rows_written,
                },
            )

        manifest_started = time.perf_counter()
        manifest_ref = publish_to_store(
            config=config,
            run_id=run_id,
            resolved_sharding=prepared_rows.resolved_sharding,
            winners=write_outcome.winners,
            key_col=key_col,
            started=started,
        )
        manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

        result = assemble_build_result(
            run_id=run_id,
            winners=write_outcome.winners,
            manifest_ref=manifest_ref,
            attempts=write_outcome.attempts,
            shard_duration_ms=prepared_rows.shard_duration_ms,
            write_duration_ms=write_outcome.write_duration_ms,
            manifest_duration_ms=manifest_duration_ms,
            started=started,
        )

        log_event(
            "write_completed",
            logger=_logger,
            total_ms=result.stats.durations.total_ms,
            rows_written=result.stats.rows_written,
        )
        if mc is not None:
            mc.emit(
                MetricEvent.WRITE_COMPLETED,
                {
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "rows_written": result.stats.rows_written,
                },
            )

        return result


def verify_routing_agreement(
    df_with_db_id: DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    resolved_sharding: ShardingSpec,
    key_encoding: KeyEncoding,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify Spark-computed db_id matches Python routing.

    Raises ShardAssignmentError if any sampled row disagrees between the
    Spark SQL expression and the equivalent Python routing function.
    """
    from bisect import bisect_right

    from shardyfusion.routing import xxhash64_db_id
    from shardyfusion.sharding_types import ShardingStrategy

    sampled = df_with_db_id.select(key_col, DB_ID_COL).limit(sample_size).collect()
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        key = row[key_col]
        spark_db_id = int(row[DB_ID_COL])

        if resolved_sharding.strategy == ShardingStrategy.HASH:
            python_db_id = xxhash64_db_id(key, num_dbs, key_encoding)
        elif (
            resolved_sharding.strategy == ShardingStrategy.RANGE
            and resolved_sharding.boundaries is not None
        ):
            python_db_id = bisect_right(resolved_sharding.boundaries, key)
        else:
            return  # cannot verify custom_expr or range without boundaries

        if python_db_id != spark_db_id:
            mismatches.append((key, spark_db_id, python_db_id))

    if mismatches:
        details = "; ".join(
            f"key={k}, spark={s}, python={p}" for k, s, p in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Spark/Python routing mismatch in {len(mismatches)}/{len(sampled)} "
            f"sampled rows. First mismatches: {details}"
        )


def _prepare_partitioned_rows(
    *,
    df: DataFrame,
    config: WriteConfig,
    key_col: str,
    sort_within_partitions: bool,
    verify_routing: bool = True,
) -> _PreparedPartitionRows:
    """Assign shard ids and build the one-writer-per-db partitioned RDD."""

    shard_started = time.perf_counter()
    df_with_db_id, resolved_sharding = add_db_id_column(
        df,
        key_col=key_col,
        num_dbs=config.num_dbs,
        sharding=config.sharding,
    )
    if verify_routing:
        verify_routing_agreement(
            df_with_db_id,
            key_col=key_col,
            num_dbs=config.num_dbs,
            resolved_sharding=resolved_sharding,
            key_encoding=config.key_encoding,
        )
    partitioned_rdd = prepare_partitioned_rdd(
        df_with_db_id,
        num_dbs=config.num_dbs,
        key_col=key_col,
        sort_within_partitions=sort_within_partitions,
    )
    shard_duration_ms = int((time.perf_counter() - shard_started) * 1000)
    return _PreparedPartitionRows(
        partitioned_rdd=partitioned_rdd,
        resolved_sharding=resolved_sharding,
        shard_duration_ms=shard_duration_ms,
    )


def _build_partition_write_runtime(
    *,
    config: WriteConfig,
    run_id: str,
    key_col: str,
    value_spec: ValueSpec,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    started: float = 0.0,
) -> PartitionWriteConfig:
    """Construct immutable worker-side runtime config for partition shard writers."""

    return PartitionWriteConfig(
        run_id=run_id,
        s3_prefix=config.s3_prefix,
        tmp_prefix=config.output.tmp_prefix,
        db_path_template=config.output.db_path_template,
        local_root=config.output.local_root,
        key_col=key_col,
        key_encoding=config.key_encoding,
        key_encoder=make_key_encoder(config.key_encoding),
        value_spec=value_spec,
        batch_size=config.batch_size,
        adapter_factory=config.adapter_factory,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
        metrics_collector=config.metrics_collector,
        started=started,
    )


def _run_partition_writes(
    *,
    partitioned_rdd: RDD[tuple[int, Row]],
    runtime: PartitionWriteConfig,
    num_dbs: int,
) -> PartitionWriteOutcome:
    """Execute partition writers and deterministically select winning shard attempts."""

    write_started = time.perf_counter()
    attempts: list[ShardAttemptResult] = partitioned_rdd.mapPartitionsWithIndex(
        lambda db_id, items: write_one_shard_partition(db_id, items, runtime)
    ).collect()
    write_duration_ms = int((time.perf_counter() - write_started) * 1000)
    winners = select_winners(attempts, num_dbs=num_dbs)
    return PartitionWriteOutcome(
        attempts=attempts,
        winners=winners,
        write_duration_ms=write_duration_ms,
    )


def write_one_shard_partition(
    db_id: int,
    rows_iter: Iterable[tuple[int, Row]],
    runtime: PartitionWriteConfig,
) -> Iterator[ShardAttemptResult]:
    """Write exactly one shard from one partition and yield one ShardAttemptResult."""

    ctx = TaskContext.get()
    attempt = int(ctx.attemptNumber()) if ctx else 0
    stage_id = int(ctx.stageId()) if ctx else None
    task_attempt_id = int(ctx.taskAttemptId()) if ctx else None

    db_rel_path = runtime.db_path_template.format(db_id=db_id)
    db_url = join_s3(
        runtime.s3_prefix,
        runtime.tmp_prefix,
        f"run_id={runtime.run_id}",
        db_rel_path,
        f"attempt={attempt:02d}",
    )
    local_dir = (
        Path(runtime.local_root)
        / f"run_id={runtime.run_id}"
        / f"db={db_id:05d}"
        / f"attempt={attempt:02d}"
    )
    local_dir.mkdir(parents=True, exist_ok=True)

    factory = runtime.adapter_factory or default_adapter_factory

    bucket: RateLimiter | None = None
    if runtime.max_writes_per_second is not None:
        bucket = TokenBucket(
            runtime.max_writes_per_second, metrics_collector=runtime.metrics_collector
        )
    byte_bucket: RateLimiter | None = None
    if runtime.max_write_bytes_per_second is not None:
        byte_bucket = TokenBucket(
            runtime.max_write_bytes_per_second,
            metrics_collector=runtime.metrics_collector,
            limiter_type="bytes",
        )

    mc = runtime.metrics_collector

    log_event(
        "shard_write_started",
        logger=_logger,
        run_id=runtime.run_id,
        db_id=db_id,
        attempt=attempt,
        db_url=db_url,
    )
    if mc is not None:
        mc.emit(
            MetricEvent.SHARD_WRITE_STARTED,
            {
                "elapsed_ms": int((time.perf_counter() - runtime.started) * 1000),
            },
        )

    partition_started = time.perf_counter()
    row_count = 0
    min_key: KeyLike | None = None
    max_key: KeyLike | None = None
    checkpoint_id: str | None = None
    batch: list[tuple[bytes, bytes]] = []

    try:
        with factory(
            db_url=db_url,
            local_dir=local_dir,
        ) as adapter:
            for _, row in rows_iter:
                key_value = row[runtime.key_col]
                key_bytes = runtime.key_encoder(key_value)
                value_bytes = runtime.value_spec.encode(row)

                batch.append((key_bytes, value_bytes))
                row_count += 1
                min_key, max_key = update_min_max(min_key, max_key, key_value)

                if len(batch) >= runtime.batch_size:
                    if bucket is not None:
                        bucket.acquire(len(batch))
                    if byte_bucket is not None:
                        byte_bucket.acquire(sum(len(k) + len(v) for k, v in batch))
                    adapter.write_batch(batch)
                    if mc is not None:
                        mc.emit(
                            MetricEvent.BATCH_WRITTEN,
                            {
                                "elapsed_ms": int(
                                    (time.perf_counter() - runtime.started) * 1000
                                ),
                                "batch_size": len(batch),
                            },
                        )
                    batch.clear()

            if batch:
                if bucket is not None:
                    bucket.acquire(len(batch))
                if byte_bucket is not None:
                    byte_bucket.acquire(sum(len(k) + len(v) for k, v in batch))
                adapter.write_batch(batch)
                if mc is not None:
                    mc.emit(
                        MetricEvent.BATCH_WRITTEN,
                        {
                            "elapsed_ms": int(
                                (time.perf_counter() - runtime.started) * 1000
                            ),
                            "batch_size": len(batch),
                        },
                    )
                batch.clear()

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
    except Exception as exc:  # pragma: no cover - worker runtime failure surface
        log_failure(
            "shard_write_failed",
            severity=FailureSeverity.ERROR,
            logger=_logger,
            error=exc,
            run_id=runtime.run_id,
            db_id=db_id,
            attempt=attempt,
            db_url=db_url,
            rows_written=row_count,
            include_traceback=True,
        )
        raise ShardyfusionError(
            f"Shard write failed for db_id={db_id}, attempt={attempt}: {exc}"
        ) from exc

    duration_ms = int((time.perf_counter() - partition_started) * 1000)
    log_event(
        "shard_write_completed",
        logger=_logger,
        run_id=runtime.run_id,
        db_id=db_id,
        attempt=attempt,
        row_count=row_count,
        duration_ms=duration_ms,
    )
    if mc is not None:
        mc.emit(
            MetricEvent.SHARD_WRITE_COMPLETED,
            {
                "elapsed_ms": int((time.perf_counter() - runtime.started) * 1000),
                "duration_ms": duration_ms,
                "row_count": row_count,
            },
        )

    writer_info: JsonObject = {
        "stage_id": stage_id,
        "task_attempt_id": task_attempt_id,
        "attempt": attempt,
        "duration_ms": duration_ms,
    }

    yield ShardAttemptResult(
        db_id=db_id,
        db_url=db_url,
        attempt=attempt,
        row_count=row_count,
        min_key=min_key,
        max_key=max_key,
        checkpoint_id=checkpoint_id,
        writer_info=writer_info,
    )
