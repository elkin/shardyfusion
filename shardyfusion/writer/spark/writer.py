"""Public sharded snapshot writer entrypoint."""

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from uuid import uuid4

from pyspark import RDD, StorageLevel, TaskContext
from pyspark.sql import DataFrame, Row

from shardyfusion._rate_limiter import RateLimiter, TokenBucket
from shardyfusion._shard_writer import (
    ShardWriteParams,
    make_shard_local_dir,
    make_shard_url,
    write_shard_core,
)
from shardyfusion._writer_core import (
    PartitionWriteOutcome,
    ShardAttemptResult,
    assemble_build_result,
    cleanup_losers,
    empty_shard_result,
    publish_to_store,
    select_winners,
)
from shardyfusion.config import WriteConfig
from shardyfusion.credentials import CredentialProvider
from shardyfusion.errors import ShardAssignmentError
from shardyfusion.logging import (
    get_logger,
    log_event,
)
from shardyfusion.manifest import BuildResult, WriterInfo
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.serde import KeyEncoder, ValueSpec, make_key_encoder
from shardyfusion.sharding_types import DB_ID_COL, KeyEncoding
from shardyfusion.slatedb_adapter import (
    DbAdapterFactory,
    SlateDbFactory,
)
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
    shard_prefix: str
    db_path_template: str
    local_root: str
    key_col: str
    key_encoding: KeyEncoding
    key_encoder: KeyEncoder
    value_spec: ValueSpec
    batch_size: int
    adapter_factory: DbAdapterFactory | None
    credential_provider: CredentialProvider | None
    max_writes_per_second: float | None
    max_write_bytes_per_second: float | None = None
    metrics_collector: MetricsCollector | None = None  # must be picklable
    started: float = 0.0


@dataclass(slots=True)
class _PreparedPartitionRows:
    partitioned_rdd: RDD[tuple[int, Row]]
    resolved_sharding: ShardingSpec
    resolved_num_dbs: int
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
        num_dbs = prepared_rows.resolved_num_dbs
        write_outcome = _run_partition_writes(
            partitioned_rdd=prepared_rows.partitioned_rdd,
            runtime=runtime,
            num_dbs=num_dbs,
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
            num_dbs=num_dbs,
        )
        manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

        cleanup_losers(
            write_outcome.all_attempt_urls,
            write_outcome.winners,
            metrics_collector=mc,
        )

        result = assemble_build_result(
            run_id=run_id,
            winners=write_outcome.winners,
            manifest_ref=manifest_ref,
            num_attempts=write_outcome.num_attempts,
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

    Since all strategies now use Python-based mapInArrow, this is a
    safety net verifying the mapInArrow output against the reader's
    route_key() function.
    """
    from shardyfusion._writer_core import route_key
    from shardyfusion.sharding_types import ShardingStrategy

    # CEL multi-column: can't verify without routing_context
    if (
        resolved_sharding.strategy == ShardingStrategy.CEL
        and resolved_sharding.cel_columns
        and set(resolved_sharding.cel_columns.keys()) != {"key"}
    ):
        return

    sampled = df_with_db_id.select(key_col, DB_ID_COL).limit(sample_size).collect()
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        key = row[key_col]
        spark_db_id = int(row[DB_ID_COL])

        python_db_id = route_key(
            key,
            num_dbs=num_dbs,
            sharding=resolved_sharding,
            key_encoding=key_encoding,
        )

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


def _resolve_num_dbs_before_sharding(df: DataFrame, config: WriteConfig) -> int:
    """Resolve num_dbs that can be determined before add_db_id_column."""
    from shardyfusion._writer_core import resolve_num_dbs

    return resolve_num_dbs(config, df.count)


def _prepare_partitioned_rows(
    *,
    df: DataFrame,
    config: WriteConfig,
    key_col: str,
    sort_within_partitions: bool,
    verify_routing: bool = True,
) -> _PreparedPartitionRows:
    """Assign shard ids and build the one-writer-per-db partitioned RDD."""

    from pyspark.sql import functions as F

    from shardyfusion._writer_core import discover_cel_num_dbs

    shard_started = time.perf_counter()
    num_dbs = _resolve_num_dbs_before_sharding(df, config)

    df_with_db_id, resolved_sharding = add_db_id_column(
        df,
        key_col=key_col,
        num_dbs=num_dbs,
        sharding=config.sharding,
    )

    # CEL: discover num_dbs from data and validate consecutive IDs
    if num_dbs == 0:
        agg_row = df_with_db_id.agg(
            F.max(DB_ID_COL).alias("max_id"),
            F.countDistinct(DB_ID_COL).alias("n_distinct"),
        ).collect()[0]
        max_id = agg_row["max_id"]
        n_distinct = agg_row["n_distinct"]
        distinct_ids = (
            set(range(n_distinct))
            if n_distinct == (max_id + 1 if max_id is not None else 0)
            else {
                row[0] for row in df_with_db_id.select(DB_ID_COL).distinct().collect()
            }
        )
        num_dbs = discover_cel_num_dbs(distinct_ids)

    if verify_routing and num_dbs > 0:
        verify_routing_agreement(
            df_with_db_id,
            key_col=key_col,
            num_dbs=num_dbs,
            resolved_sharding=resolved_sharding,
            key_encoding=config.key_encoding,
        )
    partitioned_rdd = prepare_partitioned_rdd(
        df_with_db_id,
        num_dbs=num_dbs,
        key_col=key_col,
        sort_within_partitions=sort_within_partitions,
    )
    shard_duration_ms = int((time.perf_counter() - shard_started) * 1000)
    return _PreparedPartitionRows(
        partitioned_rdd=partitioned_rdd,
        resolved_sharding=resolved_sharding,
        resolved_num_dbs=num_dbs,
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
        shard_prefix=config.output.shard_prefix,
        db_path_template=config.output.db_path_template,
        local_root=config.output.local_root,
        key_col=key_col,
        key_encoding=config.key_encoding,
        key_encoder=make_key_encoder(config.key_encoding),
        value_spec=value_spec,
        batch_size=config.batch_size,
        adapter_factory=config.adapter_factory,
        credential_provider=config.credential_provider,
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
    results_rdd = partitioned_rdd.mapPartitionsWithIndex(
        lambda db_id, items: write_one_shard_partition(db_id, items, runtime)
    )
    winners, num_attempts, all_attempt_urls = select_winners(
        results_rdd.toLocalIterator(), num_dbs=num_dbs
    )
    write_duration_ms = int((time.perf_counter() - write_started) * 1000)
    return PartitionWriteOutcome(
        num_attempts=num_attempts,
        winners=winners,
        all_attempt_urls=all_attempt_urls,
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

    # Peek at first row — if partition is empty, emit metadata-only result
    # without opening a SlateDB adapter (avoids S3 I/O for empty shards).
    it = iter(rows_iter)
    try:
        first = next(it)
    except StopIteration:
        yield empty_shard_result(
            db_id,
            attempt=attempt,
            writer_info=WriterInfo(
                stage_id=stage_id,
                task_attempt_id=task_attempt_id,
                attempt=attempt,
                duration_ms=0,
            ),
        )
        return

    rows_iter = chain([first], it)

    db_url = make_shard_url(
        runtime.s3_prefix,
        runtime.shard_prefix,
        runtime.run_id,
        runtime.db_path_template,
        db_id,
        attempt,
    )
    local_dir = make_shard_local_dir(runtime.local_root, runtime.run_id, db_id, attempt)

    factory: DbAdapterFactory = runtime.adapter_factory or SlateDbFactory(
        credential_provider=runtime.credential_provider
    )

    ops_limiter: RateLimiter | None = None
    if runtime.max_writes_per_second is not None:
        ops_limiter = TokenBucket(
            runtime.max_writes_per_second, metrics_collector=runtime.metrics_collector
        )
    bytes_limiter: RateLimiter | None = None
    if runtime.max_write_bytes_per_second is not None:
        bytes_limiter = TokenBucket(
            runtime.max_write_bytes_per_second,
            metrics_collector=runtime.metrics_collector,
            limiter_type="bytes",
        )

    params = ShardWriteParams(
        db_id=db_id,
        attempt=attempt,
        run_id=runtime.run_id,
        db_url=db_url,
        local_dir=local_dir,
        factory=factory,
        key_encoder=runtime.key_encoder,
        batch_size=runtime.batch_size,
        ops_limiter=ops_limiter,
        bytes_limiter=bytes_limiter,
        metrics_collector=runtime.metrics_collector,
        started=runtime.started,
    )

    def _iter_spark_rows() -> Iterable[tuple[object, bytes]]:
        for _, row in rows_iter:
            yield row[runtime.key_col], runtime.value_spec.encode(row)

    result = write_shard_core(
        params,
        _iter_spark_rows(),
        writer_info_base=WriterInfo(
            stage_id=stage_id,
            task_attempt_id=task_attempt_id,
            attempt=attempt,
        ),
    )
    yield result
