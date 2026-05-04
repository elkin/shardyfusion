"""Public sharded snapshot writer entrypoint."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Any
from uuid import uuid4

from pyspark import RDD, TaskContext
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType

from shardyfusion._adapter import (
    DbAdapterFactory,
)
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
    VectorColumnMapping,
    assemble_build_result,
    cleanup_losers,
    discover_cel_num_dbs,
    empty_shard_result,
    inject_sqlite_btreemeta_manifest_field,
    inject_vector_manifest_fields,
    publish_to_store,
    resolve_cel_num_dbs,
    resolve_distributed_vector_fn,
    resolve_num_dbs,
    route_cel,
    route_hash,
    select_winners,
)
from shardyfusion.cel import compile_cel
from shardyfusion.config import (
    BaseShardedWriteConfig,
    CelShardedWriteConfig,
    ColumnWriteInput,
    HashShardedWriteConfig,
    SparkWriteOptions,
    validate_configs,
    validate_shard_id_col_no_collision,
)
from shardyfusion.errors import ShardAssignmentError
from shardyfusion.logging import (
    LogContext,
    get_logger,
    log_event,
)
from shardyfusion.manifest import (
    BuildResult,
    WriterInfo,
)
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.routing import KeyType, make_hash_router
from shardyfusion.run_registry import RunRecordLifecycle
from shardyfusion.serde import ValueSpec
from shardyfusion.sharding_types import (
    DB_ID_COL,
    CelShardingSpec,
    HashShardingSpec,
    KeyEncoding,
    ShardHashAlgorithm,
    ShardingSpec,
)
from shardyfusion.writer._accumulators import (
    KvAccumulator,
    ShardBatchAccumulator,
    UnifiedAccumulator,
)
from shardyfusion.writer._runtime import PartitionWriteRuntime
from shardyfusion.writer.spark.util import (
    DataFrameCacheContext,
    SparkConfOverrideContext,
)

from .sharding import (
    _discover_categorical_routing_values,
    prepare_partitioned_rdd,
)

_logger = get_logger(__name__)


@dataclass(slots=True)
class _PreparedPartitionRows:
    partitioned_rdd: RDD[tuple[int, Row]]
    resolved_sharding: ShardingSpec
    resolved_num_dbs: int
    shard_duration_ms: int


def write_hash_sharded(
    df: DataFrame,
    config: HashShardedWriteConfig,
    input: ColumnWriteInput,
    options: SparkWriteOptions | None = None,
) -> BuildResult:
    """Write a DataFrame into N independent sharded databases using HASH routing."""
    options = options or SparkWriteOptions()
    validate_configs(config, input, options)
    validate_shard_id_col_no_collision(config, set(df.columns))
    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex
    spark = df.sparkSession
    with SparkConfOverrideContext(spark, options.spark_conf_overrides):
        with DataFrameCacheContext(
            df, storage_level=options.storage_level, enabled=options.cache_input
        ) as cached_df:
            return _write_hash_sharded(
                df=cached_df,
                config=config,
                run_id=run_id,
                started=started,
                key_col=input.key_col,
                value_spec=input.value_spec,
                sort_within_partitions=options.sort_within_partitions,
                verify_routing=options.verify_routing,
                vector_fn=input.vector_fn,
                vector_columns=input.vector,
            )


def write_cel_sharded(
    df: DataFrame,
    config: CelShardedWriteConfig,
    input: ColumnWriteInput,
    options: SparkWriteOptions | None = None,
) -> BuildResult:
    """Write a DataFrame into N independent sharded databases using CEL routing."""
    options = options or SparkWriteOptions()
    validate_configs(config, input, options)
    validate_shard_id_col_no_collision(config, set(df.columns))
    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex
    spark = df.sparkSession
    with SparkConfOverrideContext(spark, options.spark_conf_overrides):
        with DataFrameCacheContext(
            df, storage_level=options.storage_level, enabled=options.cache_input
        ) as cached_df:
            return _write_cel_sharded(
                df=cached_df,
                config=config,
                run_id=run_id,
                started=started,
                key_col=input.key_col,
                value_spec=input.value_spec,
                sort_within_partitions=options.sort_within_partitions,
                verify_routing=options.verify_routing,
                vector_fn=input.vector_fn,
                vector_columns=input.vector,
            )


def _write_hash_sharded(
    *,
    df: DataFrame,
    config: HashShardedWriteConfig,
    run_id: str,
    started: float,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool,
    verify_routing: bool,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None,
    vector_columns: VectorColumnMapping | None,
) -> BuildResult:
    """Hash-specific sharded write pipeline."""
    mc = config.metrics_collector
    num_dbs = resolve_num_dbs(config, df.count)

    with (
        LogContext(run_id=run_id),
        RunRecordLifecycle.start(
            config=config,
            run_id=run_id,
            writer_type="spark",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=num_dbs,
            s3_prefix=config.s3_prefix,
            strategy="hash",
            key_encoding=config.key_encoding,
            writer_type="spark",
        )
        if mc is not None:
            mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})
        has_vectors = (
            vector_fn is not None
            or vector_columns is not None
            or config.vector_spec is not None
        )
        distributed_vector_fn = (
            resolve_distributed_vector_fn(
                config=config,
                key_col=key_col,
                vector_fn=vector_fn,
                vector_columns=vector_columns,
            )
            if has_vectors
            else None
        )

        prepared_rows = _prepare_hash_partitioned_rows(
            df=df,
            config=config,
            num_dbs=num_dbs,
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

        runtime = PartitionWriteRuntime.from_public_config(
            config=config,
            input=ColumnWriteInput(
                key_col=key_col,
                value_spec=value_spec,
            ),
            run_id=run_id,
            started=started,
            vector_fn=distributed_vector_fn,
        )
        write_outcome = _run_partition_writes(
            partitioned_rdd=prepared_rows.partitioned_rdd,
            runtime=runtime,
            num_dbs=prepared_rows.resolved_num_dbs,
        )

        return _finalize_sharded_write(
            config=config,
            run_id=run_id,
            started=started,
            prepared_rows=prepared_rows,
            write_outcome=write_outcome,
            runtime=runtime,
            mc=mc,
            run_record=run_record,
        )


def _write_cel_sharded(
    *,
    df: DataFrame,
    config: CelShardedWriteConfig,
    run_id: str,
    started: float,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool,
    verify_routing: bool,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None,
    vector_columns: VectorColumnMapping | None,
) -> BuildResult:
    """CEL-specific sharded write pipeline."""
    mc = config.metrics_collector

    with (
        LogContext(run_id=run_id),
        RunRecordLifecycle.start(
            config=config,
            run_id=run_id,
            writer_type="spark",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=None,
            s3_prefix=config.s3_prefix,
            strategy="cel",
            key_encoding=config.key_encoding,
            writer_type="spark",
        )
        if mc is not None:
            mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})

        has_vectors = (
            vector_fn is not None
            or vector_columns is not None
            or config.vector_spec is not None
        )
        distributed_vector_fn = (
            resolve_distributed_vector_fn(
                config=config,
                key_col=key_col,
                vector_fn=vector_fn,
                vector_columns=vector_columns,
            )
            if has_vectors
            else None
        )

        prepared_rows = _prepare_cel_partitioned_rows(
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

        runtime = PartitionWriteRuntime.from_public_config(
            config=config,
            input=ColumnWriteInput(
                key_col=key_col,
                value_spec=value_spec,
            ),
            run_id=run_id,
            started=started,
            vector_fn=distributed_vector_fn,
        )
        write_outcome = _run_partition_writes(
            partitioned_rdd=prepared_rows.partitioned_rdd,
            runtime=runtime,
            num_dbs=prepared_rows.resolved_num_dbs,
        )

        return _finalize_sharded_write(
            config=config,
            run_id=run_id,
            started=started,
            prepared_rows=prepared_rows,
            write_outcome=write_outcome,
            runtime=runtime,
            mc=mc,
            run_record=run_record,
        )


def _finalize_sharded_write(
    *,
    config: BaseShardedWriteConfig,
    run_id: str,
    started: float,
    prepared_rows: _PreparedPartitionRows,
    write_outcome: PartitionWriteOutcome,
    runtime: PartitionWriteRuntime,
    mc: MetricsCollector | None,
    run_record: Any,
) -> BuildResult:
    """Publish manifest, cleanup losers, and assemble BuildResult."""
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

    if config.vector_spec is not None:
        inject_vector_manifest_fields(config, runtime.adapter_factory)
    inject_sqlite_btreemeta_manifest_field(config, runtime.adapter_factory)
    manifest_started = time.perf_counter()
    manifest_ref = publish_to_store(
        config=config,
        run_id=run_id,
        resolved_sharding=prepared_rows.resolved_sharding,
        winners=write_outcome.winners,
        key_col=runtime.key_col,
        started=started,
        num_dbs=prepared_rows.resolved_num_dbs,
    )
    run_record.set_manifest_ref(manifest_ref)
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
        run_record_ref=run_record.run_record_ref,
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

    run_record.mark_succeeded()
    return result


def _verify_hash_routing_agreement(
    df_with_db_id: DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    hash_algorithm: ShardHashAlgorithm,
    key_encoding: KeyEncoding,
    shard_id_col: str = DB_ID_COL,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify Spark-computed db_id matches Python hash routing."""

    sampled = df_with_db_id.select(key_col, shard_id_col).limit(sample_size).collect()
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        key = row[key_col]
        spark_db_id = int(row[shard_id_col])
        python_db_id = route_hash(
            key,
            num_dbs=num_dbs,
            hash_algorithm=hash_algorithm,
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


def _verify_cel_routing_agreement(
    df_with_db_id: DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    cel_expr: str,
    cel_columns: dict[str, str],
    routing_values: list[int | str | bytes] | None,
    key_encoding: KeyEncoding,
    shard_id_col: str = DB_ID_COL,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify Spark-computed db_id matches Python CEL routing."""

    # Multi-column CEL: can't verify without routing_context
    if cel_columns and set(cel_columns.keys()) != {"key"}:
        return

    sampled = df_with_db_id.select(key_col, shard_id_col).limit(sample_size).collect()
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        key = row[key_col]
        spark_db_id = int(row[shard_id_col])
        python_db_id = route_cel(
            key,
            cel_expr=cel_expr,
            cel_columns=cel_columns,
            routing_values=routing_values,
            routing_context=None,
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


def _infer_spark_key_type(spark_dtype: object) -> KeyType | None:
    """Map a Spark DataType to a KeyType hint for hash-router specialization.

    Returns ``None`` when the dtype is not one of the three supported Python
    key types — the caller falls back to the generic 3-branch dispatch.
    """
    from pyspark.sql.types import (
        BinaryType,
        ByteType,
        IntegerType,
        LongType,
        ShortType,
        StringType,
    )

    if isinstance(spark_dtype, (ByteType, ShortType, IntegerType, LongType)):
        return "int"
    if isinstance(spark_dtype, StringType):
        return "str"
    if isinstance(spark_dtype, BinaryType):
        return "bytes"
    return None


def _add_db_id_column_hash(
    df: DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    hash_algorithm: ShardHashAlgorithm,
    shard_id_col: str = DB_ID_COL,
) -> DataFrame:
    """Add deterministic db id column for HASH routing via mapInArrow."""

    output_schema = StructType(
        list(df.schema.fields) + [StructField(shard_id_col, IntegerType(), False)]
    )

    _key_col = key_col
    _num_dbs = num_dbs
    _hash_algorithm = hash_algorithm
    _shard_id_col = shard_id_col
    _key_type = _infer_spark_key_type(df.schema[key_col].dataType)

    def _hash_map_arrow(iterator):  # type: ignore[no-untyped-def]
        import pyarrow as pa  # type: ignore[import-not-found]

        # Build the routing closure once per worker partition; resolves the
        # algorithm enum and (when statically known) specializes on key type.
        route = make_hash_router(_num_dbs, _hash_algorithm, key_type=_key_type)
        for batch in iterator:
            keys = batch.column(_key_col).to_pylist()
            db_ids = [route(k) for k in keys]
            yield batch.append_column(_shard_id_col, pa.array(db_ids, type=pa.int32()))

    df_with_db_id = df.mapInArrow(_hash_map_arrow, output_schema)

    invalid_count = (
        df_with_db_id.where(
            (F.col(shard_id_col).isNull())
            | (F.col(shard_id_col) < 0)
            | (F.col(shard_id_col) >= num_dbs)
        )
        .limit(1)
        .count()
    )
    if invalid_count > 0:
        raise ShardAssignmentError("Computed db_id out of range [0, num_dbs-1].")

    return df_with_db_id


def _add_db_id_column_cel(
    df: DataFrame,
    *,
    key_col: str,
    cel_expr: str,
    cel_columns: dict[str, str],
    routing_values: list[int | str | bytes] | None,
    infer_routing_values_from_data: bool,
    shard_id_col: str = DB_ID_COL,
) -> tuple[DataFrame, CelShardingSpec]:
    """Add deterministic db id column for CEL routing via mapInArrow.

    Returns the modified DataFrame and the resolved CelShardingSpec.
    """

    output_schema = StructType(
        list(df.schema.fields) + [StructField(shard_id_col, IntegerType(), False)]
    )

    resolved_routing_values = (
        _discover_categorical_routing_values(
            df,
            cel_expr=cel_expr,
            cel_columns=dict(cel_columns),
        )
        if infer_routing_values_from_data
        else (list(routing_values) if routing_values is not None else None)
    )
    resolved = CelShardingSpec(
        cel_expr=cel_expr,
        cel_columns=dict(cel_columns),
        routing_values=resolved_routing_values,
        infer_routing_values_from_data=False,
    )

    _cel_expr = cel_expr
    _cel_cols = dict(cel_columns)
    _cel_routing_values = resolved_routing_values
    _shard_id_col = shard_id_col

    compile_cel(_cel_expr, _cel_cols)  # validate eagerly on driver

    def _cel_map_arrow(iterator):  # type: ignore[no-untyped-def]
        import pyarrow as pa  # type: ignore[import-not-found]

        from shardyfusion.cel import compile_cel_cached, route_cel_batch

        # Cached: shared across all batches in a single worker process.
        _compiled = compile_cel_cached(_cel_expr, tuple(sorted(_cel_cols.items())))
        for batch in iterator:
            db_ids = route_cel_batch(
                _compiled,
                batch,
                _cel_routing_values,
            )
            yield batch.append_column(_shard_id_col, pa.array(db_ids, type=pa.int32()))

    df_with_db_id = df.mapInArrow(_cel_map_arrow, output_schema)

    return df_with_db_id, resolved


def _prepare_hash_partitioned_rows(
    *,
    df: DataFrame,
    config: HashShardedWriteConfig,
    num_dbs: int,
    key_col: str,
    sort_within_partitions: bool,
    verify_routing: bool = True,
) -> _PreparedPartitionRows:
    """Assign shard ids and build the partitioned RDD for HASH routing."""
    shard_started = time.perf_counter()
    shard_id_col = config.shard_id_col

    sharding = HashShardingSpec(
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
        max_keys_per_shard=config.max_keys_per_shard,
    )
    df_with_db_id = _add_db_id_column_hash(
        df,
        key_col=key_col,
        num_dbs=num_dbs,
        hash_algorithm=sharding.hash_algorithm,
        shard_id_col=shard_id_col,
    )

    if verify_routing and num_dbs > 0:
        _verify_hash_routing_agreement(
            df_with_db_id,
            key_col=key_col,
            num_dbs=num_dbs,
            hash_algorithm=sharding.hash_algorithm,
            key_encoding=config.key_encoding,
            shard_id_col=shard_id_col,
        )
    partitioned_rdd = prepare_partitioned_rdd(
        df_with_db_id,
        num_dbs=num_dbs,
        key_col=key_col,
        sort_within_partitions=sort_within_partitions,
        shard_id_col=shard_id_col,
    )
    shard_duration_ms = int((time.perf_counter() - shard_started) * 1000)
    return _PreparedPartitionRows(
        partitioned_rdd=partitioned_rdd,
        resolved_sharding=sharding,
        resolved_num_dbs=num_dbs,
        shard_duration_ms=shard_duration_ms,
    )


def _prepare_cel_partitioned_rows(
    *,
    df: DataFrame,
    config: CelShardedWriteConfig,
    key_col: str,
    sort_within_partitions: bool,
    verify_routing: bool = True,
) -> _PreparedPartitionRows:
    """Assign shard ids and build the partitioned RDD for CEL routing."""
    shard_started = time.perf_counter()
    shard_id_col = config.shard_id_col

    df_with_db_id, resolved_sharding = _add_db_id_column_cel(
        df,
        key_col=key_col,
        cel_expr=config.cel_expr,
        cel_columns=config.cel_columns,
        routing_values=config.routing_values,
        infer_routing_values_from_data=config.infer_routing_values_from_data,
        shard_id_col=shard_id_col,
    )

    # Discover num_dbs from data and validate consecutive IDs
    if resolved_sharding.routing_values is not None:
        num_dbs = resolve_cel_num_dbs(resolved_sharding)
    else:
        agg_row = df_with_db_id.agg(
            F.max(shard_id_col).alias("max_id"),
            F.count_distinct(shard_id_col).alias("n_distinct"),
        ).collect()[0]
        max_id = agg_row["max_id"]
        n_distinct = agg_row["n_distinct"]
        distinct_ids = (
            set(range(n_distinct))
            if n_distinct == (max_id + 1 if max_id is not None else 0)
            else {
                row[0]
                for row in df_with_db_id.select(shard_id_col).distinct().collect()
            }
        )
        num_dbs = discover_cel_num_dbs(distinct_ids)

    if verify_routing and num_dbs > 0:
        _verify_cel_routing_agreement(
            df_with_db_id,
            key_col=key_col,
            num_dbs=num_dbs,
            cel_expr=resolved_sharding.cel_expr,
            cel_columns=resolved_sharding.cel_columns,
            routing_values=resolved_sharding.routing_values,
            key_encoding=config.key_encoding,
            shard_id_col=shard_id_col,
        )
    partitioned_rdd = prepare_partitioned_rdd(
        df_with_db_id,
        num_dbs=num_dbs,
        key_col=key_col,
        sort_within_partitions=sort_within_partitions,
        shard_id_col=shard_id_col,
    )
    shard_duration_ms = int((time.perf_counter() - shard_started) * 1000)
    return _PreparedPartitionRows(
        partitioned_rdd=partitioned_rdd,
        resolved_sharding=resolved_sharding,
        resolved_num_dbs=num_dbs,
        shard_duration_ms=shard_duration_ms,
    )


def _run_partition_writes(
    *,
    partitioned_rdd: RDD[tuple[int, Row]],
    runtime: PartitionWriteRuntime,
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


def _drop_shard_id_col(row: Row, shard_id_col: str) -> Row:
    """Return a new Row without the internal shard_id column."""
    d = row.asDict()
    d.pop(shard_id_col, None)
    return Row(**d)


def write_one_shard_partition(
    db_id: int,
    rows_iter: Iterable[tuple[int, Row]],
    runtime: PartitionWriteRuntime,
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

    factory: DbAdapterFactory = runtime.adapter_factory

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

    shard_id_col = runtime.shard_id_col
    accumulator_factory: type[ShardBatchAccumulator]
    if runtime.vector_fn is not None:
        accumulator_factory = UnifiedAccumulator

        def _iter_spark_rows_with_vectors() -> Iterable[
            tuple[object, bytes, tuple[int | str, Any, dict[str, Any] | None] | None]
        ]:
            assert runtime.vector_fn is not None  # unnecessary but satisfies pyright
            for _, row in rows_iter:
                clean_row = _drop_shard_id_col(row, shard_id_col)
                yield (
                    clean_row[runtime.key_col],
                    runtime.value_spec.encode(clean_row),
                    runtime.vector_fn(clean_row),
                )

        iter_spark_rows = _iter_spark_rows_with_vectors
    else:
        accumulator_factory = KvAccumulator

        def _iter_spark_rows_without_vectors() -> Iterable[
            tuple[object, bytes, tuple[int | str, Any, dict[str, Any] | None] | None]
        ]:
            for _, row in rows_iter:
                clean_row = _drop_shard_id_col(row, shard_id_col)
                yield (
                    clean_row[runtime.key_col],
                    runtime.value_spec.encode(clean_row),
                    None,
                )

        iter_spark_rows = _iter_spark_rows_without_vectors

    result = write_shard_core(
        params,
        iter_spark_rows(),
        accumulator_factory=accumulator_factory,
        writer_info_base=WriterInfo(
            stage_id=stage_id,
            task_attempt_id=task_attempt_id,
            attempt=attempt,
        ),
    )
    yield result
