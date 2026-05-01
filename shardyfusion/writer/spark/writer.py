"""Public sharded snapshot writer entrypoint."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pyspark import RDD, StorageLevel, TaskContext
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType

from shardyfusion.vector._distributed import ResolvedVectorRouting

if TYPE_CHECKING:
    import numpy as np

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
    inject_vector_manifest_fields,
    publish_to_store,
    resolve_cel_num_dbs,
    resolve_distributed_vector_fn,
    resolve_num_dbs,
    route_cel,
    route_hash,
    select_winners,
    wrap_factory_for_vector,
)
from shardyfusion.cel import compile_cel
from shardyfusion.config import CelWriteConfig, HashWriteConfig, WriteConfig
from shardyfusion.credentials import CredentialProvider
from shardyfusion.errors import ConfigValidationError, ShardAssignmentError
from shardyfusion.logging import (
    LogContext,
    get_logger,
    log_event,
)
from shardyfusion.manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    RequiredShardMeta,
    WriterInfo,
)
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.routing import hash_db_id
from shardyfusion.run_registry import RunRecordLifecycle
from shardyfusion.serde import KeyEncoder, ValueSpec, make_key_encoder
from shardyfusion.sharding_types import (
    DB_ID_COL,
    CelShardingSpec,
    HashShardingSpec,
    KeyEncoding,
    ShardHashAlgorithm,
    ShardingSpec,
)
from shardyfusion.slatedb_adapter import (
    DbAdapterFactory,
    SlateDbFactory,
)
from shardyfusion.storage import ObstoreBackend, create_s3_store, parse_s3_url
from shardyfusion.vector.config import VectorWriteConfig
from shardyfusion.vector.types import VectorIndexWriterFactory
from shardyfusion.writer._accumulators import (
    KvAccumulator,
    ShardBatchAccumulator,
    UnifiedAccumulator,
)
from shardyfusion.writer.spark.util import (
    DataFrameCacheContext,
    SparkConfOverrideContext,
)

from .sharding import (
    VECTOR_DB_ID_COL,
    _discover_categorical_routing_values,
    add_vector_db_id_column,
    prepare_partitioned_rdd,
)

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
    adapter_factory: DbAdapterFactory
    credential_provider: CredentialProvider | None
    max_writes_per_second: float | None
    max_write_bytes_per_second: float | None = None
    metrics_collector: MetricsCollector | None = None  # must be picklable
    started: float = 0.0
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None = (
        None
    )


@dataclass(slots=True)
class _PreparedPartitionRows:
    partitioned_rdd: RDD[tuple[int, Row]]
    resolved_sharding: ShardingSpec
    resolved_num_dbs: int
    shard_duration_ms: int


def write_sharded_by_hash(
    df: DataFrame,
    config: HashWriteConfig,
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
    vector_fn: (
        Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None
    ) = None,
    vector_columns: VectorColumnMapping | None = None,
) -> BuildResult:
    """Write a DataFrame into N independent sharded databases using HASH routing."""
    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex
    spark = df.sparkSession
    with SparkConfOverrideContext(spark, spark_conf_overrides):
        with DataFrameCacheContext(
            df, storage_level=storage_level, enabled=cache_input
        ) as cached_df:
            return _write_hash_sharded(
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
                vector_fn=vector_fn,
                vector_columns=vector_columns,
            )


def write_sharded_by_cel(
    df: DataFrame,
    config: CelWriteConfig,
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
    vector_fn: (
        Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None
    ) = None,
    vector_columns: VectorColumnMapping | None = None,
) -> BuildResult:
    """Write a DataFrame into N independent sharded databases using CEL routing."""
    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex
    spark = df.sparkSession
    with SparkConfOverrideContext(spark, spark_conf_overrides):
        with DataFrameCacheContext(
            df, storage_level=storage_level, enabled=cache_input
        ) as cached_df:
            return _write_cel_sharded(
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
                vector_fn=vector_fn,
                vector_columns=vector_columns,
            )


def _write_hash_sharded(
    *,
    df: DataFrame,
    config: HashWriteConfig,
    run_id: str,
    started: float,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
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

        runtime = _build_partition_write_runtime(
            config=config,
            run_id=run_id,
            key_col=key_col,
            value_spec=value_spec,
            max_writes_per_second=max_writes_per_second,
            max_write_bytes_per_second=max_write_bytes_per_second,
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
    config: CelWriteConfig,
    run_id: str,
    started: float,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
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

        runtime = _build_partition_write_runtime(
            config=config,
            run_id=run_id,
            key_col=key_col,
            value_spec=value_spec,
            max_writes_per_second=max_writes_per_second,
            max_write_bytes_per_second=max_write_bytes_per_second,
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
    config: WriteConfig,
    run_id: str,
    started: float,
    prepared_rows: _PreparedPartitionRows,
    write_outcome: PartitionWriteOutcome,
    runtime: PartitionWriteConfig,
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
    sample_size: int = 20,
) -> None:
    """Sample rows and verify Spark-computed db_id matches Python hash routing."""

    sampled = df_with_db_id.select(key_col, DB_ID_COL).limit(sample_size).collect()
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        key = row[key_col]
        spark_db_id = int(row[DB_ID_COL])
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
    sample_size: int = 20,
) -> None:
    """Sample rows and verify Spark-computed db_id matches Python CEL routing."""

    # Multi-column CEL: can't verify without routing_context
    if cel_columns and set(cel_columns.keys()) != {"key"}:
        return

    sampled = df_with_db_id.select(key_col, DB_ID_COL).limit(sample_size).collect()
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        key = row[key_col]
        spark_db_id = int(row[DB_ID_COL])
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


def verify_vector_routing_agreement(
    df_with_db_id: DataFrame,
    *,
    id_col: str,
    vector_col: str,
    routing: ResolvedVectorRouting,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify Spark-computed vector db ids match Python routing."""
    from shardyfusion.vector._distributed import (
        assign_vector_shard,
        coerce_vector_value,
    )
    from shardyfusion.vector.types import VectorShardingStrategy

    sample_cols = [id_col, vector_col, VECTOR_DB_ID_COL]
    if (
        routing.strategy == VectorShardingStrategy.EXPLICIT
        and shard_id_col is not None
        and shard_id_col not in sample_cols
    ):
        sample_cols.append(shard_id_col)
    if routing.strategy == VectorShardingStrategy.CEL:
        assert routing_context_cols is not None, (
            "routing_context_cols required for CEL verification"
        )
        sample_cols.extend(c for c in routing_context_cols if c not in sample_cols)

    sampled = df_with_db_id.select(*sample_cols).limit(sample_size).collect()
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        shard_id: int | None = None
        if routing.strategy == VectorShardingStrategy.EXPLICIT:
            assert shard_id_col is not None, "shard_id_col required for EXPLICIT"
            shard_id = int(row[shard_id_col])

        routing_context: dict[str, Any] | None = None
        if routing.strategy == VectorShardingStrategy.CEL:
            assert routing_context_cols is not None, (
                "routing_context_cols required for CEL"
            )
            routing_context = {col: row[col] for col in routing_context_cols}

        expected_db_id = assign_vector_shard(
            vector=coerce_vector_value(row[vector_col]),
            routing=routing,
            shard_id=shard_id,
            routing_context=routing_context,
        )
        spark_db_id = int(row[VECTOR_DB_ID_COL])
        if expected_db_id != spark_db_id:
            mismatches.append((row[id_col], spark_db_id, expected_db_id))

    if mismatches:
        details = "; ".join(
            f"id={vector_id}, spark={spark_db_id}, python={python_db_id}"
            for vector_id, spark_db_id, python_db_id in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Spark/Python vector routing mismatch in {len(mismatches)}/{len(sampled)} "
            f"sampled rows. First mismatches: {details}"
        )


def _add_db_id_column_hash(
    df: DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    hash_algorithm: ShardHashAlgorithm,
) -> DataFrame:
    """Add deterministic db id column for HASH routing via mapInArrow."""
    from shardyfusion.sharding_types import DB_ID_COL

    output_schema = StructType(
        list(df.schema.fields) + [StructField(DB_ID_COL, IntegerType(), False)]
    )

    _key_col = key_col
    _num_dbs = num_dbs
    _hash_algorithm = hash_algorithm

    def _hash_map_arrow(iterator):  # type: ignore[no-untyped-def]
        import pyarrow as pa  # type: ignore[import-not-found]

        for batch in iterator:
            keys = batch.column(_key_col).to_pylist()
            db_ids = [hash_db_id(k, _num_dbs, _hash_algorithm) for k in keys]
            yield batch.append_column(DB_ID_COL, pa.array(db_ids, type=pa.int32()))

    df_with_db_id = df.mapInArrow(_hash_map_arrow, output_schema)

    invalid_count = (
        df_with_db_id.where(
            (F.col(DB_ID_COL).isNull())
            | (F.col(DB_ID_COL) < 0)
            | (F.col(DB_ID_COL) >= num_dbs)
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
) -> tuple[DataFrame, CelShardingSpec]:
    """Add deterministic db id column for CEL routing via mapInArrow.

    Returns the modified DataFrame and the resolved CelShardingSpec.
    """
    from shardyfusion.sharding_types import DB_ID_COL

    output_schema = StructType(
        list(df.schema.fields) + [StructField(DB_ID_COL, IntegerType(), False)]
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

    compile_cel(_cel_expr, _cel_cols)  # validate eagerly on driver

    def _cel_map_arrow(iterator):  # type: ignore[no-untyped-def]
        import pyarrow as pa  # type: ignore[import-not-found]

        from shardyfusion.cel import compile_cel as _compile
        from shardyfusion.cel import route_cel_batch

        _compiled = _compile(_cel_expr, _cel_cols)
        for batch in iterator:
            db_ids = route_cel_batch(
                _compiled,
                batch,
                _cel_routing_values,
            )
            yield batch.append_column(DB_ID_COL, pa.array(db_ids, type=pa.int32()))

    df_with_db_id = df.mapInArrow(_cel_map_arrow, output_schema)

    return df_with_db_id, resolved


def _prepare_hash_partitioned_rows(
    *,
    df: DataFrame,
    config: HashWriteConfig,
    num_dbs: int,
    key_col: str,
    sort_within_partitions: bool,
    verify_routing: bool = True,
) -> _PreparedPartitionRows:
    """Assign shard ids and build the partitioned RDD for HASH routing."""
    shard_started = time.perf_counter()

    sharding = HashShardingSpec(
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
        max_keys_per_shard=config.max_keys_per_shard,
    )
    df_with_db_id = _add_db_id_column_hash(
        df,
        key_col=key_col,
        num_dbs=num_dbs,
        hash_algorithm=sharding.hash_algorithm,
    )

    if verify_routing and num_dbs > 0:
        _verify_hash_routing_agreement(
            df_with_db_id,
            key_col=key_col,
            num_dbs=num_dbs,
            hash_algorithm=sharding.hash_algorithm,
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
        resolved_sharding=sharding,
        resolved_num_dbs=num_dbs,
        shard_duration_ms=shard_duration_ms,
    )


def _prepare_cel_partitioned_rows(
    *,
    df: DataFrame,
    config: CelWriteConfig,
    key_col: str,
    sort_within_partitions: bool,
    verify_routing: bool = True,
) -> _PreparedPartitionRows:
    """Assign shard ids and build the partitioned RDD for CEL routing."""
    shard_started = time.perf_counter()

    df_with_db_id, resolved_sharding = _add_db_id_column_cel(
        df,
        key_col=key_col,
        cel_expr=config.cel_expr,
        cel_columns=config.cel_columns,
        routing_values=config.routing_values,
        infer_routing_values_from_data=config.infer_routing_values_from_data,
    )

    # Discover num_dbs from data and validate consecutive IDs
    if resolved_sharding.routing_values is not None:
        num_dbs = resolve_cel_num_dbs(resolved_sharding)
    else:
        agg_row = df_with_db_id.agg(
            F.max(DB_ID_COL).alias("max_id"),
            F.count_distinct(DB_ID_COL).alias("n_distinct"),
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
        _verify_cel_routing_agreement(
            df_with_db_id,
            key_col=key_col,
            num_dbs=num_dbs,
            cel_expr=resolved_sharding.cel_expr,
            cel_columns=resolved_sharding.cel_columns,
            routing_values=resolved_sharding.routing_values,
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
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
) -> PartitionWriteConfig:
    """Construct immutable worker-side runtime config for partition shard writers."""

    factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory(
        credential_provider=config.credential_provider
    )
    if config.vector_spec is not None:
        factory = wrap_factory_for_vector(factory, config)

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
        adapter_factory=factory,
        credential_provider=config.credential_provider,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
        metrics_collector=config.metrics_collector,
        started=started,
        vector_fn=vector_fn,
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

    accumulator_factory: type[ShardBatchAccumulator]
    if runtime.vector_fn is not None:
        accumulator_factory = UnifiedAccumulator

        def _iter_spark_rows_with_vectors() -> Iterable[
            tuple[object, bytes, tuple[int | str, Any, dict[str, Any] | None] | None]
        ]:
            assert runtime.vector_fn is not None  # unnecessary but satisfies pyright
            for _, row in rows_iter:
                yield (
                    row[runtime.key_col],
                    runtime.value_spec.encode(row),
                    runtime.vector_fn(row),
                )

        iter_spark_rows = _iter_spark_rows_with_vectors
    else:
        accumulator_factory = KvAccumulator

        def _iter_spark_rows_without_vectors() -> Iterable[
            tuple[object, bytes, tuple[int | str, Any, dict[str, Any] | None] | None]
        ]:
            for _, row in rows_iter:
                yield row[runtime.key_col], runtime.value_spec.encode(row), None

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


def write_vector_sharded(
    df: DataFrame,
    config: HashWriteConfig,
    *,
    vector_col: str,
    id_col: str,
    payload_cols: list[str] | None = None,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
    max_writes_per_second: float | None = None,
    verify_routing: bool = True,
) -> BuildResult:
    """Write vectors from a DataFrame into N sharded vector indices and publish manifest.

    Args:
        df: PySpark DataFrame with vector and ID columns.
        config: Write configuration (num_dbs, s3_prefix, sharding strategy, etc.).
        vector_col: Name of the vector column (list[float] or array[float]).
        id_col: Name of the ID column.
        payload_cols: Optional payload columns to include as metadata.
        shard_id_col: Column with explicit shard IDs (EXPLICIT strategy only).
        routing_context_cols: Columns for CEL expression evaluation (CEL strategy only).
        max_writes_per_second: Optional rate limit.
        verify_routing: If True (default), spot-check that Spark-assigned vector
            shard IDs match ``assign_vector_shard()``.

    Returns:
        BuildResult with manifest reference and stats.
    """
    import numpy as np

    from shardyfusion.vector._distributed import (
        write_vector_shard as write_vector_shard_core,
    )

    if config.vector_spec is None:
        raise ConfigValidationError("vector_spec is required for write_vector_sharded")

    started = time.perf_counter()
    mc = config.metrics_collector
    from shardyfusion.vector._distributed import (
        publish_vector_manifest,
        resolve_adapter_factory,
        resolve_vector_routing,
        upload_routing_metadata,
    )

    run_id = config.output.run_id or uuid4().hex

    with RunRecordLifecycle.start(
        config=config,
        run_id=run_id,
        writer_type="vector-spark",
    ) as run_record:
        sample_vectors: np.ndarray | None = None
        if (
            config.vector_spec.sharding.strategy == "cluster"
            and config.vector_spec.sharding.train_centroids
        ):
            sample_limit = config.vector_spec.sharding.centroids_training_sample_size
            sample_df = (
                df.select(vector_col).sample(fraction=0.1, seed=42).limit(sample_limit)
            )
            sample_rows = sample_df.collect()
            sample_vectors = np.array(
                [row[vector_col] for row in sample_rows if row[vector_col] is not None],
                dtype=np.float32,
            )

        resolved_vector_factory: VectorIndexWriterFactory | None = None
        if config.adapter_factory is not None:
            candidate_factory: Any = config.adapter_factory
            if getattr(candidate_factory, "supports_vector_writes", False) is True:
                resolved_vector_factory = candidate_factory

        vec_config = VectorWriteConfig(
            num_dbs=config.num_dbs,
            s3_prefix=config.s3_prefix,
            index_config=config.vector_spec.to_vector_index_config(),
            sharding=config.vector_spec.to_vector_sharding_spec(),
            output=config.output,
            adapter_factory=resolved_vector_factory,
            batch_size=config.batch_size,
            credential_provider=config.credential_provider,
            s3_connection_options=config.s3_connection_options,
            max_writes_per_second=max_writes_per_second,
            metrics_collector=mc,
            manifest=config.manifest,
        )

        routing = resolve_vector_routing(vec_config, sample_vectors=sample_vectors)

        adapter_factory = resolve_adapter_factory(vec_config)

        df_with_id, num_dbs = add_vector_db_id_column(
            df,
            vector_col=vector_col,
            routing=routing,
            shard_id_col=shard_id_col,
            routing_context_cols=routing_context_cols,
        )

        if verify_routing and num_dbs > 0:
            verify_vector_routing_agreement(
                df_with_id,
                id_col=id_col,
                vector_col=vector_col,
                routing=routing,
                shard_id_col=shard_id_col,
                routing_context_cols=routing_context_cols,
            )

        _id_col = id_col
        _vector_col = vector_col
        _payload_cols = payload_cols
        _max_writes_per_second = max_writes_per_second
        _metrics_collector = mc
        _s3_prefix = vec_config.s3_prefix
        _shard_prefix = vec_config.output.shard_prefix
        _db_path_template = vec_config.output.db_path_template
        _local_root = vec_config.output.local_root
        _index_config = vec_config.index_config
        _batch_size = vec_config.batch_size

        df_shuffled = df_with_id.repartition(num_dbs, VECTOR_DB_ID_COL)

        results_rdd = df_shuffled.rdd.mapPartitionsWithIndex(
            lambda db_id, rows: _write_vector_partition(
                db_id,
                rows,
                run_id,
                adapter_factory,
                _max_writes_per_second,
                _id_col,
                _vector_col,
                _payload_cols,
                write_vector_shard_core,
                _metrics_collector,
                _s3_prefix,
                _shard_prefix,
                _db_path_template,
                _local_root,
                _index_config,
                _batch_size,
            )
        )

        results = list(results_rdd.toLocalIterator())
        winners = results
        num_attempts = len(results)

        total_vectors = sum(w.row_count for w in winners)
        total_duration_ms = int((time.perf_counter() - started) * 1000)

        credentials = (
            config.credential_provider.resolve() if config.credential_provider else None
        )
        bucket, _ = parse_s3_url(config.s3_prefix)
        backend = ObstoreBackend(
            create_s3_store(
                bucket=bucket,
                credentials=credentials,
                connection_options=config.s3_connection_options,
            )
        )
        centroids_ref, hyperplanes_ref = upload_routing_metadata(
            s3_prefix=config.s3_prefix,
            run_id=run_id,
            centroids=routing.centroids,
            hyperplanes=routing.hyperplanes,
            backend=backend,
        )

        manifest_start = time.perf_counter()
        manifest_ref = publish_vector_manifest(
            config=vec_config,
            run_id=run_id,
            num_dbs=num_dbs,
            winners=winners,
            total_vectors=total_vectors,
            centroids_ref=centroids_ref,
            hyperplanes_ref=hyperplanes_ref,
        )
        manifest_duration_ms = int((time.perf_counter() - manifest_start) * 1000)
        run_record.set_manifest_ref(manifest_ref)
        run_record.mark_succeeded()

        stats = BuildStats(
            durations=BuildDurations(
                sharding_ms=0,
                write_ms=total_duration_ms - manifest_duration_ms,
                manifest_ms=manifest_duration_ms,
                total_ms=total_duration_ms,
            ),
            num_attempt_results=num_attempts,
            num_winners=len(winners),
            rows_written=total_vectors,
        )

        return BuildResult(
            run_id=run_id,
            winners=winners,
            manifest_ref=manifest_ref,
            stats=stats,
            run_record_ref=run_record.run_record_ref,
        )


def _write_vector_partition(
    db_id: int,
    rows: Iterable,
    run_id: str,
    adapter_factory: VectorIndexWriterFactory,
    max_writes_per_second: float | None,
    id_col: str,
    vector_col: str,
    payload_cols: list[str] | None,
    write_vector_shard_core: Callable[..., RequiredShardMeta],
    metrics_collector: Any | None,
    s3_prefix: str,
    shard_prefix: str,
    db_path_template: str,
    local_root: str,
    index_config: Any,
    batch_size: int,
) -> list:
    """Write vectors for a single partition."""
    from shardyfusion.vector._distributed import coerce_vector_value as _coerce_vec

    rows_list = list(rows)
    if not rows_list:
        return []

    ops_limiter: TokenBucket | None = None
    if max_writes_per_second is not None:
        ops_limiter = TokenBucket(
            max_writes_per_second,
            metrics_collector=metrics_collector,
            limiter_type="ops",
        )

    groups: dict = defaultdict(list)
    for row in rows_list:
        vector_db_id = row[VECTOR_DB_ID_COL]
        groups[vector_db_id].append(row)

    results = []
    for vid, group in groups.items():
        rows_iter = (
            (
                row[id_col],
                _coerce_vec(row[vector_col]),
                {col: row[col] for col in payload_cols} if payload_cols else None,
            )
            for row in group
        )
        result = write_vector_shard_core(
            db_id=vid,
            rows=rows_iter,
            run_id=run_id,
            s3_prefix=s3_prefix,
            shard_prefix=shard_prefix,
            db_path_template=db_path_template,
            local_root=local_root,
            index_config=index_config,
            adapter_factory=adapter_factory,
            batch_size=batch_size,
            ops_limiter=ops_limiter,
            metrics_collector=metrics_collector,
        )
        results.append(result)

    return results
