"""Ray Data-based sharded writer (no Spark/Java dependency)."""

import threading
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pandas as pd
import ray.data
from ray.data import DataContext

from shardyfusion._shard_writer import (
    results_pdf_to_attempts,
    write_shard_with_retry,
)
from shardyfusion._writer_core import (
    VectorColumnMapping,
    assemble_build_result,
    cleanup_losers,
    inject_vector_manifest_fields,
    publish_to_store,
    resolve_cel_num_dbs,
    resolve_distributed_vector_fn,
    route_cel,
    route_hash,
    select_winners,
)
from shardyfusion.config import (
    BaseShardedWriteConfig,
    CelShardedWriteConfig,
    ColumnWriteInput,
    HashShardedWriteConfig,
    RayWriteOptions,
)
from shardyfusion.errors import ShardAssignmentError
from shardyfusion.logging import (
    get_logger,
    log_event,
)
from shardyfusion.manifest import (
    BuildResult,
    RequiredShardMeta,
)
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.run_registry import RunRecordLifecycle
from shardyfusion.serde import ValueSpec
from shardyfusion.sharding_types import (
    DB_ID_COL,
    CelShardingSpec,
    HashShardingSpec,
    ShardHashAlgorithm,
    ShardingSpec,
)
from shardyfusion.slatedb_adapter import (
    DbAdapterFactory,
)
from shardyfusion.writer._accumulators import KvAccumulator, UnifiedAccumulator
from shardyfusion.writer._runtime import PartitionWriteRuntime

from .sharding import (
    add_db_id_column_cel,
    add_db_id_column_hash,
)

_logger = get_logger(__name__)

# Import ShuffleStrategy for save/restore of the process-global shuffle setting.
# Deferred to avoid import errors on older Ray versions.
try:
    from ray.data.context import ShuffleStrategy as _ShuffleStrategyEnum

    _HASH_SHUFFLE_STRATEGY = _ShuffleStrategyEnum.HASH_SHUFFLE
except ImportError:
    _HASH_SHUFFLE_STRATEGY = None  # type: ignore[assignment]

# Guards the process-global DataContext.shuffle_strategy swap against
# concurrent write_sharded() calls in the same Ray driver.

_SHUFFLE_STRATEGY_LOCK = threading.Lock()


# Columns for result DataFrames returned by partition writers.
_RESULT_COLUMNS = [
    "db_id",
    "db_url",
    "attempt",
    "row_count",
    "min_key",
    "max_key",
    "checkpoint_id",
    "writer_info",
    "db_bytes",
    "all_attempt_urls",
]


def write_hash_sharded(
    ds: ray.data.Dataset,
    config: HashShardedWriteConfig,
    input: ColumnWriteInput,
    options: RayWriteOptions | None = None,
) -> BuildResult:
    """Write a Ray Dataset into N independent sharded databases using HASH routing."""
    options = options or RayWriteOptions()
    config.validate()
    input.validate()
    options.validate()
    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex
    num_dbs = _resolve_num_dbs_before_sharding(ds, config)

    return _write_hash_sharded(
        ds=ds,
        config=config,
        num_dbs=num_dbs,
        run_id=run_id,
        started=started,
        key_col=input.key_col,
        value_spec=input.value_spec,
        sort_within_partitions=options.sort_within_partitions,
        max_writes_per_second=config.rate_limits.max_writes_per_second,
        max_write_bytes_per_second=config.rate_limits.max_write_bytes_per_second,
        verify_routing=options.verify_routing,
        vector_fn=input.vector_fn,
        vector_columns=input.vector,
    )


def write_cel_sharded(
    ds: ray.data.Dataset,
    config: CelShardedWriteConfig,
    input: ColumnWriteInput,
    options: RayWriteOptions | None = None,
) -> BuildResult:
    """Write a Ray Dataset into N independent sharded databases using CEL routing."""
    options = options or RayWriteOptions()
    config.validate()
    input.validate()
    options.validate()
    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    return _write_cel_sharded(
        ds=ds,
        config=config,
        run_id=run_id,
        started=started,
        key_col=input.key_col,
        value_spec=input.value_spec,
        sort_within_partitions=options.sort_within_partitions,
        max_writes_per_second=config.rate_limits.max_writes_per_second,
        max_write_bytes_per_second=config.rate_limits.max_write_bytes_per_second,
        verify_routing=options.verify_routing,
        vector_fn=input.vector_fn,
        vector_columns=input.vector,
    )


def _write_hash_sharded(
    *,
    ds: ray.data.Dataset,
    config: HashShardedWriteConfig,
    num_dbs: int,
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
    from shardyfusion.logging import LogContext

    resolved_sharding = HashShardingSpec(
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
        max_keys_per_shard=config.max_keys_per_shard,
    )

    has_vectors = (
        vector_fn is not None
        or vector_columns is not None
        or config.vector_spec is not None
    )
    distributed_vector_fn = None
    if has_vectors:
        distributed_vector_fn = resolve_distributed_vector_fn(
            config=config,
            key_col=key_col,
            vector_fn=vector_fn,
            vector_columns=vector_columns,
        )

    mc = config.metrics_collector

    with (
        LogContext(run_id=run_id),
        RunRecordLifecycle.start(
            config=config,
            run_id=run_id,
            writer_type="ray",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=num_dbs,
            s3_prefix=config.s3_prefix,
            strategy="hash",
            key_encoding=config.key_encoding,
            writer_type="ray",
        )
        if mc is not None:
            mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})

        # --- Phase 1: Sharding ---
        shard_started = time.perf_counter()

        ds_with_id = add_db_id_column_hash(
            ds,
            key_col=key_col,
            num_dbs=num_dbs,
            hash_algorithm=resolved_sharding.hash_algorithm,
        )

        if verify_routing and num_dbs > 0:
            _verify_hash_routing_agreement(
                ds_with_id,
                key_col=key_col,
                num_dbs=num_dbs,
                hash_algorithm=resolved_sharding.hash_algorithm,
            )

        shard_duration_ms = int((time.perf_counter() - shard_started) * 1000)
        log_event(
            "sharding_completed",
            logger=_logger,
            duration_ms=shard_duration_ms,
        )
        if mc is not None:
            mc.emit(
                MetricEvent.SHARDING_COMPLETED,
                {
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "duration_ms": shard_duration_ms,
                },
            )

        # --- Phase 2: Write ---
        runtime = _build_partition_write_runtime(
            config=config,
            run_id=run_id,
            key_col=key_col,
            value_spec=value_spec,
            max_writes_per_second=max_writes_per_second,
            max_write_bytes_per_second=max_write_bytes_per_second,
            sort_within_partitions=sort_within_partitions,
            started=started,
            vector_fn=distributed_vector_fn,
        )

        write_started = time.perf_counter()

        with _SHUFFLE_STRATEGY_LOCK:
            ctx = DataContext.get_current()
            prev_strategy = ctx.shuffle_strategy
            try:
                ctx.shuffle_strategy = _HASH_SHUFFLE_STRATEGY  # type: ignore[assignment]
                ds_shuffled = ds_with_id.repartition(
                    num_dbs, shuffle=True, keys=[DB_ID_COL]
                )
            finally:
                ctx.shuffle_strategy = prev_strategy

        ds_results = ds_shuffled.map_batches(
            _write_partition,  # type: ignore[arg-type]
            batch_format="pandas",
            fn_kwargs={"runtime": runtime},
            zero_copy_batch=False,
        )
        results_pdf = ds_results.to_pandas()

        attempts = results_pdf_to_attempts(results_pdf)
        write_duration_ms = int((time.perf_counter() - write_started) * 1000)

        rows_written = sum(a.row_count for a in attempts)
        log_event(
            "shard_writes_completed",
            logger=_logger,
            num_winners=len(attempts),
            rows_written=rows_written,
            duration_ms=write_duration_ms,
        )
        if mc is not None:
            mc.emit(
                MetricEvent.SHARD_WRITES_COMPLETED,
                {
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "duration_ms": write_duration_ms,
                    "rows_written": rows_written,
                },
            )

        # --- Phase 3: Publish ---
        winners, num_attempts, all_attempt_urls = select_winners(
            attempts, num_dbs=num_dbs
        )

        return _finalize_sharded_write(
            config=config,
            run_id=run_id,
            started=started,
            resolved_sharding=resolved_sharding,
            winners=winners,
            num_attempts=num_attempts,
            all_attempt_urls=all_attempt_urls,
            shard_duration_ms=shard_duration_ms,
            write_duration_ms=write_duration_ms,
            runtime=runtime,
            key_col=key_col,
            num_dbs=num_dbs,
            mc=mc,
            run_record=run_record,
        )


def _write_cel_sharded(
    *,
    ds: ray.data.Dataset,
    config: CelShardedWriteConfig,
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
    from shardyfusion._writer_core import discover_cel_num_dbs
    from shardyfusion.logging import LogContext

    resolved_sharding = CelShardingSpec(
        cel_expr=config.cel_expr,
        cel_columns=config.cel_columns,
        routing_values=config.routing_values,
    )

    has_vectors = (
        vector_fn is not None
        or vector_columns is not None
        or config.vector_spec is not None
    )
    distributed_vector_fn = None
    if has_vectors:
        distributed_vector_fn = resolve_distributed_vector_fn(
            config=config,
            key_col=key_col,
            vector_fn=vector_fn,
            vector_columns=vector_columns,
        )

    mc = config.metrics_collector

    with (
        LogContext(run_id=run_id),
        RunRecordLifecycle.start(
            config=config,
            run_id=run_id,
            writer_type="ray",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=None,
            s3_prefix=config.s3_prefix,
            strategy="cel",
            key_encoding=config.key_encoding,
            writer_type="ray",
        )
        if mc is not None:
            mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})

        # --- Phase 1: Sharding ---
        shard_started = time.perf_counter()

        ds_with_id, resolved_sharding = add_db_id_column_cel(
            ds,
            cel_expr=config.cel_expr,
            cel_columns=config.cel_columns,
            routing_values=config.routing_values,
            infer_routing_values_from_data=config.infer_routing_values_from_data,
        )
        assert isinstance(resolved_sharding, CelShardingSpec)

        if resolved_sharding.routing_values is not None:
            num_dbs = resolve_cel_num_dbs(resolved_sharding)
        else:
            distinct_ids = set(ds_with_id.unique(DB_ID_COL))
            num_dbs = discover_cel_num_dbs(distinct_ids)

        if verify_routing and num_dbs > 0:
            _verify_cel_routing_agreement(
                ds_with_id,
                key_col=key_col,
                num_dbs=num_dbs,
                cel_expr=resolved_sharding.cel_expr,
                cel_columns=resolved_sharding.cel_columns,
                routing_values=resolved_sharding.routing_values,
            )

        shard_duration_ms = int((time.perf_counter() - shard_started) * 1000)
        log_event(
            "sharding_completed",
            logger=_logger,
            duration_ms=shard_duration_ms,
        )
        if mc is not None:
            mc.emit(
                MetricEvent.SHARDING_COMPLETED,
                {
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "duration_ms": shard_duration_ms,
                },
            )

        # --- Phase 2: Write ---
        runtime = _build_partition_write_runtime(
            config=config,
            run_id=run_id,
            key_col=key_col,
            value_spec=value_spec,
            max_writes_per_second=max_writes_per_second,
            max_write_bytes_per_second=max_write_bytes_per_second,
            sort_within_partitions=sort_within_partitions,
            started=started,
            vector_fn=distributed_vector_fn,
        )

        write_started = time.perf_counter()

        with _SHUFFLE_STRATEGY_LOCK:
            ctx = DataContext.get_current()
            prev_strategy = ctx.shuffle_strategy
            try:
                ctx.shuffle_strategy = _HASH_SHUFFLE_STRATEGY  # type: ignore[assignment]
                ds_shuffled = ds_with_id.repartition(
                    num_dbs, shuffle=True, keys=[DB_ID_COL]
                )
            finally:
                ctx.shuffle_strategy = prev_strategy

        ds_results = ds_shuffled.map_batches(
            _write_partition,  # type: ignore[arg-type]
            batch_format="pandas",
            fn_kwargs={"runtime": runtime},
            zero_copy_batch=False,
        )
        results_pdf = ds_results.to_pandas()

        attempts = results_pdf_to_attempts(results_pdf)
        write_duration_ms = int((time.perf_counter() - write_started) * 1000)

        rows_written = sum(a.row_count for a in attempts)
        log_event(
            "shard_writes_completed",
            logger=_logger,
            num_winners=len(attempts),
            rows_written=rows_written,
            duration_ms=write_duration_ms,
        )
        if mc is not None:
            mc.emit(
                MetricEvent.SHARD_WRITES_COMPLETED,
                {
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                    "duration_ms": write_duration_ms,
                    "rows_written": rows_written,
                },
            )

        # --- Phase 3: Publish ---
        winners, num_attempts, all_attempt_urls = select_winners(
            attempts, num_dbs=num_dbs
        )

        return _finalize_sharded_write(
            config=config,
            run_id=run_id,
            started=started,
            resolved_sharding=resolved_sharding,
            winners=winners,
            num_attempts=num_attempts,
            all_attempt_urls=all_attempt_urls,
            shard_duration_ms=shard_duration_ms,
            write_duration_ms=write_duration_ms,
            runtime=runtime,
            key_col=key_col,
            num_dbs=num_dbs,
            mc=mc,
            run_record=run_record,
        )


def _finalize_sharded_write(
    *,
    config: BaseShardedWriteConfig,
    run_id: str,
    started: float,
    resolved_sharding: ShardingSpec,
    winners: list[RequiredShardMeta],
    num_attempts: int,
    all_attempt_urls: list[str],
    shard_duration_ms: int,
    write_duration_ms: int,
    runtime: PartitionWriteRuntime,
    key_col: str,
    num_dbs: int,
    mc: MetricsCollector | None,
    run_record: Any,
) -> BuildResult:
    """Publish manifest, cleanup losers, and assemble BuildResult."""
    if config.vector_spec is not None:
        inject_vector_manifest_fields(config, runtime.adapter_factory)
    manifest_started = time.perf_counter()
    manifest_ref = publish_to_store(
        config=config,
        run_id=run_id,
        resolved_sharding=resolved_sharding,
        winners=winners,
        key_col=key_col,
        started=started,
        num_dbs=num_dbs,
    )
    run_record.set_manifest_ref(manifest_ref)
    manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

    cleanup_losers(all_attempt_urls, winners, metrics_collector=mc)

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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_num_dbs_before_sharding(
    ds: ray.data.Dataset, config: HashShardedWriteConfig
) -> int:
    """Resolve num_dbs that can be determined before add_db_id_column."""
    from shardyfusion._writer_core import resolve_num_dbs

    return resolve_num_dbs(config, ds.count)


def _verify_hash_routing_agreement(
    ds_with_id: ray.data.Dataset,
    *,
    key_col: str,
    num_dbs: int,
    hash_algorithm: ShardHashAlgorithm,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify Ray-computed db_id matches Python hash routing."""

    sampled = ds_with_id.take(sample_size)
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        key = row[key_col]
        # Convert numpy scalars to Python types for routing compatibility
        if hasattr(key, "item"):
            key = key.item()
        ray_db_id = int(row[DB_ID_COL])
        python_db_id = route_hash(
            key,
            num_dbs=num_dbs,
            hash_algorithm=hash_algorithm,
        )
        if python_db_id != ray_db_id:
            mismatches.append((key, ray_db_id, python_db_id))

    if mismatches:
        details = "; ".join(
            f"key={k}, ray={r}, python={p}" for k, r, p in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Ray/Python routing mismatch in {len(mismatches)}/{len(sampled)} "
            f"sampled rows. First mismatches: {details}"
        )


def _verify_cel_routing_agreement(
    ds_with_id: ray.data.Dataset,
    *,
    key_col: str,
    num_dbs: int,
    cel_expr: str,
    cel_columns: dict[str, str],
    routing_values: list[int | str | bytes] | None,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify Ray-computed db_id matches Python CEL routing."""

    # Multi-column CEL: can't verify without routing_context
    if cel_columns and set(cel_columns.keys()) != {"key"}:
        return

    sampled = ds_with_id.take(sample_size)
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        key = row[key_col]
        # Convert numpy scalars to Python types for routing compatibility
        if hasattr(key, "item"):
            key = key.item()
        ray_db_id = int(row[DB_ID_COL])
        python_db_id = route_cel(
            key,
            cel_expr=cel_expr,
            cel_columns=cel_columns,
            routing_values=routing_values,
            routing_context=None,
        )
        if python_db_id != ray_db_id:
            mismatches.append((key, ray_db_id, python_db_id))

    if mismatches:
        details = "; ".join(
            f"key={k}, ray={r}, python={p}" for k, r, p in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Ray/Python routing mismatch in {len(mismatches)}/{len(sampled)} "
            f"sampled rows. First mismatches: {details}"
        )


def _build_partition_write_runtime(
    *,
    config: BaseShardedWriteConfig,
    run_id: str,
    key_col: str,
    value_spec: ValueSpec,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    sort_within_partitions: bool,
    started: float,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None,
) -> PartitionWriteRuntime:
    """Construct picklable runtime config for Ray partition writers."""
    runtime = PartitionWriteRuntime.from_public_config(
        config=config,
        input=ColumnWriteInput(
            key_col=key_col,
            value_spec=value_spec,
        ),
        run_id=run_id,
        started=started,
        sort_within_partitions=sort_within_partitions,
        vector_fn=vector_fn,
    )
    runtime.max_writes_per_second = max_writes_per_second
    runtime.max_write_bytes_per_second = max_write_bytes_per_second
    return runtime


def _write_partition(
    pdf: pd.DataFrame,
    runtime: PartitionWriteRuntime,
) -> pd.DataFrame:
    """Write all db_id groups within one Ray partition."""

    if pdf.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    results: list[dict[str, object]] = []

    factory: DbAdapterFactory = runtime.adapter_factory
    accumulator_factory = (
        UnifiedAccumulator if runtime.vector_fn is not None else KvAccumulator
    )

    for db_id, group_pdf in pdf.groupby(DB_ID_COL):
        if runtime.sort_within_partitions:
            group_pdf = group_pdf.sort_values(runtime.key_col)

        vec_fn = runtime.vector_fn
        attempt_result = write_shard_with_retry(
            db_id=int(db_id),  # type: ignore[arg-type]
            rows_fn=lambda pdf=group_pdf, vec_fn=vec_fn: (
                (
                    row[runtime.key_col].item()
                    if hasattr(row[runtime.key_col], "item")
                    else row[runtime.key_col],
                    runtime.value_spec.encode(row),
                    vec_fn(row) if vec_fn is not None else None,
                )
                for _, row in pdf.iterrows()
            ),
            run_id=runtime.run_id,
            s3_prefix=runtime.s3_prefix,
            shard_prefix=runtime.shard_prefix,
            db_path_template=runtime.db_path_template,
            local_root=runtime.local_root,
            key_encoder=runtime.key_encoder,
            batch_size=runtime.batch_size,
            factory=factory,
            max_writes_per_second=runtime.max_writes_per_second,
            max_write_bytes_per_second=runtime.max_write_bytes_per_second,
            metrics_collector=runtime.metrics_collector,
            started=runtime.started,
            retry_config=runtime.shard_retry,
            accumulator_factory=accumulator_factory,
        )
        results.append(
            {
                "db_id": attempt_result.db_id,
                "db_url": attempt_result.db_url,
                "attempt": attempt_result.attempt,
                "row_count": attempt_result.row_count,
                "min_key": attempt_result.min_key,
                "max_key": attempt_result.max_key,
                "checkpoint_id": attempt_result.checkpoint_id,
                "writer_info": attempt_result.writer_info,
                "db_bytes": attempt_result.db_bytes,
                "all_attempt_urls": attempt_result.all_attempt_urls,
            }
        )

    if not results:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    return pd.DataFrame(results)
