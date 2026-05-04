"""Dask-based sharded writer (no Spark/Java dependency)."""

import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import dask.dataframe as dd
import pandas as pd

from shardyfusion._adapter import (
    DbAdapterFactory,
)
from shardyfusion._shard_writer import (
    results_pdf_to_attempts,
    write_shard_with_retry,
)
from shardyfusion._writer_core import (
    VectorColumnMapping,
    assemble_build_result,
    cleanup_losers,
    inject_sqlite_btreemeta_manifest_field,
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
    DaskWriteOptions,
    HashShardedWriteConfig,
    validate_configs,
    validate_shard_id_col_no_collision,
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
    HashShardingSpec,
    ShardHashAlgorithm,
    ShardingSpec,
)
from shardyfusion.writer._accumulators import KvAccumulator, UnifiedAccumulator
from shardyfusion.writer._runtime import RetryingPartitionWriteRuntime

from .sharding import (
    add_db_id_column_cel,
    add_db_id_column_hash,
)

_logger = get_logger(__name__)


# Meta schema for result DataFrames returned by partition writers.
_RESULT_META = pd.DataFrame(
    columns=[
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
    ],
).astype(
    {
        "db_id": "int64",
        "db_url": "object",
        "attempt": "int64",
        "row_count": "int64",
        "min_key": "object",
        "max_key": "object",
        "checkpoint_id": "object",
        "writer_info": "object",
        "db_bytes": "int64",
        "all_attempt_urls": "object",
    }
)


def write_hash_sharded(
    ddf: dd.DataFrame,
    config: HashShardedWriteConfig,
    input: ColumnWriteInput,
    options: DaskWriteOptions | None = None,
) -> BuildResult:
    """Write a Dask DataFrame into N independent sharded databases using HASH routing."""
    from shardyfusion._writer_core import resolve_num_dbs

    options = options or DaskWriteOptions()
    validate_configs(config, input, options)
    validate_shard_id_col_no_collision(config, set(ddf.columns))
    num_dbs = resolve_num_dbs(config, lambda: len(ddf))
    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    return _write_hash_sharded(
        ddf=ddf,
        config=config,
        num_dbs=num_dbs,
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
    ddf: dd.DataFrame,
    config: CelShardedWriteConfig,
    input: ColumnWriteInput,
    options: DaskWriteOptions | None = None,
) -> BuildResult:
    """Write a Dask DataFrame into N independent sharded databases using CEL routing."""
    options = options or DaskWriteOptions()
    validate_configs(config, input, options)
    validate_shard_id_col_no_collision(config, set(ddf.columns))
    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    return _write_cel_sharded(
        ddf=ddf,
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
    ddf: dd.DataFrame,
    config: HashShardedWriteConfig,
    num_dbs: int,
    run_id: str,
    started: float,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool,
    verify_routing: bool,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None,
    vector_columns: VectorColumnMapping | None,
) -> BuildResult:
    """Hash-specific sharded write pipeline for Dask."""
    from shardyfusion.logging import LogContext

    mc = config.metrics_collector

    with (
        LogContext(run_id=run_id),
        RunRecordLifecycle.start(
            config=config,
            run_id=run_id,
            writer_type="dask",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=num_dbs,
            s3_prefix=config.s3_prefix,
            strategy="hash",
            key_encoding=config.key_encoding,
            writer_type="dask",
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

        # --- Phase 1: Sharding ---
        shard_started = time.perf_counter()
        shard_id_col = config.shard_id_col
        sharding = HashShardingSpec(
            hash_algorithm=ShardHashAlgorithm.XXH3_64,
            max_keys_per_shard=config.max_keys_per_shard,
        )
        ddf_with_id = add_db_id_column_hash(
            ddf,
            key_col=key_col,
            num_dbs=num_dbs,
            hash_algorithm=sharding.hash_algorithm,
            shard_id_col=shard_id_col,
        )

        if verify_routing and num_dbs > 0:
            _verify_hash_routing_agreement(
                ddf_with_id,
                key_col=key_col,
                num_dbs=num_dbs,
                hash_algorithm=sharding.hash_algorithm,
                shard_id_col=shard_id_col,
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
        runtime = RetryingPartitionWriteRuntime.from_public_config(
            config=config,
            input=ColumnWriteInput(
                key_col=key_col,
                value_spec=value_spec,
            ),
            run_id=run_id,
            started=started,
            sort_within_partitions=sort_within_partitions,
            vector_fn=distributed_vector_fn,
        )

        write_started = time.perf_counter()
        ddf_shuffled = ddf_with_id.shuffle(on=shard_id_col, npartitions=num_dbs)
        ddf_results = ddf_shuffled.map_partitions(
            _write_partition,
            runtime=runtime,
            meta=_RESULT_META,
        )
        results_pdf = ddf_results.compute()

        attempts = results_pdf_to_attempts(results_pdf)
        write_duration_ms = int((time.perf_counter() - write_started) * 1000)

        # --- Phase 3: Publish ---
        winners, num_attempts, all_attempt_urls = select_winners(
            attempts, num_dbs=num_dbs
        )

        return _finalize_sharded_write(
            config=config,
            run_id=run_id,
            started=started,
            resolved_sharding=sharding,
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
    ddf: dd.DataFrame,
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
    """CEL-specific sharded write pipeline for Dask."""
    from shardyfusion._writer_core import discover_cel_num_dbs
    from shardyfusion.logging import LogContext

    mc = config.metrics_collector

    with (
        LogContext(run_id=run_id),
        RunRecordLifecycle.start(
            config=config,
            run_id=run_id,
            writer_type="dask",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=None,
            s3_prefix=config.s3_prefix,
            strategy="cel",
            key_encoding=config.key_encoding,
            writer_type="dask",
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

        # --- Phase 1: Sharding ---
        shard_started = time.perf_counter()
        shard_id_col = config.shard_id_col
        ddf_with_id, resolved_sharding = add_db_id_column_cel(
            ddf,
            cel_expr=config.cel_expr,
            cel_columns=config.cel_columns,
            routing_values=config.routing_values,
            infer_routing_values_from_data=config.infer_routing_values_from_data,
            shard_id_col=shard_id_col,
        )

        if resolved_sharding.routing_values is not None:
            num_dbs = resolve_cel_num_dbs(resolved_sharding)
        else:
            distinct_ids = set(ddf_with_id[shard_id_col].unique().compute().tolist())
            num_dbs = discover_cel_num_dbs(distinct_ids)

        if verify_routing and num_dbs is not None and num_dbs > 0:
            _verify_cel_routing_agreement(
                ddf_with_id,
                key_col=key_col,
                num_dbs=num_dbs,
                cel_expr=resolved_sharding.cel_expr,
                cel_columns=resolved_sharding.cel_columns,
                routing_values=resolved_sharding.routing_values,
                shard_id_col=shard_id_col,
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
        runtime = RetryingPartitionWriteRuntime.from_public_config(
            config=config,
            input=ColumnWriteInput(
                key_col=key_col,
                value_spec=value_spec,
            ),
            run_id=run_id,
            started=started,
            sort_within_partitions=sort_within_partitions,
            vector_fn=distributed_vector_fn,
        )

        write_started = time.perf_counter()
        assert num_dbs is not None, "num_dbs must be resolved before shuffle"
        ddf_shuffled = ddf_with_id.shuffle(on=shard_id_col, npartitions=num_dbs)
        ddf_results = ddf_shuffled.map_partitions(
            _write_partition,
            runtime=runtime,
            meta=_RESULT_META,
        )
        results_pdf = ddf_results.compute()

        attempts = results_pdf_to_attempts(results_pdf)
        write_duration_ms = int((time.perf_counter() - write_started) * 1000)

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
    runtime: RetryingPartitionWriteRuntime,
    key_col: str,
    num_dbs: int,
    mc: MetricsCollector | None,
    run_record: Any,
) -> BuildResult:
    """Publish manifest, cleanup losers, and assemble BuildResult."""
    rows_written = sum(w.row_count for w in winners)
    log_event(
        "shard_writes_completed",
        logger=_logger,
        num_winners=len(winners),
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

    if config.vector_spec is not None:
        inject_vector_manifest_fields(config, runtime.adapter_factory)
    inject_sqlite_btreemeta_manifest_field(config, runtime.adapter_factory)

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


def _verify_hash_routing_agreement(
    ddf_with_id: dd.DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    hash_algorithm: ShardHashAlgorithm,
    shard_id_col: str = DB_ID_COL,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify db_id column matches Python hash routing."""

    sampled = ddf_with_id[[key_col, shard_id_col]].head(sample_size, npartitions=-1)
    if sampled.empty:
        return

    mismatches: list[tuple[object, int, int]] = []
    for _, row in sampled.iterrows():
        key = row[key_col]
        if hasattr(key, "item"):
            key = key.item()
        computed_db_id = int(row[shard_id_col])
        expected_db_id = route_hash(
            key,
            num_dbs=num_dbs,
            hash_algorithm=hash_algorithm,
        )
        if expected_db_id != computed_db_id:
            mismatches.append((key, computed_db_id, expected_db_id))

    if mismatches:
        details = "; ".join(
            f"key={k}, dask={d}, python={p}" for k, d, p in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Dask/Python routing mismatch in {len(mismatches)}/{len(sampled)} "
            f"sampled rows. First mismatches: {details}"
        )


def _verify_cel_routing_agreement(
    ddf_with_id: dd.DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    cel_expr: str,
    cel_columns: dict[str, str],
    routing_values: list[int | str | bytes] | None,
    shard_id_col: str = DB_ID_COL,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify db_id column matches Python CEL routing."""

    # Multi-column CEL: include context columns in sample
    sample_cols = [key_col, shard_id_col]
    if set(cel_columns.keys()) != {"key"}:
        sample_cols.extend(c for c in cel_columns if c not in (key_col, shard_id_col))

    sampled = ddf_with_id[sample_cols].head(sample_size, npartitions=-1)
    if sampled.empty:
        return

    mismatches: list[tuple[object, int, int]] = []
    for _, row in sampled.iterrows():
        key = row[key_col]
        if hasattr(key, "item"):
            key = key.item()
        computed_db_id = int(row[shard_id_col])

        routing_context: dict[str, object] | None = None
        if set(cel_columns.keys()) != {"key"}:
            routing_context = {
                col: row[col].item() if hasattr(row[col], "item") else row[col]
                for col in cel_columns
            }

        expected_db_id = route_cel(
            key,
            cel_expr=cel_expr,
            cel_columns=cel_columns,
            routing_values=routing_values,
            routing_context=routing_context,
        )
        if expected_db_id != computed_db_id:
            mismatches.append((key, computed_db_id, expected_db_id))

    if mismatches:
        details = "; ".join(
            f"key={k}, dask={d}, python={p}" for k, d, p in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Dask/Python routing mismatch in {len(mismatches)}/{len(sampled)} "
            f"sampled rows. First mismatches: {details}"
        )


def _write_partition(
    pdf: pd.DataFrame,
    runtime: RetryingPartitionWriteRuntime,
) -> pd.DataFrame:
    """Write all db_id groups within one Dask partition."""

    if pdf.empty:
        return _RESULT_META.iloc[:0].copy()

    results: list[dict[str, object]] = []

    factory: DbAdapterFactory = runtime.adapter_factory
    accumulator_factory = (
        UnifiedAccumulator if runtime.vector_fn is not None else KvAccumulator
    )

    shard_id_col = runtime.shard_id_col
    for db_id, group_pdf in pdf.groupby(shard_id_col):
        if runtime.sort_within_partitions:
            group_pdf = group_pdf.sort_values(runtime.key_col)

        # Drop the internal shard_id column so it doesn't leak into stored data
        if shard_id_col in group_pdf.columns:
            group_pdf = group_pdf.drop(columns=[shard_id_col])

        vec_fn = runtime.vector_fn
        attempt_result = write_shard_with_retry(
            db_id=int(db_id),  # type: ignore[invalid-argument-type]
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
        return _RESULT_META.iloc[:0].copy()

    return pd.DataFrame(results)
