"""Dask-based sharded writer (no Spark/Java dependency)."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import dask.dataframe as dd
import pandas as pd

from shardyfusion._shard_writer import (
    results_pdf_to_attempts,
    write_shard_with_retry,
)
from shardyfusion._writer_core import (
    VectorColumnMapping,
    _normalize_vector_id,
    assemble_build_result,
    cleanup_losers,
    inject_vector_manifest_fields,
    publish_to_store,
    resolve_cel_num_dbs,
    resolve_distributed_vector_fn,
    route_cel,
    route_hash,
    select_winners,
    wrap_factory_for_vector,
)
from shardyfusion.config import CelWriteConfig, HashWriteConfig, WriteConfig
from shardyfusion.credentials import CredentialProvider
from shardyfusion.errors import ShardAssignmentError
from shardyfusion.logging import (
    get_logger,
    log_event,
)
from shardyfusion.manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    RequiredShardMeta,
)
from shardyfusion.metrics import MetricEvent, MetricsCollector
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
from shardyfusion.type_defs import RetryConfig
from shardyfusion.writer._accumulators import KvAccumulator, UnifiedAccumulator

from .sharding import (
    VECTOR_DB_ID_COL,
    _stack_vector_values,
    add_db_id_column,
)

_logger = get_logger(__name__)


@dataclass(slots=True)
class _PartitionWriteRuntime:
    """Picklable runtime config for Dask partition writers."""

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
    sort_within_partitions: bool = False
    metrics_collector: MetricsCollector | None = None  # must be picklable
    started: float = 0.0
    shard_retry: RetryConfig | None = None
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None = (
        None
    )


@dataclass(slots=True)
class _VectorPartitionWriteRuntime:
    """Picklable runtime config for Dask vector partition writers."""

    run_id: str
    s3_prefix: str
    shard_prefix: str
    db_path_template: str
    local_root: str
    index_config: Any
    adapter_factory: Any
    batch_size: int
    payload_cols: list[str] | None = None


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


def _vector_result_row(result: RequiredShardMeta) -> dict[str, object]:
    return {
        "db_id": result.db_id,
        "db_url": result.db_url,
        "attempt": result.attempt,
        "row_count": result.row_count,
        "min_key": result.min_key,
        "max_key": result.max_key,
        "checkpoint_id": result.checkpoint_id,
        "writer_info": result.writer_info,
        "db_bytes": result.db_bytes,
        "all_attempt_urls": (),
    }


def write_sharded_by_hash(
    ddf: dd.DataFrame,
    config: HashWriteConfig,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool = False,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    verify_routing: bool = True,
    vector_fn: (
        Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None
    ) = None,
    vector_columns: VectorColumnMapping | None = None,
) -> BuildResult:
    """Write a Dask DataFrame into N independent sharded databases using HASH routing."""
    from shardyfusion._writer_core import resolve_num_dbs

    num_dbs = resolve_num_dbs(config, lambda: len(ddf))
    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    return _write_hash_sharded(
        ddf=ddf,
        config=config,
        num_dbs=num_dbs,
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
    ddf: dd.DataFrame,
    config: CelWriteConfig,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool = False,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    verify_routing: bool = True,
    vector_fn: (
        Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None
    ) = None,
    vector_columns: VectorColumnMapping | None = None,
) -> BuildResult:
    """Write a Dask DataFrame into N independent sharded databases using CEL routing."""
    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    return _write_cel_sharded(
        ddf=ddf,
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
    ddf: dd.DataFrame,
    config: HashWriteConfig,
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
        sharding = HashShardingSpec(
            hash_algorithm=ShardHashAlgorithm.XXH3_64,
            max_keys_per_shard=config.max_keys_per_shard,
        )
        ddf_with_id, resolved_sharding = add_db_id_column(
            ddf,
            key_col=key_col,
            num_dbs=num_dbs,
            sharding=sharding,
            key_encoding=config.key_encoding,
        )

        if verify_routing and num_dbs > 0:
            _verify_routing_agreement(
                ddf_with_id,
                key_col=key_col,
                num_dbs=num_dbs,
                resolved_sharding=resolved_sharding,
                key_encoding=config.key_encoding,
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
        ddf_shuffled = ddf_with_id.shuffle(on=DB_ID_COL, npartitions=num_dbs)
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


def _write_cel_sharded(
    *,
    ddf: dd.DataFrame,
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
        sharding = CelShardingSpec(
            cel_expr=config.cel_expr,
            cel_columns=config.cel_columns,
            routing_values=config.routing_values,
            infer_routing_values_from_data=config.infer_routing_values_from_data,
        )
        ddf_with_id, resolved_sharding = add_db_id_column(
            ddf,
            key_col=key_col,
            num_dbs=None,
            sharding=sharding,
            key_encoding=config.key_encoding,
        )
        assert isinstance(resolved_sharding, CelShardingSpec)

        if resolved_sharding.routing_values is not None:
            num_dbs = resolve_cel_num_dbs(resolved_sharding)
        else:
            distinct_ids = set(ddf_with_id[DB_ID_COL].unique().compute().tolist())
            num_dbs = discover_cel_num_dbs(distinct_ids)

        if verify_routing and num_dbs is not None and num_dbs > 0:
            _verify_routing_agreement(
                ddf_with_id,
                key_col=key_col,
                num_dbs=num_dbs,
                resolved_sharding=resolved_sharding,
                key_encoding=config.key_encoding,
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
        assert num_dbs is not None, "num_dbs must be resolved before shuffle"
        ddf_shuffled = ddf_with_id.shuffle(on=DB_ID_COL, npartitions=num_dbs)
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
    config: WriteConfig,
    run_id: str,
    started: float,
    resolved_sharding: ShardingSpec,
    winners: list[RequiredShardMeta],
    num_attempts: int,
    all_attempt_urls: list[str],
    shard_duration_ms: int,
    write_duration_ms: int,
    runtime: _PartitionWriteRuntime,
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


def _verify_routing_agreement(
    ddf_with_id: dd.DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    resolved_sharding: ShardingSpec,
    key_encoding: KeyEncoding,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify db_id column matches Python routing."""

    # For CEL with non-key columns, include those columns in the sample.
    cel_columns = (
        resolved_sharding.cel_columns
        if isinstance(resolved_sharding, CelShardingSpec)
        else None
    )
    sample_cols = [key_col, DB_ID_COL]
    if cel_columns is not None:
        sample_cols.extend(c for c in cel_columns if c not in (key_col, DB_ID_COL))

    sampled = ddf_with_id[sample_cols].head(sample_size, npartitions=-1)
    if sampled.empty:
        return

    mismatches: list[tuple[object, int, int]] = []
    for _, row in sampled.iterrows():
        key = row[key_col]
        # Convert numpy scalars to Python types for routing compatibility
        if hasattr(key, "item"):
            key = key.item()
        computed_db_id = int(row[DB_ID_COL])

        routing_context: dict[str, object] | None = None
        if cel_columns is not None:
            routing_context = {
                col: row[col].item() if hasattr(row[col], "item") else row[col]
                for col in cel_columns
            }

        if isinstance(resolved_sharding, HashShardingSpec):
            expected_db_id = route_hash(
                key,
                num_dbs=num_dbs,
                hash_algorithm=resolved_sharding.hash_algorithm,
            )
        else:
            assert isinstance(resolved_sharding, CelShardingSpec)
            expected_db_id = route_cel(
                key,
                cel_expr=resolved_sharding.cel_expr,
                cel_columns=resolved_sharding.cel_columns,
                routing_values=resolved_sharding.routing_values,
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


def _verify_vector_routing_agreement(
    ddf_with_id: dd.DataFrame,
    *,
    id_col: str,
    vector_col: str,
    routing: Any,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify vector db_id column matches Python routing."""
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

    sampled = ddf_with_id[sample_cols].head(sample_size, npartitions=-1)
    if sampled.empty:
        return

    mismatches: list[tuple[object, int, int]] = []
    for _, row in sampled.iterrows():
        vector_id = row[id_col].item() if hasattr(row[id_col], "item") else row[id_col]
        shard_id: int | None = None
        if routing.strategy == VectorShardingStrategy.EXPLICIT:
            assert shard_id_col is not None, "shard_id_col required for EXPLICIT"
            shard_value = row[shard_id_col]
            shard_id = int(
                shard_value.item() if hasattr(shard_value, "item") else shard_value
            )

        routing_context: dict[str, object] | None = None
        if routing.strategy == VectorShardingStrategy.CEL:
            assert routing_context_cols is not None
            routing_context = {
                col: row[col].item() if hasattr(row[col], "item") else row[col]
                for col in routing_context_cols
            }

        expected_db_id = assign_vector_shard(
            vector=coerce_vector_value(row[vector_col]),
            routing=routing,
            shard_id=shard_id,
            routing_context=routing_context,
        )
        computed_db_id = int(row[VECTOR_DB_ID_COL])
        if expected_db_id != computed_db_id:
            mismatches.append((vector_id, computed_db_id, expected_db_id))

    if mismatches:
        details = "; ".join(
            f"id={vector_id}, dask={dask_db_id}, python={python_db_id}"
            for vector_id, dask_db_id, python_db_id in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Dask/Python vector routing mismatch in {len(mismatches)}/{len(sampled)} "
            f"sampled rows. First mismatches: {details}"
        )


def _build_partition_write_runtime(
    *,
    config: WriteConfig,
    run_id: str,
    key_col: str,
    value_spec: ValueSpec,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    sort_within_partitions: bool,
    started: float,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None,
) -> _PartitionWriteRuntime:
    """Construct picklable runtime config for Dask partition writers."""

    factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory(
        credential_provider=config.credential_provider
    )
    if config.vector_spec is not None:
        factory = wrap_factory_for_vector(factory, config)

    return _PartitionWriteRuntime(
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
        sort_within_partitions=sort_within_partitions,
        metrics_collector=config.metrics_collector,
        started=started,
        shard_retry=config.shard_retry,
        vector_fn=vector_fn,
    )


def _write_partition(
    pdf: pd.DataFrame,
    runtime: _PartitionWriteRuntime,
) -> pd.DataFrame:
    """Write all db_id groups within one Dask partition."""

    if pdf.empty:
        return _RESULT_META.iloc[:0].copy()

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


def write_vector_sharded(
    ddf: dd.DataFrame,
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
    """Write vectors from a Dask DataFrame into N sharded vector indices.

    Args:
        ddf: Dask DataFrame with vector and ID columns.
        config: Hash write configuration.
        vector_col: Name of the vector column.
        id_col: Name of the ID column.
        payload_cols: Optional payload columns.
        shard_id_col: Column with explicit shard IDs (EXPLICIT strategy only).
        routing_context_cols: Columns for CEL expression evaluation.
        max_writes_per_second: Optional rate limit.
        verify_routing: If True (default), spot-check that Dask-assigned vector
            shard IDs match ``assign_vector_shard()``.

    Returns:
        BuildResult with manifest reference and stats.
    """
    import numpy as np

    from shardyfusion.vector._distributed import (
        publish_vector_manifest,
        resolve_adapter_factory,
        resolve_vector_routing,
        upload_routing_metadata,
    )
    from shardyfusion.vector.config import VectorWriteConfig

    from .sharding import add_vector_db_id_column

    assert config.vector_spec is not None, "vector_spec is required for vector writes"

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    mc = config.metrics_collector
    with RunRecordLifecycle.start(
        config=config,
        run_id=run_id,
        writer_type="vector-dask",
    ) as run_record:
        sample_vectors: np.ndarray | None = None
        if (
            config.vector_spec.sharding.strategy == "cluster"
            and config.vector_spec.sharding.train_centroids
        ):
            sample_limit = config.vector_spec.sharding.centroids_training_sample_size
            sample_values = (
                ddf[vector_col]
                .sample(
                    frac=0.1,
                    random_state=42,
                )
                .head(sample_limit, compute=True)
                .tolist()
            )
            if not sample_values:
                sample_values = (
                    ddf[vector_col].head(sample_limit, compute=True).tolist()
                )
            sample_vectors = _stack_vector_values(sample_values)

        resolved_vector_factory = None
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

        ddf_with_id, num_dbs = add_vector_db_id_column(
            ddf,
            vector_col=vector_col,
            routing=routing,
            shard_id_col=shard_id_col,
            routing_context_cols=routing_context_cols,
        )

        if verify_routing and num_dbs > 0:
            _verify_vector_routing_agreement(
                ddf_with_id,
                id_col=id_col,
                vector_col=vector_col,
                routing=routing,
                shard_id_col=shard_id_col,
                routing_context_cols=routing_context_cols,
            )

        ddf_shuffled = ddf_with_id.shuffle(on=VECTOR_DB_ID_COL, npartitions=num_dbs)

        _id_col = id_col
        _vector_col = vector_col
        _runtime = _VectorPartitionWriteRuntime(
            run_id=run_id,
            s3_prefix=vec_config.s3_prefix,
            shard_prefix=vec_config.output.shard_prefix,
            db_path_template=vec_config.output.db_path_template,
            local_root=vec_config.output.local_root,
            index_config=vec_config.index_config,
            adapter_factory=adapter_factory,
            batch_size=vec_config.batch_size,
            payload_cols=payload_cols,
        )

        def _write_partition_vector(
            pdf: pd.DataFrame, runtime: _VectorPartitionWriteRuntime
        ) -> pd.DataFrame:  # noqa: N803
            from shardyfusion.vector._distributed import (
                VectorTuple,
                coerce_vector_value,
                write_vector_shard,
            )

            if pdf.empty:
                return pd.DataFrame(columns=["db_id", "db_url", "attempt", "row_count"])

            pdf_copy = pdf.copy()
            pdf_copy["_temp_vec_id"] = pdf_copy[VECTOR_DB_ID_COL].astype(int)
            groups = pdf_copy.groupby("_temp_vec_id")

            results = []
            for db_id, group in groups:
                _pcols = runtime.payload_cols
                rows_iter: list[VectorTuple] = [
                    (
                        _normalize_vector_id(row[_id_col]),
                        coerce_vector_value(row[_vector_col]),
                        {col: row[col] for col in _pcols} if _pcols else None,
                    )
                    for _, row in group.iterrows()
                ]
                result = write_vector_shard(
                    db_id=int(db_id),  # type: ignore[arg-type]
                    rows=rows_iter,
                    run_id=runtime.run_id,
                    s3_prefix=runtime.s3_prefix,
                    shard_prefix=runtime.shard_prefix,
                    db_path_template=runtime.db_path_template,
                    local_root=runtime.local_root,
                    index_config=runtime.index_config,
                    adapter_factory=runtime.adapter_factory,
                    batch_size=runtime.batch_size,
                )
                results.append(result)

            return pd.DataFrame.from_records(
                [_vector_result_row(result) for result in results],
                columns=list(_RESULT_META.columns),
            )

        ddf_results = ddf_shuffled.map_partitions(
            _write_partition_vector,
            _runtime,
            meta=_RESULT_META,
        )

        results = ddf_results.compute()

        winners = list(results.itertuples(index=False, name=None))
        winners = [
            RequiredShardMeta(
                db_id=int(db_id),
                db_url=db_url,
                attempt=int(attempt),
                row_count=int(row_count),
                checkpoint_id=checkpoint_id,
                writer_info=writer_info,
                db_bytes=int(db_bytes),
            )
            for (
                db_id,
                db_url,
                attempt,
                row_count,
                _min_key,
                _max_key,
                checkpoint_id,
                writer_info,
                db_bytes,
                _all_attempt_urls,
            ) in winners
        ]
        num_attempts = len(winners)

        total_vectors = sum(w.row_count for w in winners)
        total_duration_ms = int((time.perf_counter() - started) * 1000)

        from shardyfusion.storage import ObstoreBackend, create_s3_store, parse_s3_url

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
