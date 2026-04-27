"""Ray Data-based sharded writer (no Spark/Java dependency)."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pandas as pd
import ray.data
from ray.data import DataContext

from shardyfusion._shard_writer import (
    iter_pandas_rows,
    results_pdf_to_attempts,
    write_shard_with_retry,
    write_shard_with_retry_distributed,
)
from shardyfusion._writer_core import (
    VectorColumnMapping,
    _normalize_vector_id,
    assemble_build_result,
    cleanup_losers,
    inject_vector_manifest_fields,
    publish_to_store,
    resolve_distributed_vector_fn,
    route_key,
    select_winners,
    wrap_factory_for_vector,
)
from shardyfusion.config import WriteConfig
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
from shardyfusion.run_registry import managed_run_record
from shardyfusion.serde import KeyEncoder, ValueSpec, make_key_encoder
from shardyfusion.sharding_types import (
    DB_ID_COL,
    KeyEncoding,
    ShardingSpec,
    ShardingStrategy,
)
from shardyfusion.slatedb_adapter import (
    DbAdapterFactory,
    SlateDbFactory,
)
from shardyfusion.type_defs import RetryConfig

from .sharding import (
    VECTOR_DB_ID_COL,
    add_db_id_column,
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


@dataclass(slots=True)
class _PartitionWriteRuntime:
    """Picklable runtime config for Ray partition writers."""

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
    """Picklable runtime config for Ray vector partition writers."""

    run_id: str
    s3_prefix: str
    shard_prefix: str
    db_path_template: str
    local_root: str
    index_config: Any
    adapter_factory: Any
    batch_size: int
    id_col: str
    vector_col: str
    payload_cols: list[str] | None = None


# Meta schema for result DataFrames returned by partition writers.
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


def write_sharded(
    ds: ray.data.Dataset,
    config: WriteConfig,
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
    """Write a Ray Dataset into N independent sharded databases and publish manifest.

    Args:
        ds: Ray Dataset containing at least the key column and value column(s).
        config: Write configuration (num_dbs, s3_prefix, sharding strategy, etc.).
            HASH and CEL sharding strategies are supported.
        key_col: Name of the key column used for shard routing.
        value_spec: Specifies how DataFrame rows are serialized to bytes
            (binary_col, json_cols, or a callable encoder).
        sort_within_partitions: If True, sort rows by key within each partition.
        max_writes_per_second: Optional rate limit (token-bucket) for write ops/sec.
        max_write_bytes_per_second: Optional rate limit (token-bucket) for write bytes/sec.
        verify_routing: If True (default), spot-check that Ray-assigned shard IDs
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
    from shardyfusion.logging import LogContext

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    mc = config.metrics_collector

    with (
        LogContext(run_id=run_id),
        managed_run_record(
            config=config,
            run_id=run_id,
            writer_type="ray",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=config.num_dbs,
            s3_prefix=config.s3_prefix,
            strategy=config.sharding.strategy,
            key_encoding=config.key_encoding,
            writer_type="ray",
        )
        if mc is not None:
            mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})

        distributed_vector_fn = resolve_distributed_vector_fn(
            config=config,
            key_col=key_col,
            vector_fn=vector_fn,
            vector_columns=vector_columns,
        )
        # --- Phase 1: Sharding ---
        shard_started = time.perf_counter()
        from shardyfusion._writer_core import discover_cel_num_dbs

        resolved_sharding = config.sharding
        num_dbs = _resolve_num_dbs_before_sharding(ds, config)

        ds_with_id, resolved_sharding = add_db_id_column(
            ds,
            key_col=key_col,
            num_dbs=num_dbs,
            sharding=resolved_sharding,
            key_encoding=config.key_encoding,
        )

        # CEL: discover num_dbs from data and validate consecutive IDs
        if num_dbs is None and resolved_sharding.routing_values is not None:
            num_dbs = max(1, len(resolved_sharding.routing_values))
        elif num_dbs is None:
            distinct_ids = set(ds_with_id.unique(DB_ID_COL))
            num_dbs = discover_cel_num_dbs(distinct_ids)

        if verify_routing and num_dbs > 0:
            _verify_routing_agreement(
                ds_with_id,
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

        # Use hash shuffle for efficient repartition by db_id.
        # HASH_SHUFFLE is the default strategy in Ray Data, so we just need
        # shuffle=True + keys=[DB_ID_COL] for key-based co-location.
        # Save/restore the strategy in case the caller changed it.
        # Lock protects against concurrent write_sharded() calls racing
        # on the process-global DataContext.shuffle_strategy.
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
            _write_partition,  # type: ignore[arg-type]  # Ray stubs don't account for fn_kwargs
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
    ds: ray.data.Dataset, config: WriteConfig
) -> int | None:
    """Resolve num_dbs that can be determined before add_db_id_column."""
    from shardyfusion._writer_core import resolve_num_dbs

    return resolve_num_dbs(config, ds.count)


def _verify_routing_agreement(
    ds_with_id: ray.data.Dataset,
    *,
    key_col: str,
    num_dbs: int,
    resolved_sharding: ShardingSpec,
    key_encoding: KeyEncoding,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify db_id column matches Python routing."""

    sampled = ds_with_id.take(sample_size)
    if not sampled:
        return

    # For CEL with non-key columns, build routing_context from row data.
    cel_columns = (
        resolved_sharding.cel_columns
        if resolved_sharding.strategy == ShardingStrategy.CEL
        else None
    )

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
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

        expected_db_id = route_key(
            key,
            num_dbs=num_dbs,
            sharding=resolved_sharding,
            routing_context=routing_context,
        )

        if expected_db_id != computed_db_id:
            mismatches.append((key, computed_db_id, expected_db_id))

    if mismatches:
        details = "; ".join(
            f"key={k}, ray={d}, python={p}" for k, d, p in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Ray/Python routing mismatch in {len(mismatches)}/{len(sampled)} "
            f"sampled rows. First mismatches: {details}"
        )


def _verify_vector_routing_agreement(
    ds_with_id: ray.data.Dataset,
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

    sampled = ds_with_id.take(sample_size)
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
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
            assert routing_context_cols is not None, (
                "routing_context_cols required for CEL verification"
            )
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
            f"id={vector_id}, ray={ray_db_id}, python={python_db_id}"
            for vector_id, ray_db_id, python_db_id in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Ray/Python vector routing mismatch in {len(mismatches)}/{len(sampled)} "
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
    """Construct picklable runtime config for Ray partition writers."""

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
    """Write all db_id groups within one Ray partition."""

    if pdf.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    results: list[dict[str, object]] = []

    factory: DbAdapterFactory = runtime.adapter_factory

    for db_id, group_pdf in pdf.groupby(DB_ID_COL):
        if runtime.sort_within_partitions:
            group_pdf = group_pdf.sort_values(runtime.key_col)

        if runtime.vector_fn is None:
            attempt_result = write_shard_with_retry(
                db_id=int(db_id),  # type: ignore[arg-type]
                rows_fn=lambda pdf=group_pdf: iter_pandas_rows(
                    pdf, runtime.key_col, runtime.value_spec
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
            )
        else:
            vec_fn = runtime.vector_fn
            assert vec_fn is not None
            attempt_result = write_shard_with_retry_distributed(
                db_id=int(db_id),  # type: ignore[arg-type]
                rows_fn=lambda pdf=group_pdf, vec_fn=vec_fn: (
                    (
                        row[runtime.key_col].item()
                        if hasattr(row[runtime.key_col], "item")
                        else row[runtime.key_col],
                        runtime.value_spec.encode(row),
                        vec_fn(row),
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


def write_vector_sharded(
    ds: ray.data.Dataset,
    config: WriteConfig,
    *,
    vector_col: str,
    id_col: str,
    payload_cols: list[str] | None = None,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
    max_writes_per_second: float | None = None,
    verify_routing: bool = True,
) -> BuildResult:
    """Write vectors from a Ray Dataset into N sharded vector indices.

    Args:
        ds: Ray Dataset with vector and ID columns.
        config: Write configuration.
        vector_col: Name of the vector column.
        id_col: Name of the ID column.
        payload_cols: Optional payload columns.
        shard_id_col: Column with explicit shard IDs (EXPLICIT strategy only).
        routing_context_cols: Columns for CEL expression evaluation.
        max_writes_per_second: Optional rate limit.
        verify_routing: If True (default), spot-check that Ray-assigned vector
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
    with managed_run_record(
        config=config,
        run_id=run_id,
        writer_type="vector-ray",
    ) as run_record:
        sample_vectors: np.ndarray | None = None
        if (
            config.vector_spec.sharding.strategy == "cluster"
            and config.vector_spec.sharding.train_centroids
        ):
            sample_limit = config.vector_spec.sharding.centroids_training_sample_size
            sample_rows = (
                ds.random_sample(0.1, seed=42)
                .select_columns([vector_col])
                .take(sample_limit)
            )
            if not sample_rows:
                sample_rows = ds.select_columns([vector_col]).take(sample_limit)
            sample_vectors = np.array(
                [row[vector_col] for row in sample_rows], dtype=np.float32
            )

        credentials = (
            config.credential_provider.resolve() if config.credential_provider else None
        )
        from shardyfusion.storage import create_s3_client

        s3_client = create_s3_client(credentials, config.s3_connection_options)

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

        adapter_factory = resolve_adapter_factory(vec_config, s3_client)

        ds_with_id, num_dbs = add_vector_db_id_column(
            ds,
            vector_col=vector_col,
            routing=routing,
            shard_id_col=shard_id_col,
            routing_context_cols=routing_context_cols,
        )

        if verify_routing and num_dbs > 0:
            _verify_vector_routing_agreement(
                ds_with_id,
                id_col=id_col,
                vector_col=vector_col,
                routing=routing,
                shard_id_col=shard_id_col,
                routing_context_cols=routing_context_cols,
            )

        ds_shuffled = ds_with_id.repartition(
            num_dbs,
            shuffle=True,
            keys=[VECTOR_DB_ID_COL],
        )

        runtime = _VectorPartitionWriteRuntime(
            run_id=run_id,
            s3_prefix=vec_config.s3_prefix,
            shard_prefix=vec_config.output.shard_prefix,
            db_path_template=vec_config.output.db_path_template,
            local_root=vec_config.output.local_root,
            index_config=vec_config.index_config,
            adapter_factory=adapter_factory,
            batch_size=vec_config.batch_size,
            id_col=id_col,
            vector_col=vector_col,
            payload_cols=payload_cols,
        )

        def _write_partition_vector(
            pdf: pd.DataFrame, runtime: _VectorPartitionWriteRuntime
        ) -> pd.DataFrame:
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
                        _normalize_vector_id(row[runtime.id_col]),
                        coerce_vector_value(row[runtime.vector_col]),
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
                columns=_RESULT_COLUMNS,
            )

        ds_results = ds_shuffled.map_batches(
            _write_partition_vector,  # type: ignore[arg-type]
            batch_format="pandas",
            fn_kwargs={"runtime": runtime},
        )

        results = ds_results.to_pandas()

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
            ) in results.itertuples(index=False, name=None)
        ]
        num_attempts = len(winners)

        total_vectors = sum(w.row_count for w in winners)
        total_duration_ms = int((time.perf_counter() - started) * 1000)

        centroids_ref, hyperplanes_ref = upload_routing_metadata(
            s3_prefix=config.s3_prefix,
            run_id=run_id,
            centroids=routing.centroids,
            hyperplanes=routing.hyperplanes,
            s3_client=s3_client,
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
