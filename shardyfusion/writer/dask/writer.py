"""Dask-based sharded writer (no Spark/Java dependency)."""

import time
from dataclasses import dataclass
from uuid import uuid4

import dask.dataframe as dd
import pandas as pd

from shardyfusion._shard_writer import (
    iter_pandas_rows,
    results_pdf_to_attempts,
    write_shard_with_retry,
)
from shardyfusion._writer_core import (
    assemble_build_result,
    cleanup_losers,
    publish_to_store,
    route_key,
    select_winners,
)
from shardyfusion.config import WriteConfig
from shardyfusion.credentials import CredentialProvider
from shardyfusion.errors import ShardAssignmentError
from shardyfusion.logging import (
    get_logger,
    log_event,
)
from shardyfusion.manifest import BuildResult
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

from .sharding import add_db_id_column

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
    adapter_factory: DbAdapterFactory | None
    credential_provider: CredentialProvider | None
    max_writes_per_second: float | None
    max_write_bytes_per_second: float | None = None
    sort_within_partitions: bool = False
    metrics_collector: MetricsCollector | None = None  # must be picklable
    started: float = 0.0
    shard_retry: RetryConfig | None = None


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
        "all_attempt_urls": "object",
    }
)


def write_sharded(
    ddf: dd.DataFrame,
    config: WriteConfig,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool = False,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    verify_routing: bool = True,
) -> BuildResult:
    """Write a Dask DataFrame into N independent sharded databases and publish manifest.

    Args:
        ddf: Dask DataFrame containing at least the key column and value column(s).
        config: Write configuration (num_dbs, s3_prefix, sharding strategy, etc.).
            HASH and CEL sharding strategies are supported.
        key_col: Name of the key column used for shard routing.
        value_spec: Specifies how DataFrame rows are serialized to bytes
            (binary_col, json_cols, or a callable encoder).
        sort_within_partitions: If True, sort rows by key within each partition.
        max_writes_per_second: Optional rate limit (token-bucket) for write ops/sec.
        max_write_bytes_per_second: Optional rate limit (token-bucket) for write bytes/sec.
        verify_routing: If True (default), spot-check that Dask-assigned shard IDs
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
            writer_type="dask",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=config.num_dbs,
            s3_prefix=config.s3_prefix,
            strategy=config.sharding.strategy,
            key_encoding=config.key_encoding,
            writer_type="dask",
        )
        if mc is not None:
            mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})

        # --- Phase 1: Sharding ---
        shard_started = time.perf_counter()
        from shardyfusion._writer_core import discover_cel_num_dbs

        resolved_sharding = config.sharding
        num_dbs = _resolve_num_dbs_before_sharding(ddf, config)

        ddf_with_id = add_db_id_column(
            ddf,
            key_col=key_col,
            num_dbs=num_dbs,
            sharding=resolved_sharding,
            key_encoding=config.key_encoding,
        )

        # CEL: discover num_dbs from data and validate consecutive IDs
        if num_dbs is None:
            distinct_ids = set(ddf_with_id[DB_ID_COL].unique().compute().tolist())
            num_dbs = discover_cel_num_dbs(distinct_ids)

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
    ddf: dd.DataFrame, config: WriteConfig
) -> int | None:
    """Resolve num_dbs that can be determined before add_db_id_column."""
    from shardyfusion._writer_core import resolve_num_dbs

    return resolve_num_dbs(config, lambda: len(ddf))


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
        if resolved_sharding.strategy == ShardingStrategy.CEL
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
            f"key={k}, dask={d}, python={p}" for k, d, p in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Dask/Python routing mismatch in {len(mismatches)}/{len(sampled)} "
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
) -> _PartitionWriteRuntime:
    """Construct picklable runtime config for Dask partition writers."""

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
        adapter_factory=config.adapter_factory,
        credential_provider=config.credential_provider,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
        sort_within_partitions=sort_within_partitions,
        metrics_collector=config.metrics_collector,
        started=started,
        shard_retry=config.shard_retry,
    )


def _write_partition(
    pdf: pd.DataFrame,
    runtime: _PartitionWriteRuntime,
) -> pd.DataFrame:
    """Write all db_id groups within one Dask partition."""

    if pdf.empty:
        return _RESULT_META.iloc[:0].copy()

    results: list[dict[str, object]] = []

    factory: DbAdapterFactory = runtime.adapter_factory or SlateDbFactory(
        credential_provider=runtime.credential_provider
    )

    for db_id, group_pdf in pdf.groupby(DB_ID_COL):
        if runtime.sort_within_partitions:
            group_pdf = group_pdf.sort_values(runtime.key_col)

        attempt_result = write_shard_with_retry(
            db_id=int(db_id),  # type: ignore[invalid-argument-type]  # pandas groupby key
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
                "all_attempt_urls": attempt_result.all_attempt_urls,
            }
        )

    if not results:
        return _RESULT_META.iloc[:0].copy()

    return pd.DataFrame(results)
