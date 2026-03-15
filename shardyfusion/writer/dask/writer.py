"""Dask-based sharded writer (no Spark/Java dependency)."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import dask.dataframe as dd
import pandas as pd

from shardyfusion._rate_limiter import RateLimiter, TokenBucket
from shardyfusion._writer_core import (
    ShardAttemptResult,
    assemble_build_result,
    publish_to_store,
    route_key,
    select_winners,
    update_min_max,
)
from shardyfusion.config import WriteConfig
from shardyfusion.errors import (
    ConfigValidationError,
    ShardAssignmentError,
    ShardyfusionError,
)
from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_event,
    log_failure,
)
from shardyfusion.manifest import BuildResult
from shardyfusion.metrics import MetricEvent, MetricsCollector
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
from shardyfusion.storage import join_s3
from shardyfusion.type_defs import JsonObject, KeyLike

from .sharding import add_db_id_column, compute_range_boundaries

_logger = get_logger(__name__)


@dataclass(slots=True)
class _PartitionWriteRuntime:
    """Picklable runtime config for Dask partition writers."""

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
    sort_within_partitions: bool = False
    metrics_collector: MetricsCollector | None = None  # must be picklable
    started: float = 0.0


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
            CUSTOM_EXPR sharding is not supported — only HASH and RANGE.
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
        ConfigValidationError: If configuration is invalid or CUSTOM_EXPR is used.
        ShardAssignmentError: If rows cannot be assigned to valid shard IDs.
        ShardCoverageError: If partition results don't cover all expected shards.
        PublishManifestError: If manifest upload to S3 fails.
        PublishCurrentError: If CURRENT pointer upload fails (manifest already published).
    """

    from shardyfusion.logging import LogContext

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    mc = config.metrics_collector

    with LogContext(run_id=run_id):
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
        resolved_sharding = _validate_and_resolve_sharding(config, ddf, key_col)

        ddf_with_id = add_db_id_column(
            ddf,
            key_col=key_col,
            num_dbs=config.num_dbs,
            sharding=resolved_sharding,
            key_encoding=config.key_encoding,
        )

        if verify_routing:
            _verify_routing_agreement(
                ddf_with_id,
                key_col=key_col,
                num_dbs=config.num_dbs,
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
        ddf_shuffled = ddf_with_id.shuffle(on=DB_ID_COL, npartitions=config.num_dbs)
        ddf_results = ddf_shuffled.map_partitions(
            _write_partition,
            runtime=runtime,
            meta=_RESULT_META,
        )
        results_pdf = ddf_results.compute()

        attempts = _results_pdf_to_attempts(results_pdf)
        attempts = _fill_empty_shards(attempts, config, run_id)
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
        winners, num_attempts, _attempt_urls = select_winners(
            attempts, num_dbs=config.num_dbs
        )

        manifest_started = time.perf_counter()
        manifest_ref = publish_to_store(
            config=config,
            run_id=run_id,
            resolved_sharding=resolved_sharding,
            winners=winners,
            key_col=key_col,
            started=started,
        )
        manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

        result = assemble_build_result(
            run_id=run_id,
            winners=winners,
            manifest_ref=manifest_ref,
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

        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_and_resolve_sharding(
    config: WriteConfig,
    ddf: dd.DataFrame,
    key_col: str,
) -> ShardingSpec:
    """Validate sharding config and resolve RANGE boundaries via Dask quantiles."""

    if config.sharding.strategy == ShardingStrategy.CUSTOM_EXPR:
        raise ConfigValidationError(
            "Custom expression sharding is not supported in the Dask writer."
        )

    if (
        config.sharding.strategy == ShardingStrategy.RANGE
        and config.sharding.boundaries is None
    ):
        boundaries = compute_range_boundaries(
            ddf,
            key_col=key_col,
            num_dbs=config.num_dbs,
            rel_error=config.sharding.approx_quantile_rel_error,
        )
        return ShardingSpec(
            strategy=ShardingStrategy.RANGE,
            boundaries=boundaries,
            approx_quantile_rel_error=config.sharding.approx_quantile_rel_error,
        )

    return config.sharding


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

    sampled = ddf_with_id[[key_col, DB_ID_COL]].head(sample_size, npartitions=-1)
    if sampled.empty:
        return

    mismatches: list[tuple[object, int, int]] = []
    for _, row in sampled.iterrows():
        key = row[key_col]
        # Convert numpy scalars to Python types for routing compatibility
        if hasattr(key, "item"):
            key = key.item()
        computed_db_id = int(row[DB_ID_COL])

        expected_db_id = route_key(
            key,
            num_dbs=num_dbs,
            sharding=resolved_sharding,
            key_encoding=key_encoding,
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
        sort_within_partitions=sort_within_partitions,
        metrics_collector=config.metrics_collector,
        started=started,
    )


def _write_partition(
    pdf: pd.DataFrame,
    runtime: _PartitionWriteRuntime,
) -> pd.DataFrame:
    """Write all db_id groups within one Dask partition."""

    if pdf.empty:
        return _RESULT_META.iloc[:0].copy()

    results: list[dict[str, object]] = []

    for db_id, group_pdf in pdf.groupby(DB_ID_COL):
        if runtime.sort_within_partitions:
            group_pdf = group_pdf.sort_values(runtime.key_col)

        attempt_result = _write_one_shard(
            int(db_id),  # type: ignore[invalid-argument-type]  # pandas groupby key
            group_pdf,
            runtime,
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
            }
        )

    if not results:
        return _RESULT_META.iloc[:0].copy()

    return pd.DataFrame(results)


def _write_one_shard(
    db_id: int,
    pdf: pd.DataFrame,
    runtime: _PartitionWriteRuntime,
) -> ShardAttemptResult:
    """Write one shard from a pandas DataFrame group."""

    attempt = 0
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

    factory: DbAdapterFactory = runtime.adapter_factory or SlateDbFactory()

    bucket: RateLimiter | None = None
    if runtime.max_writes_per_second is not None:
        bucket = TokenBucket(runtime.max_writes_per_second, metrics_collector=mc)
    byte_bucket: RateLimiter | None = None
    if runtime.max_write_bytes_per_second is not None:
        byte_bucket = TokenBucket(
            runtime.max_write_bytes_per_second,
            metrics_collector=mc,
            limiter_type="bytes",
        )

    partition_started = time.perf_counter()
    row_count = 0
    min_key: KeyLike | None = None
    max_key: KeyLike | None = None
    checkpoint_id: str | None = None
    batch: list[tuple[bytes, bytes]] = []

    try:
        with factory(db_url=db_url, local_dir=local_dir) as adapter:
            for _, row in pdf.iterrows():
                key_value: Any = row[runtime.key_col]
                # Convert numpy scalars to Python types for downstream compatibility
                if hasattr(key_value, "item"):
                    key_value = key_value.item()

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
    except ShardyfusionError:
        raise
    except Exception as exc:
        log_failure(
            "dask_shard_write_failed",
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
        "stage_id": None,
        "task_attempt_id": None,
        "attempt": attempt,
        "duration_ms": duration_ms,
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


def _results_pdf_to_attempts(
    results_pdf: pd.DataFrame,
) -> list[ShardAttemptResult]:
    """Convert result pandas DataFrame to list of ShardAttemptResult."""

    if results_pdf.empty:
        return []

    attempts: list[ShardAttemptResult] = []
    for _, row in results_pdf.iterrows():
        # Extract pandas row values as Any to satisfy both ty and pyright
        r: Any = row
        min_key = r["min_key"]
        max_key = r["max_key"]
        checkpoint_id = r["checkpoint_id"]

        # Handle pandas NaN for None values (int + None columns become float64)
        if isinstance(min_key, float) and pd.isna(min_key):
            min_key = None
        elif isinstance(min_key, float):
            min_key = int(min_key)
        if isinstance(max_key, float) and pd.isna(max_key):
            max_key = None
        elif isinstance(max_key, float):
            max_key = int(max_key)
        if isinstance(checkpoint_id, float) and pd.isna(checkpoint_id):
            checkpoint_id = None

        attempts.append(
            ShardAttemptResult(
                db_id=int(r["db_id"]),
                db_url=str(r["db_url"]),
                attempt=int(r["attempt"]),
                row_count=int(r["row_count"]),
                min_key=min_key,
                max_key=max_key,
                checkpoint_id=None if checkpoint_id is None else str(checkpoint_id),
                writer_info=r["writer_info"],
            )
        )

    return attempts


def _fill_empty_shards(
    attempts: list[ShardAttemptResult],
    config: WriteConfig,
    run_id: str,
) -> list[ShardAttemptResult]:
    """Add zero-row results for any db_ids not covered by partition writes."""

    seen_db_ids = {a.db_id for a in attempts}
    for db_id in range(config.num_dbs):
        if db_id not in seen_db_ids:
            db_rel_path = config.output.db_path_template.format(db_id=db_id)
            db_url = join_s3(
                config.s3_prefix,
                config.output.tmp_prefix,
                f"run_id={run_id}",
                db_rel_path,
                "attempt=00",
            )
            attempts.append(
                ShardAttemptResult(
                    db_id=db_id,
                    db_url=db_url,
                    attempt=0,
                    row_count=0,
                    min_key=None,
                    max_key=None,
                    checkpoint_id=None,
                    writer_info={
                        "stage_id": None,
                        "task_attempt_id": None,
                        "attempt": 0,
                        "duration_ms": 0,
                    },
                )
            )

    return attempts
