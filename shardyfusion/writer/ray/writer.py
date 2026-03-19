"""Ray Data-based sharded writer (no Spark/Java dependency)."""

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import ray.data
from ray.data import DataContext

from shardyfusion._rate_limiter import RateLimiter, TokenBucket
from shardyfusion._writer_core import (
    ShardAttemptResult,
    assemble_build_result,
    cleanup_losers,
    publish_to_store,
    route_key,
    select_winners,
    update_min_max,
)
from shardyfusion.config import WriteConfig
from shardyfusion.credentials import CredentialProvider
from shardyfusion.errors import (
    ShardAssignmentError,
    ShardyfusionError,
)
from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_event,
    log_failure,
)
from shardyfusion.manifest import BuildResult, WriterInfo
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
from shardyfusion.type_defs import KeyLike

from .sharding import add_db_id_column

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
    adapter_factory: DbAdapterFactory | None
    credential_provider: CredentialProvider | None
    max_writes_per_second: float | None
    max_write_bytes_per_second: float | None = None
    sort_within_partitions: bool = False
    metrics_collector: MetricsCollector | None = None  # must be picklable
    started: float = 0.0


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
]


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

    with LogContext(run_id=run_id):
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

        # --- Phase 1: Sharding ---
        shard_started = time.perf_counter()
        from shardyfusion._writer_core import discover_cel_num_dbs

        resolved_sharding = config.sharding
        num_dbs = _resolve_num_dbs_before_sharding(ds, config)

        ds_with_id = add_db_id_column(
            ds,
            key_col=key_col,
            num_dbs=num_dbs,
            sharding=resolved_sharding,
            key_encoding=config.key_encoding,
        )

        # CEL: discover num_dbs from data and validate consecutive IDs
        if num_dbs == 0:
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

        attempts = _results_pdf_to_attempts(results_pdf)
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
        manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

        cleanup_losers(all_attempt_urls, winners, metrics_collector=mc)

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


def _resolve_num_dbs_before_sharding(ds: ray.data.Dataset, config: WriteConfig) -> int:
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
            key_encoding=key_encoding,
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
    """Construct picklable runtime config for Ray partition writers."""

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
    )


def _write_partition(
    pdf: pd.DataFrame,
    runtime: _PartitionWriteRuntime,
) -> pd.DataFrame:
    """Write all db_id groups within one Ray partition."""

    if pdf.empty:
        return pd.DataFrame(columns=_RESULT_COLUMNS)

    results: list[dict[str, object]] = []

    for db_id, group_pdf in pdf.groupby(DB_ID_COL):
        if runtime.sort_within_partitions:
            group_pdf = group_pdf.sort_values(runtime.key_col)

        attempt_result = _write_one_shard(
            int(db_id),  # type: ignore[arg-type]
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
        return pd.DataFrame(columns=_RESULT_COLUMNS)

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
        runtime.shard_prefix,
        f"run_id={runtime.run_id}",
        db_rel_path,
        f"attempt={attempt:02d}",
    )
    local_dir = os.path.join(
        runtime.local_root,
        f"run_id={runtime.run_id}",
        f"db={db_id:05d}",
        f"attempt={attempt:02d}",
    )
    os.makedirs(local_dir, exist_ok=True)

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

    factory: DbAdapterFactory = runtime.adapter_factory or SlateDbFactory(
        credential_provider=runtime.credential_provider
    )

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
        with factory(db_url=db_url, local_dir=Path(local_dir)) as adapter:
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
            "ray_shard_write_failed",
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

    writer_info = WriterInfo(attempt=attempt, duration_ms=duration_ms)

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
