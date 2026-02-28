"""Public sharded snapshot writer entrypoint."""

import logging
import os
import time
import types
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Self
from uuid import uuid4

from pyspark import RDD, StorageLevel, TaskContext
from pyspark.sql import DataFrame, Row, SparkSession

from slatedb_spark_sharded._rate_limiter import TokenBucket
from slatedb_spark_sharded._writer_core import (
    PartitionWriteOutcome,
    ShardAttemptResult,
    assemble_build_result,
    build_manifest_artifact,
    publish_manifest_and_current,
    select_winners,
    update_min_max,
)
from slatedb_spark_sharded.config import WriteConfig
from slatedb_spark_sharded.errors import ShardAssignmentError, SlatedbSparkShardedError
from slatedb_spark_sharded.logging import FailureSeverity, log_event, log_failure
from slatedb_spark_sharded.manifest import BuildResult
from slatedb_spark_sharded.serde import KeyEncoder, ValueSpec, make_key_encoder
from slatedb_spark_sharded.sharding_types import DB_ID_COL, KeyEncoding
from slatedb_spark_sharded.slatedb_adapter import (
    DbAdapterFactory,
    default_adapter_factory,
)
from slatedb_spark_sharded.storage import join_s3
from slatedb_spark_sharded.type_defs import JsonObject, KeyLike

from .sharding import ShardingSpec, add_db_id_column, prepare_partitioned_rdd


@dataclass(slots=True)
class PartitionWriteConfig:
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


@dataclass(slots=True)
class _PreparedPartitionRows:
    partitioned_rdd: RDD[tuple[int, Row]]
    resolved_sharding: ShardingSpec
    shard_duration_ms: int


class SparkConfOverrideContext:
    """Temporarily override Spark configuration values and restore them on exit."""

    def __init__(self, spark: SparkSession, overrides: dict[str, str] | None) -> None:
        self._spark = spark
        self._overrides = dict(overrides or {})
        self._original_values: dict[str, str | None] = {}

    def __enter__(self) -> Self:
        for key, value in self._overrides.items():
            self._original_values[key] = self._get_conf_or_none(key)
            try:
                self._spark.conf.set(key, value)
            except Exception as exc:  # pragma: no cover - Spark environment dependent
                log_event(
                    "spark_conf_override_failed",
                    level=logging.WARNING,
                    key=key,
                    value=value,
                    error=str(exc),
                )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        for key in reversed(list(self._overrides.keys())):
            original = self._original_values.get(key)
            try:
                if original is None:
                    self._unset_conf_if_supported(key)
                else:
                    self._spark.conf.set(key, original)
            except (
                Exception
            ) as restore_exc:  # pragma: no cover - Spark environment dependent
                log_event(
                    "spark_conf_restore_failed",
                    level=logging.WARNING,
                    key=key,
                    original_value=original,
                    error=str(restore_exc),
                )

    def _get_conf_or_none(self, key: str) -> str | None:
        try:
            return self._spark.conf.get(key, None)
        except Exception:  # pragma: no cover - Spark environment dependent
            log_event(
                "spark_conf_get_failed",
                level=logging.WARNING,
                key=key,
            )
            return None

    def _unset_conf_if_supported(self, key: str) -> None:
        self._spark.conf.unset(key)


class DataFrameCacheContext:
    """Cache a DataFrame for the lifetime of the context and unpersist on exit."""

    def __init__(
        self,
        df: DataFrame,
        storage_level: StorageLevel | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._df = df
        self._storage_level = storage_level
        self._enabled = enabled
        self._cached_df: DataFrame | None = None

    def __enter__(self) -> DataFrame:
        if not self._enabled:
            return self._df
        try:
            if self._storage_level is None:
                self._cached_df = self._df.persist()
            else:
                self._cached_df = self._df.persist(self._storage_level)
        except Exception as exc:  # pragma: no cover - Spark environment dependent
            log_event(
                "dataframe_cache_failed",
                level=logging.WARNING,
                error=str(exc),
            )
            self._cached_df = self._df
        return self._cached_df

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        if not self._enabled or self._cached_df is None:
            return
        try:
            self._cached_df.unpersist(blocking=False)
        except Exception as unpersist_exc:  # pragma: no cover - Spark env dependent
            log_event(
                "dataframe_unpersist_failed",
                level=logging.WARNING,
                error=str(unpersist_exc),
            )


def write_sharded_spark(
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
    verify_routing: bool = True,
) -> BuildResult:
    """Write a DataFrame into N independent sharded databases and publish manifest metadata."""

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex
    spark = df.sparkSession
    with SparkConfOverrideContext(spark, spark_conf_overrides):
        with DataFrameCacheContext(
            df, storage_level=storage_level, enabled=cache_input
        ) as cached_df:
            return _write_sharded_spark_impl(
                df=cached_df,
                config=config,
                run_id=run_id,
                started=started,
                key_col=key_col,
                value_spec=value_spec,
                sort_within_partitions=sort_within_partitions,
                max_writes_per_second=max_writes_per_second,
                verify_routing=verify_routing,
            )


def _write_sharded_spark_impl(
    *,
    df: DataFrame,
    config: WriteConfig,
    run_id: str,
    started: float,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool,
    max_writes_per_second: float | None,
    verify_routing: bool,
) -> BuildResult:
    """Implementation assuming Spark conf already prepared."""
    prepared_rows = _prepare_partitioned_rows(
        df=df,
        config=config,
        key_col=key_col,
        sort_within_partitions=sort_within_partitions,
        verify_routing=verify_routing,
    )
    runtime = _build_partition_write_runtime(
        config=config,
        run_id=run_id,
        key_col=key_col,
        value_spec=value_spec,
        max_writes_per_second=max_writes_per_second,
    )
    write_outcome = _run_partition_writes(
        partitioned_rdd=prepared_rows.partitioned_rdd,
        runtime=runtime,
        num_dbs=config.num_dbs,
    )

    manifest_started = time.perf_counter()
    artifact = build_manifest_artifact(
        config=config,
        run_id=run_id,
        resolved_sharding=prepared_rows.resolved_sharding,
        winners=write_outcome.winners,
        key_col=key_col,
    )
    publish_result = publish_manifest_and_current(
        config=config,
        run_id=run_id,
        artifact=artifact,
    )
    manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

    return assemble_build_result(
        run_id=run_id,
        winners=write_outcome.winners,
        artifact=artifact,
        manifest_ref=publish_result.manifest_ref,
        current_ref=publish_result.current_ref,
        attempts=write_outcome.attempts,
        shard_duration_ms=prepared_rows.shard_duration_ms,
        write_duration_ms=write_outcome.write_duration_ms,
        manifest_duration_ms=manifest_duration_ms,
        started=started,
    )


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

    Raises ShardAssignmentError if any sampled row disagrees between the
    Spark SQL expression and the equivalent Python routing function.
    """
    from bisect import bisect_right

    from slatedb_spark_sharded.routing import xxhash64_db_id
    from slatedb_spark_sharded.sharding_types import ShardingStrategy

    sampled = df_with_db_id.select(key_col, DB_ID_COL).limit(sample_size).collect()
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        key = row[key_col]
        spark_db_id = int(row[DB_ID_COL])

        if resolved_sharding.strategy == ShardingStrategy.HASH:
            python_db_id = xxhash64_db_id(key, num_dbs, key_encoding)
        elif (
            resolved_sharding.strategy == ShardingStrategy.RANGE
            and resolved_sharding.boundaries is not None
        ):
            python_db_id = bisect_right(resolved_sharding.boundaries, key)
        else:
            return  # cannot verify custom_expr or range without boundaries

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


def _prepare_partitioned_rows(
    *,
    df: DataFrame,
    config: WriteConfig,
    key_col: str,
    sort_within_partitions: bool,
    verify_routing: bool = True,
) -> _PreparedPartitionRows:
    """Assign shard ids and build the one-writer-per-db partitioned RDD."""

    shard_started = time.perf_counter()
    df_with_db_id, resolved_sharding = add_db_id_column(
        df,
        key_col=key_col,
        num_dbs=config.num_dbs,
        sharding=config.sharding,
    )
    if verify_routing:
        verify_routing_agreement(
            df_with_db_id,
            key_col=key_col,
            num_dbs=config.num_dbs,
            resolved_sharding=resolved_sharding,
            key_encoding=config.key_encoding,
        )
    partitioned_rdd = prepare_partitioned_rdd(
        df_with_db_id,
        num_dbs=config.num_dbs,
        key_col=key_col,
        sort_within_partitions=sort_within_partitions,
    )
    shard_duration_ms = int((time.perf_counter() - shard_started) * 1000)
    return _PreparedPartitionRows(
        partitioned_rdd=partitioned_rdd,
        resolved_sharding=resolved_sharding,
        shard_duration_ms=shard_duration_ms,
    )


def _build_partition_write_runtime(
    *,
    config: WriteConfig,
    run_id: str,
    key_col: str,
    value_spec: ValueSpec,
    max_writes_per_second: float | None,
) -> PartitionWriteConfig:
    """Construct immutable worker-side runtime config for partition shard writers."""

    return PartitionWriteConfig(
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
    )


def _run_partition_writes(
    *,
    partitioned_rdd: RDD[tuple[int, Row]],
    runtime: PartitionWriteConfig,
    num_dbs: int,
) -> PartitionWriteOutcome:
    """Execute partition writers and deterministically select winning shard attempts."""

    write_started = time.perf_counter()
    attempts: list[ShardAttemptResult] = partitioned_rdd.mapPartitionsWithIndex(
        lambda db_id, items: write_one_shard_partition(db_id, items, runtime)
    ).collect()
    write_duration_ms = int((time.perf_counter() - write_started) * 1000)
    winners = select_winners(attempts, num_dbs=num_dbs)
    return PartitionWriteOutcome(
        attempts=attempts,
        winners=winners,
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

    db_rel_path = runtime.db_path_template.format(db_id=db_id)
    db_url = join_s3(
        runtime.s3_prefix,
        runtime.tmp_prefix,
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

    factory = runtime.adapter_factory or default_adapter_factory

    bucket: TokenBucket | None = None
    if runtime.max_writes_per_second is not None:
        bucket = TokenBucket(runtime.max_writes_per_second)

    partition_started = time.perf_counter()
    row_count = 0
    min_key: KeyLike | None = None
    max_key: KeyLike | None = None
    checkpoint_id: str | None = None
    batch: list[tuple[bytes, bytes]] = []

    try:
        with factory(
            db_url=db_url,
            local_dir=local_dir,
        ) as adapter:
            for _, row in rows_iter:
                key_value = row[runtime.key_col]
                key_bytes = runtime.key_encoder(key_value)
                value_bytes = runtime.value_spec.encode(row)

                batch.append((key_bytes, value_bytes))
                row_count += 1
                min_key, max_key = update_min_max(min_key, max_key, key_value)

                if len(batch) >= runtime.batch_size:
                    if bucket is not None:
                        bucket.acquire(len(batch))
                    adapter.write_batch(batch)
                    batch.clear()

            if batch:
                if bucket is not None:
                    bucket.acquire(len(batch))
                adapter.write_batch(batch)
                batch.clear()

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
    except Exception as exc:  # pragma: no cover - worker runtime failure surface
        log_failure(
            "shard_write_failed",
            severity=FailureSeverity.ERROR,
            error=exc,
            db_id=db_id,
            attempt=attempt,
            db_url=db_url,
            rows_written=row_count,
            include_traceback=True,
        )
        raise SlatedbSparkShardedError(
            f"Shard write failed for db_id={db_id}, attempt={attempt}: {exc}"
        ) from exc

    writer_info: JsonObject = {
        "stage_id": stage_id,
        "task_attempt_id": task_attempt_id,
        "attempt": attempt,
        "duration_ms": int((time.perf_counter() - partition_started) * 1000),
    }

    yield ShardAttemptResult(
        db_id=db_id,
        db_url=db_url,
        attempt=attempt,
        row_count=row_count,
        min_key=min_key,
        max_key=max_key,
        checkpoint_id=checkpoint_id,
        writer_info=writer_info,
    )
