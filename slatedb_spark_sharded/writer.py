"""Public sharded snapshot writer entrypoint."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4

from pyspark import StorageLevel, TaskContext
from pyspark.sql import DataFrame, SparkSession

from .config import SlateDbConfig
from .errors import (
    ManifestBuildError,
    PublishCurrentError,
    PublishManifestError,
    ShardCoverageError,
    SlatedbSparkShardedError,
)
from .logging import log_event
from .manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    CurrentPointer,
    JsonManifestBuilder,
    ManifestArtifact,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from .publish import DefaultS3Publisher
from .serde import encode_key
from .sharding import ShardingSpec, add_db_id_column, prepare_partitioned_rdd
from .slatedb_adapter import default_adapter_factory

KEY_ENCODING = "u64be"


@dataclass(slots=True)
class _PartitionWriteConfig:
    run_id: str
    s3_prefix: str
    tmp_prefix: str
    db_path_template: str
    local_root: str
    key_col: str
    value_spec: Any
    batch_size: int
    slate_env_file: str | None
    slate_settings: dict[str, Any] | None
    slatedb_adapter_factory: Any | None


@dataclass(slots=True)
class _ShardAttemptResult:
    db_id: int
    db_url: str
    attempt: int
    row_count: int
    min_key: int | str | None
    max_key: int | str | None
    checkpoint_id: str | None
    writer_info: dict[str, Any]


class SparkConfOverrideContext:
    """Temporarily override Spark configuration values and restore them on exit."""

    def __init__(self, spark: SparkSession, overrides: dict[str, str] | None) -> None:
        self._spark = spark
        self._overrides = dict(overrides or {})
        self._original_values: dict[str, str | None] = {}

    def __enter__(self) -> "SparkConfOverrideContext":
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

    def __exit__(self, exc_type, exc, tb) -> None:
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
            return None

    def _unset_conf_if_supported(self, key: str) -> None:
        self._spark.conf.unset(key)


class DataFrameCacheContext:
    """Cache a DataFrame for the lifetime of the context and unpersist on exit."""

    def __init__(
        self, df: DataFrame, storage_level: StorageLevel | None = None
    ) -> None:
        self._df = df
        self._storage_level = storage_level
        self._cached_df: DataFrame | None = None

    def __enter__(self) -> DataFrame:
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

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._cached_df is None:
            return
        try:
            self._cached_df.unpersist(blocking=False)
        except Exception as unpersist_exc:  # pragma: no cover - Spark env dependent
            log_event(
                "dataframe_unpersist_failed",
                level=logging.WARNING,
                error=str(unpersist_exc),
            )


def write_sharded_slatedb(
    df: DataFrame,
    config: SlateDbConfig,
    spark_conf_overrides: dict[str, str] | None = None,
    cache_input: bool = False,
    storage_level: StorageLevel | None = None,
) -> BuildResult:
    """Write a DataFrame into N independent SlateDB shards and publish manifest metadata."""

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex
    spark = df.sparkSession
    with SparkConfOverrideContext(spark, spark_conf_overrides):
        if cache_input:
            with DataFrameCacheContext(df, storage_level=storage_level) as cached_df:
                return _write_sharded_slatedb_impl(
                    df=cached_df,
                    config=config,
                    run_id=run_id,
                    started=started,
                )
        return _write_sharded_slatedb_impl(
            df=df,
            config=config,
            run_id=run_id,
            started=started,
        )


def _write_sharded_slatedb_impl(
    *,
    df: DataFrame,
    config: SlateDbConfig,
    run_id: str,
    started: float,
) -> BuildResult:
    """Implementation for write_sharded_slatedb assuming Spark conf already prepared."""
    partitioned_rdd, resolved_sharding, shard_duration_ms = _prepare_partitioned_rows(
        df=df,
        config=config,
    )
    runtime = _build_partition_write_runtime(config=config, run_id=run_id)
    attempts, winners, write_duration_ms = _run_partition_writes(
        partitioned_rdd=partitioned_rdd,
        runtime=runtime,
        num_dbs=config.num_dbs,
    )

    manifest_started = time.perf_counter()
    artifact = _build_manifest_artifact(
        config=config,
        run_id=run_id,
        resolved_sharding=resolved_sharding,
        winners=winners,
    )
    manifest_ref, current_ref = _publish_manifest_and_current(
        config=config,
        run_id=run_id,
        artifact=artifact,
    )
    manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

    return _assemble_build_result(
        run_id=run_id,
        winners=winners,
        artifact=artifact,
        manifest_ref=manifest_ref,
        current_ref=current_ref,
        attempts=attempts,
        shard_duration_ms=shard_duration_ms,
        write_duration_ms=write_duration_ms,
        manifest_duration_ms=manifest_duration_ms,
        started=started,
    )


def _prepare_partitioned_rows(
    *,
    df: DataFrame,
    config: SlateDbConfig,
) -> tuple[Any, ShardingSpec, int]:
    """Assign shard ids and build the one-writer-per-db partitioned RDD."""

    shard_started = time.perf_counter()
    df_with_db_id, resolved_sharding = add_db_id_column(
        df,
        key_col=config.key_col,
        num_dbs=config.num_dbs,
        sharding=config.sharding.spec,
    )
    partitioned_rdd = prepare_partitioned_rdd(
        df_with_db_id,
        num_dbs=config.num_dbs,
        key_col=config.key_col,
        sort_within_partitions=config.sharding.sort_within_partitions,
    )
    shard_duration_ms = int((time.perf_counter() - shard_started) * 1000)
    return partitioned_rdd, resolved_sharding, shard_duration_ms


def _build_partition_write_runtime(
    *,
    config: SlateDbConfig,
    run_id: str,
) -> _PartitionWriteConfig:
    """Construct immutable worker-side runtime config for partition shard writers."""

    return _PartitionWriteConfig(
        run_id=run_id,
        s3_prefix=config.s3_prefix,
        tmp_prefix=config.output.tmp_prefix,
        db_path_template=config.output.db_path_template,
        local_root=config.output.local_root,
        key_col=config.key_col,
        value_spec=config.value_spec,
        batch_size=config.engine.batch_size,
        slate_env_file=config.engine.slate_env_file,
        slate_settings=config.engine.slate_settings,
        slatedb_adapter_factory=config.engine.slatedb_adapter_factory,
    )


def _run_partition_writes(
    *,
    partitioned_rdd: Any,
    runtime: _PartitionWriteConfig,
    num_dbs: int,
) -> tuple[list[_ShardAttemptResult], list[RequiredShardMeta], int]:
    """Execute partition writers and deterministically select winning shard attempts."""

    write_started = time.perf_counter()
    json_lines = partitioned_rdd.mapPartitionsWithIndex(
        lambda db_id, items: _write_one_shard_partition(db_id, items, runtime)
    ).collect()
    write_duration_ms = int((time.perf_counter() - write_started) * 1000)

    attempts = [_parse_attempt_line(line) for line in json_lines]
    winners = _select_winners(attempts, num_dbs=num_dbs)
    return attempts, winners, write_duration_ms


def _build_manifest_artifact(
    *,
    config: SlateDbConfig,
    run_id: str,
    resolved_sharding: ShardingSpec,
    winners: list[RequiredShardMeta],
) -> ManifestArtifact:
    """Build manifest bytes using configured builder and custom fields."""

    required_build = RequiredBuildMeta(
        run_id=run_id,
        created_at=_utc_now_iso(),
        num_dbs=config.num_dbs,
        s3_prefix=config.s3_prefix,
        key_col=config.key_col,
        sharding=_manifest_safe_sharding(resolved_sharding),
        db_path_template=config.output.db_path_template,
        tmp_prefix=config.output.tmp_prefix,
        key_encoding=KEY_ENCODING,
    )

    builder = config.manifest.manifest_builder or JsonManifestBuilder()
    for key, value in config.manifest.custom_manifest_fields.items():
        builder.add_custom_field(key, value)

    try:
        return builder.build(
            required_build=required_build,
            shards=winners,
            custom_fields=config.manifest.custom_manifest_fields,
        )
    except Exception as exc:  # pragma: no cover - custom builder surface
        raise ManifestBuildError("Failed to build manifest artifact") from exc


def _publish_manifest_and_current(
    *,
    config: SlateDbConfig,
    run_id: str,
    artifact: ManifestArtifact,
) -> tuple[str, str | None]:
    """Publish manifest and CURRENT pointer, enforcing publish ordering semantics."""

    publisher = config.manifest.publisher or DefaultS3Publisher(
        config.s3_prefix,
        manifest_name=config.manifest.manifest_name,
        current_name=config.manifest.current_name,
        s3_client_config=config.manifest.s3_client_config,
    )

    try:
        manifest_ref = publisher.publish_manifest(
            name=config.manifest.manifest_name,
            artifact=artifact,
            run_id=run_id,
        )
    except Exception as exc:  # pragma: no cover - runtime publisher failures
        raise PublishManifestError("Failed to publish manifest") from exc

    current_artifact = _build_current_artifact(
        manifest_ref=manifest_ref,
        manifest_content_type=artifact.content_type,
        run_id=run_id,
    )

    try:
        current_ref = publisher.publish_current(
            name=config.manifest.current_name,
            artifact=current_artifact,
        )
    except Exception as exc:  # pragma: no cover - runtime publisher failures
        raise PublishCurrentError(
            f"Manifest already published at {manifest_ref}; failed publishing CURRENT"
        ) from exc

    return manifest_ref, current_ref


def _assemble_build_result(
    *,
    run_id: str,
    winners: list[RequiredShardMeta],
    artifact: ManifestArtifact,
    manifest_ref: str,
    current_ref: str | None,
    attempts: list[_ShardAttemptResult],
    shard_duration_ms: int,
    write_duration_ms: int,
    manifest_duration_ms: int,
    started: float,
) -> BuildResult:
    """Assemble final BuildResult and fixed-schema BuildStats."""

    total_duration_ms = int((time.perf_counter() - started) * 1000)
    stats = BuildStats(
        durations=BuildDurations(
            sharding_ms=shard_duration_ms,
            write_ms=write_duration_ms,
            manifest_ms=manifest_duration_ms,
            total_ms=total_duration_ms,
        ),
        num_attempt_results=len(attempts),
        num_winners=len(winners),
        rows_written=sum(winner.row_count for winner in winners),
    )

    return BuildResult(
        run_id=run_id,
        winners=winners,
        manifest_artifact=artifact,
        manifest_ref=manifest_ref,
        current_ref=current_ref,
        stats=stats,
    )


def _write_one_shard_partition(
    db_id: int,
    rows_iter: Iterator[tuple[int, Any]],
    runtime: _PartitionWriteConfig,
) -> Iterator[str]:
    """Write exactly one shard from one partition and emit one JSON result line."""

    ctx = TaskContext.get()
    attempt = int(ctx.attemptNumber()) if ctx else 0
    stage_id = int(ctx.stageId()) if ctx else None
    task_attempt_id = int(ctx.taskAttemptId()) if ctx else None

    db_rel_path = runtime.db_path_template.format(db_id=db_id)
    db_url = _join_s3(
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

    adapter_factory = runtime.slatedb_adapter_factory or default_adapter_factory
    adapter = adapter_factory()

    partition_started = time.perf_counter()
    row_count = 0
    min_key: int | str | None = None
    max_key: int | str | None = None
    checkpoint_id: str | None = None
    batch: list[tuple[bytes, bytes]] = []

    db = adapter.open(
        local_dir=local_dir,
        db_url=db_url,
        env_file=runtime.slate_env_file,
        settings=runtime.slate_settings,
    )

    try:
        for _, row in rows_iter:
            key_value = row[runtime.key_col]
            key_bytes = encode_key(key_value, encoding=KEY_ENCODING)
            value_bytes = runtime.value_spec.encode(row)

            batch.append((key_bytes, value_bytes))
            row_count += 1
            min_key, max_key = _update_min_max(min_key, max_key, key_value)

            if len(batch) >= runtime.batch_size:
                adapter.write_pairs(db, batch)
                batch.clear()

        if batch:
            adapter.write_pairs(db, batch)
            batch.clear()

        adapter.flush_wal_if_supported(db)
        checkpoint_id = adapter.create_checkpoint_if_supported(db)
    except Exception as exc:  # pragma: no cover - worker runtime failure surface
        raise SlatedbSparkShardedError(
            f"Shard write failed for db_id={db_id}, attempt={attempt}: {exc}"
        ) from exc
    finally:
        adapter.close_if_supported(db)

    writer_info = {
        "stage_id": stage_id,
        "task_attempt_id": task_attempt_id,
        "attempt": attempt,
        "duration_ms": int((time.perf_counter() - partition_started) * 1000),
    }

    payload = {
        "db_id": db_id,
        "db_url": db_url,
        "attempt": attempt,
        "row_count": row_count,
        "min_key": _normalize_key(min_key),
        "max_key": _normalize_key(max_key),
        "checkpoint_id": checkpoint_id,
        "writer_info": writer_info,
    }
    yield json.dumps(payload, sort_keys=True)


def _update_min_max(
    min_key: int | str | None,
    max_key: int | str | None,
    key: Any,
) -> tuple[int | str | None, int | str | None]:
    if key is None:
        return min_key, max_key

    if min_key is None:
        min_key = key
    else:
        min_key = key if key < min_key else min_key

    if max_key is None:
        max_key = key
    else:
        max_key = key if key > max_key else max_key

    return min_key, max_key


def _normalize_key(key: Any) -> int | str | None:
    if key is None:
        return None
    if isinstance(key, (int, str)):
        return key
    return str(key)


def _parse_attempt_line(line: str) -> _ShardAttemptResult:
    payload = json.loads(line)
    return _ShardAttemptResult(
        db_id=int(payload["db_id"]),
        db_url=str(payload["db_url"]),
        attempt=int(payload["attempt"]),
        row_count=int(payload.get("row_count", 0)),
        min_key=payload.get("min_key"),
        max_key=payload.get("max_key"),
        checkpoint_id=payload.get("checkpoint_id"),
        writer_info=dict(payload.get("writer_info") or {}),
    )


def _select_winners(
    attempts: list[_ShardAttemptResult],
    *,
    num_dbs: int,
) -> list[RequiredShardMeta]:
    grouped: dict[int, list[_ShardAttemptResult]] = defaultdict(list)
    for item in attempts:
        grouped[item.db_id].append(item)

    expected_ids = set(range(num_dbs))
    got_ids = set(grouped.keys())
    if got_ids != expected_ids:
        missing = sorted(expected_ids - got_ids)
        extra = sorted(got_ids - expected_ids)
        raise ShardCoverageError(
            f"Shard coverage mismatch; missing={missing}, extra={extra}"
        )

    winners: list[RequiredShardMeta] = []
    for db_id in range(num_dbs):
        winner = sorted(grouped[db_id], key=_winner_sort_key)[0]
        winners.append(
            RequiredShardMeta(
                db_id=winner.db_id,
                db_url=winner.db_url,
                attempt=winner.attempt,
                row_count=winner.row_count,
                min_key=winner.min_key,
                max_key=winner.max_key,
                checkpoint_id=winner.checkpoint_id,
                writer_info=winner.writer_info,
            )
        )

    return winners


def _winner_sort_key(item: _ShardAttemptResult) -> tuple[int, int, str]:
    task_attempt_id = item.writer_info.get("task_attempt_id")
    if task_attempt_id is None:
        normalized_task_attempt_id = 2**63 - 1
    else:
        normalized_task_attempt_id = int(task_attempt_id)
    return (item.attempt, normalized_task_attempt_id, item.db_url)


def _build_current_artifact(
    *,
    manifest_ref: str,
    manifest_content_type: str,
    run_id: str,
) -> ManifestArtifact:
    pointer = CurrentPointer(
        manifest_ref=manifest_ref,
        manifest_content_type=manifest_content_type,
        run_id=run_id,
        updated_at=_utc_now_iso(),
    )
    payload = json.dumps(asdict(pointer), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return ManifestArtifact(payload=payload, content_type="application/json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_safe_sharding(sharding: ShardingSpec) -> ShardingSpec:
    return ShardingSpec(
        strategy=sharding.strategy,
        boundaries=list(sharding.boundaries)
        if sharding.boundaries is not None
        else None,
        approx_quantile_rel_error=sharding.approx_quantile_rel_error,
        custom_expr=sharding.custom_expr,
        custom_column_builder=None,
    )


def _join_s3(base: str, *parts: str) -> str:
    clean = [base.rstrip("/")]
    clean.extend(part.strip("/") for part in parts if part)
    return "/".join(clean)
