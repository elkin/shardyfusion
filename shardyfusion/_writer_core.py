"""Framework-agnostic core functions shared by all writer implementations."""

import logging
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from .config import WriteConfig
from .errors import (
    ConfigValidationError,
    PublishManifestError,
    ShardCoverageError,
)
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from .manifest_store import S3ManifestStore
from .metrics import MetricEvent
from .ordering import compare_ordered
from .routing import xxhash64_db_id  # SHARDING INVARIANT: direct import, not reimpl.
from .sharding_types import KeyEncoding, ShardingSpec, ShardingStrategy
from .type_defs import JsonObject, KeyLike

_logger = get_logger(__name__)


@dataclass(slots=True)
class ShardAttemptResult:
    db_id: int
    db_url: str
    attempt: int
    row_count: int
    min_key: int | str | None
    max_key: int | str | None
    checkpoint_id: str | None
    writer_info: JsonObject


@dataclass(slots=True)
class PartitionWriteOutcome:
    attempts: list[ShardAttemptResult]
    winners: list[RequiredShardMeta]
    write_duration_ms: int


def route_key(
    key: KeyLike,
    *,
    num_dbs: int,
    sharding: ShardingSpec,
    key_encoding: KeyEncoding,
) -> int:
    """Route a key to a shard db_id (non-Spark path)."""

    if sharding.strategy == ShardingStrategy.HASH:
        return xxhash64_db_id(key, num_dbs, key_encoding)
    if sharding.strategy == ShardingStrategy.RANGE:
        if sharding.boundaries is None:
            raise ConfigValidationError(
                "Range sharding without explicit boundaries requires a framework writer (Spark or Dask)."
            )
        return bisect_right(sharding.boundaries, key)
    raise ConfigValidationError(
        f"Sharding strategy {sharding.strategy!r} not supported in non-Spark writers."
    )


def select_winners(
    attempts: list[ShardAttemptResult],
    *,
    num_dbs: int,
) -> list[RequiredShardMeta]:
    grouped: dict[int, list[ShardAttemptResult]] = defaultdict(list)
    for item in attempts:
        grouped[item.db_id].append(item)

    expected_ids = set(range(num_dbs))
    got_ids = set(grouped.keys())
    if got_ids != expected_ids:
        missing = sorted(expected_ids - got_ids)
        extra = sorted(got_ids - expected_ids)
        log_failure(
            "shard_coverage_mismatch",
            severity=FailureSeverity.CRITICAL,
            logger=_logger,
            missing_shards=missing,
            extra_shards=extra,
            expected_count=num_dbs,
            got_count=len(got_ids),
        )
        raise ShardCoverageError(
            f"Shard coverage mismatch; missing={missing}, extra={extra}"
        )

    winners: list[RequiredShardMeta] = []
    for db_id in range(num_dbs):
        winner = sorted(grouped[db_id], key=_winner_sort_key)[0]
        log_event(
            "winner_selected",
            level=logging.DEBUG,
            logger=_logger,
            db_id=winner.db_id,
            attempt=winner.attempt,
            db_url=winner.db_url,
        )
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


def _winner_sort_key(item: ShardAttemptResult) -> tuple[int, int, str]:
    task_attempt_id = item.writer_info.get("task_attempt_id")
    if task_attempt_id is None:
        normalized_task_attempt_id = 2**63 - 1
    elif isinstance(task_attempt_id, (int, float, str)):
        normalized_task_attempt_id = int(task_attempt_id)
    else:
        normalized_task_attempt_id = 2**63 - 1
    return (item.attempt, normalized_task_attempt_id, item.db_url)


def publish_to_store(
    *,
    config: WriteConfig,
    run_id: str,
    resolved_sharding: ShardingSpec,
    winners: list[RequiredShardMeta],
    key_col: str = "_key",
    started: float = 0.0,
) -> str:
    """Build required metadata and publish via the configured ManifestStore."""

    required_build = RequiredBuildMeta(
        run_id=run_id,
        created_at=_utc_now(),
        num_dbs=config.num_dbs,
        s3_prefix=config.s3_prefix,
        key_col=key_col,
        sharding=manifest_safe_sharding(resolved_sharding),
        db_path_template=config.output.db_path_template,
        tmp_prefix=config.output.tmp_prefix,
        key_encoding=config.key_encoding,
    )

    store = config.manifest.store or S3ManifestStore(
        config.s3_prefix,
        manifest_builder=config.manifest.manifest_builder,
        s3_client_config=config.manifest.s3_client_config,
        metrics_collector=config.metrics_collector,
    )

    mc = config.metrics_collector

    try:
        manifest_ref = store.publish(
            run_id=run_id,
            required_build=required_build,
            shards=winners,
            custom=config.manifest.custom_manifest_fields,
        )
    except Exception as exc:  # pragma: no cover - runtime store failures
        log_failure(
            "manifest_publish_failed",
            severity=FailureSeverity.ERROR,
            logger=_logger,
            error=exc,
            run_id=run_id,
            s3_prefix=config.s3_prefix,
            include_traceback=True,
        )
        raise PublishManifestError("Failed to publish manifest") from exc

    log_event(
        "manifest_published",
        logger=_logger,
        run_id=run_id,
        manifest_ref=manifest_ref,
    )
    if mc is not None:
        mc.emit(
            MetricEvent.MANIFEST_PUBLISHED,
            {
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )

    return manifest_ref


def assemble_build_result(
    *,
    run_id: str,
    winners: list[RequiredShardMeta],
    manifest_ref: str,
    attempts: list[ShardAttemptResult],
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
        manifest_ref=manifest_ref,
        stats=stats,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def manifest_safe_sharding(sharding: ShardingSpec) -> ManifestShardingSpec:
    return ManifestShardingSpec(
        strategy=sharding.strategy,
        boundaries=list(sharding.boundaries)
        if sharding.boundaries is not None
        else None,
        approx_quantile_rel_error=sharding.approx_quantile_rel_error,
        custom_expr=sharding.custom_expr,
    )


def update_min_max(
    min_key: KeyLike | None,
    max_key: KeyLike | None,
    key: KeyLike | None,
) -> tuple[KeyLike | None, KeyLike | None]:
    normalized_key = key
    if normalized_key is None:
        return min_key, max_key

    if min_key is None:
        min_key = normalized_key
    elif (
        compare_ordered(
            normalized_key,
            min_key,
            mismatch_message="Shard key type changed within partition: mixed int/str keys",
        )
        < 0
    ):
        min_key = normalized_key

    if max_key is None:
        max_key = normalized_key
    elif (
        compare_ordered(
            normalized_key,
            max_key,
            mismatch_message="Shard key type changed within partition: mixed int/str keys",
        )
        > 0
    ):
        max_key = normalized_key

    return min_key, max_key
