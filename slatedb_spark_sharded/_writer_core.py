"""Framework-agnostic core functions shared by all writer implementations."""

import json
import time
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import WriteConfig
from .errors import (
    ConfigValidationError,
    ManifestBuildError,
    PublishCurrentError,
    PublishManifestError,
    ShardCoverageError,
)
from .logging import FailureSeverity, log_failure
from .manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    CurrentPointer,
    JsonManifestBuilder,
    ManifestArtifact,
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from .ordering import compare_ordered
from .publish import DefaultS3Publisher
from .routing import _xxhash64_db_id  # SHARDING INVARIANT: direct import, not reimpl.
from .sharding_types import KeyEncoding, ShardingSpec, ShardingStrategy
from .type_defs import JsonObject, KeyLike


@dataclass(slots=True)
class _ShardAttemptResult:
    db_id: int
    db_url: str
    attempt: int
    row_count: int
    min_key: int | str | None
    max_key: int | str | None
    checkpoint_id: str | None
    writer_info: JsonObject


@dataclass(slots=True)
class _PartitionWriteOutcome:
    attempts: list[_ShardAttemptResult]
    winners: list[RequiredShardMeta]
    write_duration_ms: int


@dataclass(slots=True)
class _PublishResult:
    manifest_ref: str
    current_ref: str | None


def _route_key(
    key: KeyLike,
    *,
    num_dbs: int,
    sharding: ShardingSpec,
    key_encoding: KeyEncoding,
) -> int:
    """Route a key to a shard db_id (non-Spark path)."""

    if sharding.strategy == ShardingStrategy.HASH:
        return _xxhash64_db_id(key, num_dbs, key_encoding)
    if sharding.strategy == ShardingStrategy.RANGE:
        if sharding.boundaries is None:
            raise ConfigValidationError(
                "Range sharding without explicit boundaries requires Spark."
            )
        return bisect_right(sharding.boundaries, key)
    raise ConfigValidationError(
        f"Sharding strategy {sharding.strategy!r} not supported in non-Spark writers."
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
        log_failure(
            "shard_coverage_mismatch",
            severity=FailureSeverity.CRITICAL,
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
    elif isinstance(task_attempt_id, (int, float, str)):
        normalized_task_attempt_id = int(task_attempt_id)
    else:
        normalized_task_attempt_id = 2**63 - 1
    return (item.attempt, normalized_task_attempt_id, item.db_url)


def _build_manifest_artifact(
    *,
    config: WriteConfig,
    run_id: str,
    resolved_sharding: ShardingSpec,
    winners: list[RequiredShardMeta],
    key_col: str = "_key",
) -> ManifestArtifact:
    """Build manifest bytes using configured builder and custom fields."""

    required_build = RequiredBuildMeta(
        run_id=run_id,
        created_at=_utc_now_iso(),
        num_dbs=config.num_dbs,
        s3_prefix=config.s3_prefix,
        key_col=key_col,
        sharding=_manifest_safe_sharding(resolved_sharding),
        db_path_template=config.output.db_path_template,
        tmp_prefix=config.output.tmp_prefix,
        key_encoding=config.key_encoding,
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
        log_failure(
            "manifest_build_failed",
            severity=FailureSeverity.ERROR,
            error=exc,
            run_id=run_id,
            builder_type=type(builder).__name__,
            include_traceback=True,
        )
        raise ManifestBuildError("Failed to build manifest artifact") from exc


def _publish_manifest_and_current(
    *,
    config: WriteConfig,
    run_id: str,
    artifact: ManifestArtifact,
) -> _PublishResult:
    """Publish manifest and CURRENT pointer."""

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
        log_failure(
            "manifest_publish_failed",
            severity=FailureSeverity.ERROR,
            error=exc,
            run_id=run_id,
            s3_prefix=config.s3_prefix,
            include_traceback=True,
        )
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
        log_failure(
            "current_publish_failed",
            severity=FailureSeverity.CRITICAL,
            error=exc,
            run_id=run_id,
            manifest_ref=manifest_ref,
            include_traceback=True,
        )
        raise PublishCurrentError(
            f"Manifest already published at {manifest_ref}; failed publishing CURRENT"
        ) from exc

    return _PublishResult(manifest_ref=manifest_ref, current_ref=current_ref)


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
    payload = json.dumps(
        pointer.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return ManifestArtifact(payload=payload, content_type="application/json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest_safe_sharding(sharding: ShardingSpec) -> ManifestShardingSpec:
    return ManifestShardingSpec(
        strategy=sharding.strategy,
        boundaries=list(sharding.boundaries)
        if sharding.boundaries is not None
        else None,
        approx_quantile_rel_error=sharding.approx_quantile_rel_error,
        custom_expr=sharding.custom_expr,
    )


def _join_s3(base: str, *parts: str) -> str:
    clean = [base.rstrip("/")]
    clean.extend(part.strip("/") for part in parts if part)
    return "/".join(clean)


def _update_min_max(
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
