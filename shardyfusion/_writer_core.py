"""Framework-agnostic core functions shared by all writer implementations."""

import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from operator import index as operator_index
from typing import Any, Literal, Protocol, SupportsIndex, cast

from .config import WriteConfig, vector_metric_to_str
from .errors import (
    ConfigValidationError,
    PublishCurrentError,
    PublishManifestError,
    ShardAssignmentError,
    ShardCoverageError,
)
from .logging import FailureSeverity, get_logger, log_event, log_failure
from .manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
    WriterInfo,
)
from .manifest_store import S3ManifestStore
from .metrics import MetricEvent, MetricsCollector
from .ordering import compare_ordered
from .routing import hash_db_id
from .sharding_types import (
    RoutingValue,
    ShardingSpec,
    ShardingStrategy,
    validate_routing_values,
)
from .storage import (
    ObstoreBackend,
    StorageBackend,
    create_s3_store,
    join_s3,
    parse_s3_url,
)
from .type_defs import KeyInput

_logger = get_logger(__name__)

_PUBLISH_CURRENT_MAX_RETRIES = 3


class _RowLike(Protocol):
    def __getitem__(self, key: str) -> object: ...


def _normalize_vector_id(raw_id: object) -> int | str:
    """Normalize framework scalar wrappers to the logical vector ID contract."""
    if isinstance(raw_id, (int, str)):
        return raw_id
    try:
        return operator_index(cast(SupportsIndex, raw_id))
    except TypeError:
        return str(raw_id)


@dataclass(slots=True, frozen=True)
class VectorColumnMapping:
    """Column-based vector extraction config for distributed writers."""

    vector_col: str
    id_col: str | None = None
    payload_cols: Sequence[str] | None = None


@dataclass(slots=True)
class ShardAttemptResult:
    db_id: int
    db_url: str | None
    attempt: int
    row_count: int
    min_key: KeyInput | None
    max_key: KeyInput | None
    checkpoint_id: str | None
    writer_info: WriterInfo
    db_bytes: int = 0
    all_attempt_urls: tuple[str, ...] = ()


def empty_shard_result(
    db_id: int, attempt: int = 0, writer_info: WriterInfo | None = None
) -> ShardAttemptResult:
    """Construct a metadata-only ShardAttemptResult for an empty (unwritten) shard."""
    return ShardAttemptResult(
        db_id=db_id,
        db_url=None,
        attempt=attempt,
        row_count=0,
        min_key=None,
        max_key=None,
        checkpoint_id=None,
        writer_info=writer_info or WriterInfo(),
        db_bytes=0,
        all_attempt_urls=(),
    )


@dataclass(slots=True)
class PartitionWriteOutcome:
    num_attempts: int
    winners: list[RequiredShardMeta]
    all_attempt_urls: list[str]
    write_duration_ms: int


_cel_imports: tuple[Any, ...] | None = None


def _get_cel_imports() -> tuple[Any, ...]:
    global _cel_imports
    if _cel_imports is None:
        from .cel import _compile_cel_cached, route_cel

        _cel_imports = (_compile_cel_cached, route_cel)
    return _cel_imports


def route_key(
    key: KeyInput,
    *,
    num_dbs: int | None,
    sharding: ShardingSpec,
    routing_context: dict[str, object] | None = None,
    cel_lookup: dict[RoutingValue, int] | None = None,
) -> int:
    """Route a key to a shard db_id (shared by all writer paths)."""

    if sharding.strategy == ShardingStrategy.HASH:
        assert num_dbs is not None, "num_dbs required for HASH routing"
        return hash_db_id(key, num_dbs, sharding.hash_algorithm)
    if sharding.strategy == ShardingStrategy.CEL:
        _compile_cel_cached, route_cel = _get_cel_imports()
        assert sharding.cel_expr is not None and sharding.cel_columns is not None
        columns_key = tuple(sorted(sharding.cel_columns.items()))
        cached = _compile_cel_cached(sharding.cel_expr, columns_key)
        ctx = routing_context if routing_context is not None else {"key": key}
        try:
            return route_cel(
                cached,
                ctx,
                sharding.routing_values,
                cel_lookup,
            )
        except Exception as exc:
            from .cel import UnknownRoutingTokenError

            if isinstance(exc, UnknownRoutingTokenError):
                raise ShardAssignmentError(str(exc)) from exc
            raise
    raise ConfigValidationError(
        f"Sharding strategy {sharding.strategy!r} not supported."
    )


def resolve_num_dbs(config: WriteConfig, count_fn: Callable[[], int]) -> int | None:
    """Resolve num_dbs from config or max_keys_per_shard.

    Returns ``None`` for CEL when shard cardinality must be discovered from data.

    Args:
        config: Write configuration.
        count_fn: Framework-specific callable returning the row count
            (only called if max_keys_per_shard is set).
    """
    import math

    if config.num_dbs is not None and config.num_dbs > 0:
        return config.num_dbs

    if (
        config.sharding.strategy == ShardingStrategy.CEL
        and config.sharding.routing_values is not None
    ):
        return max(1, len(config.sharding.routing_values))

    if config.sharding.max_keys_per_shard is not None:
        count = count_fn()
        if count == 0:
            return 1
        return max(1, math.ceil(count / config.sharding.max_keys_per_shard))

    # CEL: will discover after add_db_id_column
    return None


def resolve_distributed_vector_fn(
    *,
    config: WriteConfig,
    key_col: str,
    vector_fn: Callable[[_RowLike], tuple[int | str, Any, dict[str, Any] | None]]
    | None,
    vector_columns: VectorColumnMapping | None,
) -> Callable[[_RowLike], tuple[int | str, Any, dict[str, Any] | None]] | None:
    """Resolve vector extraction for DataFrame-based writers.

    Matches Python writer validation semantics:
    - ``vector_fn`` requires ``config.vector_spec``.
    - ``config.vector_spec`` requires either ``vector_fn`` or vector column mapping.
    """
    if vector_fn is not None and config.vector_spec is None:
        raise ConfigValidationError("vector_fn requires config.vector_spec to be set")

    if config.vector_spec is None:
        return None

    if vector_fn is not None:
        return vector_fn

    mapping = vector_columns
    vector_col = (
        mapping.vector_col if mapping is not None else config.vector_spec.vector_col
    )
    if vector_col is None:
        raise ConfigValidationError(
            "config.vector_spec is set but no vector_fn was provided "
            "and no vector column mapping is available. Either provide vector_fn, "
            "vector_columns, or set vector_spec.vector_col."
        )

    id_col = mapping.id_col if mapping is not None else key_col
    if id_col is None:
        id_col = key_col
    payload_cols = tuple(mapping.payload_cols or ()) if mapping is not None else ()

    def _auto_vector_fn(
        row: _RowLike,
    ) -> tuple[int | str, Any, dict[str, Any] | None]:
        raw_id = row[id_col]
        # Distributed row APIs often surface numpy/pandas scalar wrappers; normalize
        # them once here so lower-level vector adapters only see logical int|str IDs.
        vec_id = _normalize_vector_id(raw_id)
        vector = row[vector_col]
        payload = {col: row[col] for col in payload_cols} if payload_cols else None
        return (vec_id, vector, payload)

    return _auto_vector_fn


def discover_cel_num_dbs(
    distinct_db_ids: set[int],
) -> int:
    """Discover num_dbs from CEL-assigned db_ids and validate they are consecutive.

    CEL expressions must produce consecutive 0-based shard IDs directly.

    Returns:
        num_dbs (= max(db_id) + 1 = len(distinct_db_ids) for consecutive IDs).

    Raises:
        ShardAssignmentError: If db_ids are not consecutive starting from 0.
    """
    from .errors import ShardAssignmentError

    if not distinct_db_ids:
        return 1

    max_id = max(distinct_db_ids)
    expected = max_id + 1

    if len(distinct_db_ids) != expected:
        raise ShardAssignmentError(
            f"CEL expression produced non-consecutive shard IDs: "
            f"distinct count={len(distinct_db_ids)}, max+1={expected}. "
            f"CEL expressions must produce consecutive 0-based IDs "
            f"(e.g. shard_hash(key) % N)."
        )

    return expected


def build_categorical_routing_values(
    values: Iterable[RoutingValue],
) -> list[RoutingValue]:
    """Return deterministic sorted distinct routing values for categorical CEL."""

    raw_values = list(values)
    if not raw_values:
        return []
    try:
        validate_routing_values(raw_values, require_unique=False)
    except ValueError as exc:
        raise ConfigValidationError(str(exc)) from exc
    distinct = list(set(raw_values))
    try:
        return sorted(distinct)
    except TypeError as exc:
        raise ConfigValidationError(
            "Categorical CEL routing values must be orderable."
        ) from exc


def select_winners(
    attempts: Iterable[ShardAttemptResult],
    *,
    num_dbs: int,
) -> tuple[list[RequiredShardMeta], int, list[str]]:
    """Select winning attempt for each shard from an iterable of attempt results.

    Returns:
        Tuple of (winners list, total attempt count, all attempt URLs).
    """
    grouped: dict[int, list[ShardAttemptResult]] = defaultdict(list)
    num_attempts = 0
    all_attempt_urls: list[str] = []
    for item in attempts:
        grouped[item.db_id].append(item)
        num_attempts += 1
        if item.all_attempt_urls:
            all_attempt_urls.extend(item.all_attempt_urls)
        elif item.db_url is not None:
            all_attempt_urls.append(item.db_url)

    # Validate: no extra db_ids beyond expected range.
    # Missing db_ids are fine — they are empty shards omitted from the manifest.
    expected_ids = set(range(num_dbs))
    got_ids = set(grouped.keys())
    extra = sorted(got_ids - expected_ids)
    if extra:
        log_failure(
            "shard_coverage_mismatch",
            severity=FailureSeverity.CRITICAL,
            logger=_logger,
            extra_shards=extra,
            expected_count=num_dbs,
            got_count=len(got_ids),
        )
        raise ShardCoverageError(f"Shard coverage mismatch; extra={extra}")

    winners: list[RequiredShardMeta] = []
    for db_id in sorted(got_ids):
        winner = sorted(grouped[db_id], key=_winner_sort_key)[0]
        # Skip empty shards (db_url=None) — they are not written to the manifest.
        if winner.db_url is None:
            continue
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
                db_bytes=winner.db_bytes,
            )
        )

    return winners, num_attempts, all_attempt_urls


def _winner_sort_key(item: ShardAttemptResult) -> tuple[int, int, str]:
    tid = item.writer_info.task_attempt_id
    normalized = tid if tid is not None else 2**63 - 1
    return (item.attempt, normalized, item.db_url or "")


def wrap_factory_for_vector(factory: Any, config: WriteConfig) -> Any:
    """Wrap KV factory for unified KV+vector mode when needed."""
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory

    vs = config.vector_spec
    assert vs is not None

    if getattr(factory, "supports_vector_writes", False) is True:
        return factory
    if isinstance(factory, SqliteVecFactory):
        return factory

    from shardyfusion.composite_adapter import CompositeFactory

    try:
        from shardyfusion.vector.adapters.lancedb_adapter import LanceDbWriterFactory

        vector_factory = LanceDbWriterFactory(
            s3_connection_options=config.s3_connection_options,
            credential_provider=config.credential_provider,
        )
    except ImportError as exc:
        raise ConfigValidationError(
            "Unified KV+vector mode with SlateDB requires the 'vector-lancedb' "
            "extra. Install with: pip install shardyfusion[vector-lancedb]"
        ) from exc

    return CompositeFactory(
        kv_factory=factory,
        vector_factory=vector_factory,
        vector_spec=vs,
    )


def detect_vector_backend(factory: Any) -> str:
    from shardyfusion.sqlite_vec_adapter import SqliteVecFactory

    if isinstance(factory, SqliteVecFactory):
        return "sqlite-vec"
    return "lancedb"


def detect_kv_backend(factory: Any) -> str:
    from shardyfusion.composite_adapter import CompositeFactory

    actual = factory.kv_factory if isinstance(factory, CompositeFactory) else factory
    try:
        from shardyfusion.sqlite_vec_adapter import SqliteVecFactory

        if isinstance(actual, SqliteVecFactory):
            return "sqlite-vec"
    except ImportError:
        pass
    try:
        from shardyfusion.sqlite_adapter import SqliteFactory

        if isinstance(actual, SqliteFactory):
            return "sqlite"
    except ImportError:
        pass
    return "slatedb"


def inject_vector_manifest_fields(config: WriteConfig, factory: Any) -> None:
    """Add unified vector metadata to manifest custom fields."""
    vs = config.vector_spec
    assert vs is not None
    vector_meta: dict[str, Any] = {
        "dim": vs.dim,
        "metric": vector_metric_to_str(vs.metric),
        "index_type": vs.index_type,
        "quantization": vs.quantization,
        "unified": True,
        "backend": detect_vector_backend(factory),
        "kv_backend": detect_kv_backend(factory),
    }
    if vs.index_params:
        vector_meta["index_params"] = vs.index_params
    custom_fields = dict(config.manifest.custom_manifest_fields)
    custom_fields["vector"] = vector_meta
    config.manifest.custom_manifest_fields = custom_fields


def publish_to_store(
    *,
    config: WriteConfig,
    run_id: str,
    resolved_sharding: ShardingSpec,
    winners: list[RequiredShardMeta],
    key_col: str = "_key",
    started: float = 0.0,
    num_dbs: int = 0,
) -> str:
    """Build required metadata and publish via the configured ManifestStore.

    Args:
        num_dbs: Resolved shard count (must be > 0). Writers must always
            pass the resolved value since ``config.num_dbs`` may be ``None``
            for CEL or max_keys_per_shard modes.
    """

    resolved_num_dbs = num_dbs if num_dbs > 0 else config.num_dbs
    assert resolved_num_dbs is not None and resolved_num_dbs > 0, (
        "num_dbs must be resolved to a positive integer before publishing"
    )
    required_build = RequiredBuildMeta(
        run_id=run_id,
        created_at=_utc_now(),
        num_dbs=resolved_num_dbs,
        s3_prefix=config.s3_prefix,
        key_col=key_col,
        sharding=manifest_safe_sharding(resolved_sharding),
        db_path_template=config.output.db_path_template,
        shard_prefix=config.output.shard_prefix,
        format_version=_manifest_format_version_for_sharding(resolved_sharding),
        key_encoding=config.key_encoding,
    )

    if config.manifest.store is not None:
        store = config.manifest.store
    else:
        credentials = config.manifest.credential_provider or config.credential_provider
        conn_opts = (
            config.manifest.s3_connection_options or config.s3_connection_options
        )
        bucket, _ = parse_s3_url(config.s3_prefix)
        s3_store = create_s3_store(
            bucket=bucket,
            credentials=credentials.resolve() if credentials else None,
            connection_options=conn_opts,
        )
        backend = ObstoreBackend(s3_store)
        store = S3ManifestStore(
            backend,
            config.s3_prefix,
        )

    mc = config.metrics_collector

    try:
        manifest_ref = store.publish(
            run_id=run_id,
            required_build=required_build,
            shards=winners,
            custom=config.manifest.custom_manifest_fields,
        )
    except PublishCurrentError as exc:
        # Manifest is already on S3 — retry the pointer-set (idempotent).
        manifest_ref_from_exc = exc.manifest_ref
        if manifest_ref_from_exc is not None:
            for retry in range(_PUBLISH_CURRENT_MAX_RETRIES):
                log_failure(
                    "publish_current_retry",
                    severity=FailureSeverity.TRANSIENT,
                    logger=_logger,
                    error=exc,
                    run_id=run_id,
                    manifest_ref=manifest_ref_from_exc,
                    retry=retry + 1,
                )
                try:
                    store.set_current(manifest_ref_from_exc)
                    manifest_ref = manifest_ref_from_exc
                    break
                except Exception:
                    if retry == _PUBLISH_CURRENT_MAX_RETRIES - 1:
                        raise PublishManifestError(
                            "Failed to set CURRENT pointer after retries"
                        ) from exc
                    time.sleep(1.0 * (2**retry))
            else:
                raise  # pragma: no cover
        else:
            raise
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
    run_record_ref: str | None,
    num_attempts: int,
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
        num_attempt_results=num_attempts,
        num_winners=len(winners),
        rows_written=sum(winner.row_count for winner in winners),
    )

    return BuildResult(
        run_id=run_id,
        winners=winners,
        manifest_ref=manifest_ref,
        stats=stats,
        run_record_ref=run_record_ref,
    )


def cleanup_losers(
    all_attempt_urls: list[str],
    winners: list[RequiredShardMeta],
    *,
    backend: StorageBackend | None = None,
    metrics_collector: MetricsCollector | None = None,
) -> int:
    """Delete temp databases for non-winning attempts.

    Best-effort: logs errors but does not raise.

    Returns:
        Total number of S3 objects deleted across all losing attempts.
    """

    if not all_attempt_urls:
        return 0

    winner_urls = {w.db_url for w in winners if w.db_url is not None}
    total_deleted = 0
    num_losers = 0

    if backend is None:
        from .storage import parse_s3_url

        bucket, _ = parse_s3_url(all_attempt_urls[0])
        store = create_s3_store(bucket=bucket)
        backend = ObstoreBackend(store)

    for url in all_attempt_urls:
        if url not in winner_urls:
            num_losers += 1
            try:
                deleted = backend.delete_prefix(url)
            except Exception as exc:
                log_failure(
                    "loser_cleanup_failed",
                    severity=FailureSeverity.TRANSIENT,
                    logger=_logger,
                    error=exc,
                    db_url=url,
                )
                deleted = 0
            total_deleted += deleted
            if deleted > 0:
                log_event(
                    "loser_attempt_cleaned",
                    level=logging.DEBUG,
                    logger=_logger,
                    db_url=url,
                    objects_deleted=deleted,
                )

    if total_deleted > 0:
        log_event(
            "losers_cleanup_completed",
            logger=_logger,
            total_objects_deleted=total_deleted,
            num_losers=num_losers,
        )

    return total_deleted


CleanupKind = Literal["stale_attempt", "old_run"]


@dataclass(slots=True)
class CleanupAction:
    """One cleanup operation performed (or planned in dry-run) by the cleanup command."""

    kind: CleanupKind
    prefix_url: str
    db_id: int | None  # None for old_run
    run_id: str
    objects_deleted: int  # 0 in dry-run mode


def cleanup_stale_attempts(
    manifest: ParsedManifest,
    *,
    backend: StorageBackend | None = None,
    dry_run: bool = False,
) -> list[CleanupAction]:
    """Find and delete non-winning attempt directories for the current manifest's run.

    For each shard in the manifest, lists all ``attempt=XX/`` directories
    under the shard's ``db=NNNNN/`` prefix and removes any that don't match
    the winning ``db_url``.

    Returns a list of :class:`CleanupAction` describing what was (or would be) deleted.
    """

    actions: list[CleanupAction] = []
    build = manifest.required_build
    winner_urls = {
        shard.db_url.rstrip("/")
        for shard in manifest.shards
        if shard.db_url is not None
    }

    if backend is None:
        from .storage import parse_s3_url

        bucket, _ = parse_s3_url(build.s3_prefix)
        store = create_s3_store(bucket=bucket)
        backend = ObstoreBackend(store)

    for shard in manifest.shards:
        if shard.db_url is None:
            continue
        # Build the scan prefix: s3_prefix/shard_prefix/run_id=XXX/db=NNNNN/
        db_segment = build.db_path_template.format(db_id=shard.db_id)
        scan_prefix = join_s3(
            build.s3_prefix, build.shard_prefix, f"run_id={build.run_id}", db_segment
        )
        if not scan_prefix.endswith("/"):
            scan_prefix += "/"

        attempt_dirs = backend.list_prefixes(scan_prefix)
        for attempt_url in attempt_dirs:
            if attempt_url.rstrip("/") not in winner_urls:
                deleted = 0
                if not dry_run:
                    deleted = backend.delete_prefix(attempt_url)
                actions.append(
                    CleanupAction(
                        kind="stale_attempt",
                        prefix_url=attempt_url,
                        db_id=shard.db_id,
                        run_id=build.run_id,
                        objects_deleted=deleted,
                    )
                )

    return actions


def cleanup_old_runs(
    s3_prefix: str,
    shard_prefix: str,
    *,
    protected_run_ids: set[str],
    backend: StorageBackend | None = None,
    dry_run: bool = False,
) -> list[CleanupAction]:
    """Delete shard data for runs not in *protected_run_ids*.

    Lists all ``run_id=XXX/`` directories under ``s3_prefix/shard_prefix/``
    and removes any whose run_id is not in the protected set.

    Returns a list of :class:`CleanupAction` describing what was (or would be) deleted.
    """

    actions: list[CleanupAction] = []
    runs_prefix = join_s3(s3_prefix, shard_prefix)
    if not runs_prefix.endswith("/"):
        runs_prefix += "/"

    if backend is None:
        from .storage import parse_s3_url

        bucket, _ = parse_s3_url(s3_prefix)
        store = create_s3_store(bucket=bucket)
        backend = ObstoreBackend(store)

    run_dirs = backend.list_prefixes(runs_prefix)

    for run_dir in run_dirs:
        # Extract run_id from directory name like "s3://bucket/prefix/shards/run_id=abc123/"
        segment = run_dir.rstrip("/").rsplit("/", 1)[-1]
        if not segment.startswith("run_id="):
            continue
        run_id = segment[len("run_id=") :]

        if run_id not in protected_run_ids:
            deleted = 0
            if not dry_run:
                deleted = backend.delete_prefix(run_dir)
            actions.append(
                CleanupAction(
                    kind="old_run",
                    prefix_url=run_dir,
                    db_id=None,
                    run_id=run_id,
                    objects_deleted=deleted,
                )
            )

    return actions


def _utc_now() -> datetime:
    return datetime.now(UTC)


def manifest_safe_sharding(sharding: ShardingSpec) -> ManifestShardingSpec:
    return ManifestShardingSpec(
        strategy=sharding.strategy,
        routing_values=list(sharding.routing_values)
        if sharding.routing_values is not None
        else None,
        cel_expr=sharding.cel_expr,
        cel_columns=dict(sharding.cel_columns)
        if sharding.cel_columns is not None
        else None,
        hash_algorithm=sharding.hash_algorithm,
    )


def _manifest_format_version_for_sharding(sharding: ShardingSpec) -> int:
    del sharding  # All snapshots use the unified format_version=4
    return 4


def update_min_max(
    min_key: KeyInput | None,
    max_key: KeyInput | None,
    key: KeyInput | None,
) -> tuple[KeyInput | None, KeyInput | None]:
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
