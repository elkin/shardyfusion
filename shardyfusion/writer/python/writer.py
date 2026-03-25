"""Pure-Python iterator-based sharded writer (no Spark dependency)."""

import contextlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, TypeVar
from uuid import uuid4

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
from shardyfusion.errors import ConfigValidationError
from shardyfusion.logging import get_logger, log_event
from shardyfusion.manifest import BuildResult, WriterInfo
from shardyfusion.metrics import MetricEvent
from shardyfusion.run_registry import managed_run_record
from shardyfusion.serde import make_key_encoder
from shardyfusion.slatedb_adapter import DbAdapterFactory, SlateDbFactory
from shardyfusion.type_defs import KeyInput
from shardyfusion.writer.python._parallel_writer import (
    _DEFAULT_MAX_PARALLEL_SHARED_MEMORY_BYTES,
    _DEFAULT_MAX_PARALLEL_SHARED_MEMORY_BYTES_PER_WORKER,
    _make_db_url,
    _make_local_dir,
    _validate_parallel_shared_memory_limit,
    _write_parallel,
)

_logger = get_logger(__name__)

T = TypeVar("T")


@dataclass(slots=True)
class _SingleProcessState:
    adapters: dict[int, Any] = field(default_factory=dict)
    db_urls: dict[int, str] = field(default_factory=dict)
    batches: dict[int, list[tuple[bytes, bytes]]] = field(default_factory=dict)
    batch_byte_sizes: dict[int, int] = field(default_factory=dict)
    row_counts: dict[int, int] = field(default_factory=dict)
    min_keys: dict[int, KeyInput | None] = field(default_factory=dict)
    max_keys: dict[int, KeyInput | None] = field(default_factory=dict)
    checkpoint_ids: dict[int, str | None] = field(default_factory=dict)
    total_batched_items: int = 0
    total_batched_bytes: int = 0


def write_sharded(
    records: Iterable[T],
    config: WriteConfig,
    *,
    key_fn: Callable[[T], KeyInput],
    value_fn: Callable[[T], bytes],
    columns_fn: Callable[[T], dict[str, Any]] | None = None,
    parallel: bool = False,
    max_queue_size: int = 100,
    max_parallel_shared_memory_bytes: int | None = (
        _DEFAULT_MAX_PARALLEL_SHARED_MEMORY_BYTES
    ),
    max_parallel_shared_memory_bytes_per_worker: int | None = (
        _DEFAULT_MAX_PARALLEL_SHARED_MEMORY_BYTES_PER_WORKER
    ),
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    max_total_batched_items: int | None = None,
    max_total_batched_bytes: int | None = None,
) -> BuildResult:
    """Write an iterable of records into N sharded databases.

    Args:
        records: Iterable of records to write. Each record is passed to
            ``key_fn`` and ``value_fn`` to extract the key and value bytes.
        config: Write configuration (num_dbs, s3_prefix, sharding strategy, etc.).
            HASH and CEL sharding strategies are supported.
        key_fn: Callable that extracts the routing key from each record.
        value_fn: Callable that serializes each record to value bytes.
        parallel: If True, use one subprocess per shard (``multiprocessing.spawn``).
            If False (default), all shard adapters are open simultaneously in
            a single process.
        max_queue_size: Maximum per-shard queue depth in parallel mode.
            In shared-memory parallel mode this bounds control messages,
            not payload bytes.
        max_parallel_shared_memory_bytes: Global cap on outstanding shared-memory
            payload bytes in parallel mode. When exceeded, the parent blocks until
            workers reclaim segments. ``None`` disables the global cap.
        max_parallel_shared_memory_bytes_per_worker: Per-worker cap on outstanding
            shared-memory payload bytes in parallel mode. When exceeded, the parent
            blocks until that worker reclaims segments. ``None`` disables the
            per-worker cap.
        max_writes_per_second: Optional rate limit (token-bucket) for write ops/sec.
        max_write_bytes_per_second: Optional rate limit (token-bucket) for write bytes/sec.
        max_total_batched_items: Global cap on total buffered items across all shard
            batches (single-process mode only). When exceeded, the shard with the
            largest batch is flushed. Prevents OOM with many shards.
        max_total_batched_bytes: Global cap on total buffered bytes across all shard
            batches (single-process mode only). When exceeded, the shard with the
            largest byte footprint is flushed.

    Returns:
        BuildResult with manifest reference, shard metadata, and build statistics.

    Raises:
        ConfigValidationError: If configuration is invalid.
        ShardCoverageError: If partition results don't cover all expected shards.
        PublishManifestError: If manifest upload to S3 fails.
        PublishCurrentError: If CURRENT pointer upload fails (manifest already published).
        ShardyfusionError: If a worker process fails in parallel mode.
    """

    from shardyfusion.logging import LogContext

    _validate_parallel_shared_memory_limit(
        max_parallel_shared_memory_bytes,
        field_name="max_parallel_shared_memory_bytes",
    )
    _validate_parallel_shared_memory_limit(
        max_parallel_shared_memory_bytes_per_worker,
        field_name="max_parallel_shared_memory_bytes_per_worker",
    )
    num_dbs = _resolve_num_dbs_before_sharding(records, config)
    run_id = config.output.run_id or uuid4().hex
    started = time.perf_counter()
    mc = config.metrics_collector

    with (
        LogContext(run_id=run_id),
        managed_run_record(
            config=config,
            run_id=run_id,
            writer_type="python",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=config.num_dbs,
            s3_prefix=config.s3_prefix,
            strategy=config.sharding.strategy,
            key_encoding=config.key_encoding,
            writer_type="python",
        )
        if mc is not None:
            mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})

        factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory(
            credential_provider=config.credential_provider
        )

        if parallel:
            if num_dbs is None:
                raise ConfigValidationError(
                    "Parallel mode requires num_dbs > 0 to spawn workers. "
                    "CEL direct mode (num_dbs discovered from data) "
                    "is only supported in single-process mode."
                )
            attempts = _write_parallel(
                records=records,
                config=config,
                num_dbs=num_dbs,
                run_id=run_id,
                factory=factory,
                key_fn=key_fn,
                value_fn=value_fn,
                columns_fn=columns_fn,
                max_queue_size=max_queue_size,
                max_parallel_shared_memory_bytes=max_parallel_shared_memory_bytes,
                max_parallel_shared_memory_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
                max_writes_per_second=max_writes_per_second,
                max_write_bytes_per_second=max_write_bytes_per_second,
            )
        else:
            attempts, num_dbs = _write_single_process(
                records=records,
                config=config,
                num_dbs=num_dbs,
                run_id=run_id,
                factory=factory,
                key_fn=key_fn,
                value_fn=value_fn,
                columns_fn=columns_fn,
                max_writes_per_second=max_writes_per_second,
                max_write_bytes_per_second=max_write_bytes_per_second,
                max_total_batched_items=max_total_batched_items,
                max_total_batched_bytes=max_total_batched_bytes,
            )

        write_duration_ms = int((time.perf_counter() - started) * 1000)

        rows_written = sum(a.row_count for a in attempts)
        log_event(
            "shard_writes_completed",
            logger=_logger,
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

        winners, num_attempts, all_attempt_urls = select_winners(
            attempts, num_dbs=num_dbs
        )

        manifest_started = time.perf_counter()
        manifest_ref = publish_to_store(
            config=config,
            run_id=run_id,
            resolved_sharding=config.sharding,
            winners=winners,
            key_col="_key",
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
            shard_duration_ms=0,
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


def _resolve_num_dbs_before_sharding(
    records: Iterable[Any],
    config: WriteConfig,
) -> int | None:
    """Resolve num_dbs that can be determined before writing.

    Returns ``None`` for CEL (discovered during iteration from data).
    """
    from collections.abc import Sized

    from shardyfusion._writer_core import resolve_num_dbs

    if config.sharding.max_keys_per_shard is not None and not isinstance(
        records, Sized
    ):
        raise ConfigValidationError(
            "max_keys_per_shard requires a Sized input (e.g. list) "
            "so num_dbs can be computed from len(records). "
            "Provide num_dbs explicitly for non-Sized iterables."
        )

    return resolve_num_dbs(
        config,
        lambda: len(records) if isinstance(records, Sized) else 0,
    )


def _largest_batch_id(
    batches: list[list[tuple[bytes, bytes]]],
    batch_byte_sizes: list[int],
) -> int:
    """Return the db_id with the largest batch (by byte size, then item count)."""
    best = 0
    best_bytes = batch_byte_sizes[0]
    best_items = len(batches[0])
    for i in range(1, len(batches)):
        i_bytes = batch_byte_sizes[i]
        i_items = len(batches[i])
        if (i_bytes, i_items) > (best_bytes, best_items):
            best = i
            best_bytes = i_bytes
            best_items = i_items
    return best


def _ensure_single_process_shard(state: _SingleProcessState, *, db_id: int) -> None:
    if db_id in state.batches:
        return
    state.batches[db_id] = []
    state.batch_byte_sizes[db_id] = 0
    state.row_counts[db_id] = 0
    state.min_keys[db_id] = None
    state.max_keys[db_id] = None


def _ensure_single_process_adapter(
    state: _SingleProcessState,
    *,
    db_id: int,
    stack: contextlib.ExitStack,
    config: WriteConfig,
    run_id: str,
    attempt: int,
    factory: DbAdapterFactory,
) -> None:
    if db_id in state.adapters:
        return
    db_url = _make_db_url(config, run_id, db_id, attempt)
    local_dir = _make_local_dir(config, run_id, db_id, attempt)
    local_dir.mkdir(parents=True, exist_ok=True)
    adapter = factory(db_url=db_url, local_dir=local_dir)
    stack.enter_context(adapter)
    state.adapters[db_id] = adapter
    state.db_urls[db_id] = db_url


def _flush_single_process_shard(
    state: _SingleProcessState,
    *,
    db_id: int,
    stack: contextlib.ExitStack,
    config: WriteConfig,
    run_id: str,
    attempt: int,
    factory: DbAdapterFactory,
) -> None:
    _ensure_single_process_adapter(
        state,
        db_id=db_id,
        stack=stack,
        config=config,
        run_id=run_id,
        attempt=attempt,
        factory=factory,
    )
    state.adapters[db_id].write_batch(state.batches[db_id])
    state.total_batched_items -= len(state.batches[db_id])
    state.total_batched_bytes -= state.batch_byte_sizes[db_id]
    state.batches[db_id].clear()
    state.batch_byte_sizes[db_id] = 0


def _buffer_single_process_record(
    state: _SingleProcessState,
    *,
    db_id: int,
    key: KeyInput,
    key_bytes: bytes,
    value_bytes: bytes,
) -> None:
    _ensure_single_process_shard(state, db_id=db_id)
    pair_bytes = len(key_bytes) + len(value_bytes)
    state.batches[db_id].append((key_bytes, value_bytes))
    state.batch_byte_sizes[db_id] += pair_bytes
    state.total_batched_items += 1
    state.total_batched_bytes += pair_bytes
    state.row_counts[db_id] += 1
    state.min_keys[db_id], state.max_keys[db_id] = update_min_max(
        state.min_keys[db_id], state.max_keys[db_id], key
    )


def _flush_single_process_shard_with_limits(
    state: _SingleProcessState,
    *,
    db_id: int,
    stack: contextlib.ExitStack,
    config: WriteConfig,
    run_id: str,
    attempt: int,
    factory: DbAdapterFactory,
    bucket: RateLimiter | None,
    byte_bucket: RateLimiter | None,
) -> None:
    if bucket is not None:
        bucket.acquire(len(state.batches[db_id]))
    if byte_bucket is not None:
        byte_bucket.acquire(state.batch_byte_sizes[db_id])
    _flush_single_process_shard(
        state,
        db_id=db_id,
        stack=stack,
        config=config,
        run_id=run_id,
        attempt=attempt,
        factory=factory,
    )


def _maybe_flush_single_process_batch(
    state: _SingleProcessState,
    *,
    db_id: int,
    stack: contextlib.ExitStack,
    config: WriteConfig,
    run_id: str,
    attempt: int,
    factory: DbAdapterFactory,
    bucket: RateLimiter | None,
    byte_bucket: RateLimiter | None,
) -> None:
    batch_size = len(state.batches[db_id])
    if batch_size < config.batch_size:
        return
    if bucket is None or bucket.try_acquire(batch_size):
        if byte_bucket is not None:
            byte_bucket.acquire(state.batch_byte_sizes[db_id])
        _flush_single_process_shard(
            state,
            db_id=db_id,
            stack=stack,
            config=config,
            run_id=run_id,
            attempt=attempt,
            factory=factory,
        )
        return
    if batch_size >= 2 * config.batch_size:
        _flush_single_process_shard_with_limits(
            state,
            db_id=db_id,
            stack=stack,
            config=config,
            run_id=run_id,
            attempt=attempt,
            factory=factory,
            bucket=bucket,
            byte_bucket=byte_bucket,
        )


def _single_process_over_memory_ceiling(
    state: _SingleProcessState,
    *,
    max_total_batched_items: int | None,
    max_total_batched_bytes: int | None,
) -> bool:
    return (
        max_total_batched_items is not None
        and state.total_batched_items > max_total_batched_items
    ) or (
        max_total_batched_bytes is not None
        and state.total_batched_bytes > max_total_batched_bytes
    )


def _largest_buffered_shard_id(state: _SingleProcessState) -> int:
    batch_ids = list(state.batches)
    return batch_ids[
        _largest_batch_id(
            [state.batches[db_id] for db_id in batch_ids],
            [state.batch_byte_sizes[db_id] for db_id in batch_ids],
        )
    ]


def _enforce_single_process_memory_ceiling(
    state: _SingleProcessState,
    *,
    stack: contextlib.ExitStack,
    config: WriteConfig,
    run_id: str,
    attempt: int,
    factory: DbAdapterFactory,
    bucket: RateLimiter | None,
    byte_bucket: RateLimiter | None,
    max_total_batched_items: int | None,
    max_total_batched_bytes: int | None,
) -> None:
    while state.batches and _single_process_over_memory_ceiling(
        state,
        max_total_batched_items=max_total_batched_items,
        max_total_batched_bytes=max_total_batched_bytes,
    ):
        db_id = _largest_buffered_shard_id(state)
        if not state.batches[db_id]:
            break
        _flush_single_process_shard_with_limits(
            state,
            db_id=db_id,
            stack=stack,
            config=config,
            run_id=run_id,
            attempt=attempt,
            factory=factory,
            bucket=bucket,
            byte_bucket=byte_bucket,
        )


def _flush_remaining_single_process_batches(
    state: _SingleProcessState,
    *,
    stack: contextlib.ExitStack,
    config: WriteConfig,
    run_id: str,
    attempt: int,
    factory: DbAdapterFactory,
    bucket: RateLimiter | None,
    byte_bucket: RateLimiter | None,
) -> None:
    for db_id in list(state.batches):
        if not state.batches[db_id]:
            continue
        _flush_single_process_shard_with_limits(
            state,
            db_id=db_id,
            stack=stack,
            config=config,
            run_id=run_id,
            attempt=attempt,
            factory=factory,
            bucket=bucket,
            byte_bucket=byte_bucket,
        )


def _finalize_single_process_adapters(state: _SingleProcessState) -> None:
    for db_id, adapter in state.adapters.items():
        adapter.flush()
        state.checkpoint_ids[db_id] = adapter.checkpoint()


def _build_single_process_results(
    state: _SingleProcessState,
    *,
    attempt: int,
    num_dbs: int,
) -> list[ShardAttemptResult]:
    all_db_ids = set(state.row_counts) | set(range(num_dbs))
    return [
        ShardAttemptResult(
            db_id=db_id,
            db_url=state.db_urls.get(db_id),
            attempt=attempt,
            row_count=state.row_counts.get(db_id, 0),
            min_key=state.min_keys.get(db_id),
            max_key=state.max_keys.get(db_id),
            checkpoint_id=state.checkpoint_ids.get(db_id),
            writer_info=WriterInfo(attempt=attempt),
        )
        for db_id in sorted(all_db_ids)
    ]


def _write_single_process(
    *,
    records: Iterable[T],
    config: WriteConfig,
    num_dbs: int | None,
    run_id: str,
    factory: DbAdapterFactory,
    key_fn: Callable[[T], KeyInput],
    value_fn: Callable[[T], bytes],
    columns_fn: Callable[[T], dict[str, Any]] | None = None,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    max_total_batched_items: int | None,
    max_total_batched_bytes: int | None,
) -> tuple[list[ShardAttemptResult], int]:
    """Single-process mode: all adapters open simultaneously, single pass.

    Returns (attempts, resolved_num_dbs) — num_dbs may be discovered from
    data for CEL direct mode.
    """

    attempt = 0
    cel_mode = num_dbs is None
    bucket: RateLimiter | None = None
    if max_writes_per_second is not None:
        bucket = TokenBucket(
            max_writes_per_second, metrics_collector=config.metrics_collector
        )
    byte_bucket: RateLimiter | None = None
    if max_write_bytes_per_second is not None:
        byte_bucket = TokenBucket(
            max_write_bytes_per_second,
            metrics_collector=config.metrics_collector,
            limiter_type="bytes",
        )

    state = _SingleProcessState()

    with contextlib.ExitStack() as stack:
        key_encoder = make_key_encoder(config.key_encoding)

        for record in records:
            key = key_fn(record)
            routing_context = columns_fn(record) if columns_fn is not None else None
            db_id = route_key(
                key,
                num_dbs=num_dbs,
                sharding=config.sharding,
                routing_context=routing_context,
            )
            key_bytes = key_encoder(key)
            value_bytes = value_fn(record)
            _buffer_single_process_record(
                state,
                db_id=db_id,
                key=key,
                key_bytes=key_bytes,
                value_bytes=value_bytes,
            )
            _maybe_flush_single_process_batch(
                state,
                db_id=db_id,
                stack=stack,
                config=config,
                run_id=run_id,
                attempt=attempt,
                factory=factory,
                bucket=bucket,
                byte_bucket=byte_bucket,
            )
            _enforce_single_process_memory_ceiling(
                state,
                stack=stack,
                config=config,
                run_id=run_id,
                attempt=attempt,
                factory=factory,
                bucket=bucket,
                byte_bucket=byte_bucket,
                max_total_batched_items=max_total_batched_items,
                max_total_batched_bytes=max_total_batched_bytes,
            )

        _flush_remaining_single_process_batches(
            state,
            stack=stack,
            config=config,
            run_id=run_id,
            attempt=attempt,
            factory=factory,
            bucket=bucket,
            byte_bucket=byte_bucket,
        )
        _finalize_single_process_adapters(state)

    if cel_mode:
        from shardyfusion._writer_core import discover_cel_num_dbs

        num_dbs = discover_cel_num_dbs(set(state.row_counts))

    return _build_single_process_results(
        state, attempt=attempt, num_dbs=num_dbs
    ), num_dbs
