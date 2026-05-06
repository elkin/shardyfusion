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
    build_categorical_routing_values,
    cleanup_losers,
    detect_kv_backend,
    detect_vector_backend,
    inject_sqlite_btreemeta_manifest_field,
    inject_vector_manifest_fields,
    publish_to_store,
    resolve_cel_num_dbs,
    select_winners,
    update_min_max,
    wrap_factory_for_vector,
)
from shardyfusion.config import (
    BaseShardedWriteConfig,
    CelShardedWriteConfig,
    HashShardedWriteConfig,
    PythonRecordInput,
    PythonWriteOptions,
    validate_configs,
)
from shardyfusion.errors import ConfigValidationError, ShardAssignmentError
from shardyfusion.logging import get_logger, log_event
from shardyfusion.manifest import BuildResult, WriterInfo
from shardyfusion.metrics import MetricEvent
from shardyfusion.run_registry import RunRecordLifecycle
from shardyfusion.serde import make_key_encoder
from shardyfusion.sharding_types import (
    CelShardingSpec,
    HashShardingSpec,
    ShardHashAlgorithm,
    ShardingSpec,
)
from shardyfusion.slatedb_adapter import DbAdapterFactory, SlateDbFactory
from shardyfusion.type_defs import KeyInput
from shardyfusion.writer.python._parallel_writer import (
    _make_db_url,
    _make_local_dir,
    _validate_parallel_shared_memory_limit,
    _write_parallel_cel,
    _write_parallel_hash,
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
    db_bytes_per_shard: dict[int, int] = field(default_factory=dict)
    total_batched_items: int = 0
    total_batched_bytes: int = 0
    # Vector batches (unified KV+vector mode)
    vector_ids: dict[int, list[int | str]] = field(default_factory=dict)
    vector_vecs: dict[int, list[Any]] = field(default_factory=dict)
    vector_payloads: dict[int, list[dict[str, Any] | None]] = field(
        default_factory=dict
    )


def _sharding_strategy_name(sharding: ShardingSpec) -> str:
    if isinstance(sharding, HashShardingSpec):
        return "hash"
    if isinstance(sharding, CelShardingSpec):
        return "cel"
    return "unknown"


def write_hash_sharded(
    records: Iterable[T],
    config: HashShardedWriteConfig,
    input: PythonRecordInput[T],
    options: PythonWriteOptions | None = None,
) -> BuildResult:
    """Write an iterable of records into N sharded databases using HASH routing.

    Args:
        records: Iterable of records to write. Each record is passed to
            ``input.key_fn`` and ``input.value_fn`` to extract the key and value
            bytes.
        config: Hash write configuration.
        input: Record extraction callbacks.
        options: Per-call execution options for multiprocessing and buffering.

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

    options = options or PythonWriteOptions()
    validate_configs(config, input, options)
    key_fn = input.key_fn
    value_fn = input.value_fn
    columns_fn = input.columns_fn
    vector_fn = input.vector_fn
    parallel = options.parallel
    max_queue_size = options.max_queue_size
    max_parallel_shared_memory_bytes = options.shared_memory.max_total_bytes
    max_parallel_shared_memory_bytes_per_worker = (
        options.shared_memory.max_bytes_per_worker
    )
    max_writes_per_second = config.rate_limits.max_writes_per_second
    max_write_bytes_per_second = config.rate_limits.max_write_bytes_per_second
    max_total_batched_items = options.buffering.max_total_batched_items
    max_total_batched_bytes = options.buffering.max_total_batched_bytes

    sharding = HashShardingSpec(
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
        max_keys_per_shard=config.max_keys_per_shard,
    )
    num_dbs = _resolve_num_dbs_before_sharding(records, config)

    _validate_parallel_shared_memory_limit(
        max_parallel_shared_memory_bytes,
        field_name="max_parallel_shared_memory_bytes",
    )
    _validate_parallel_shared_memory_limit(
        max_parallel_shared_memory_bytes_per_worker,
        field_name="max_parallel_shared_memory_bytes_per_worker",
    )

    has_vectors, vector_fn, factory = _resolve_vector_fn_and_factory(
        config=config,
        vector_fn=vector_fn,
        columns_fn=columns_fn,
        key_fn=key_fn,
    )
    if parallel and has_vectors:
        raise ConfigValidationError(
            "Parallel mode is not supported for unified KV+vector writes."
        )

    run_id = config.output.run_id or uuid4().hex
    started = time.perf_counter()
    mc = config.metrics_collector

    with (
        LogContext(run_id=run_id),
        RunRecordLifecycle.start(
            config=config,
            run_id=run_id,
            writer_type="python",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=num_dbs,
            s3_prefix=config.s3_prefix,
            strategy="hash",
            key_encoding=config.key_encoding,
            writer_type="python",
        )
        if mc is not None:
            mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})

        if parallel:
            attempts = _write_parallel_hash(
                records=records,
                config=config,
                num_dbs=num_dbs,
                run_id=run_id,
                factory=factory,
                key_fn=key_fn,
                value_fn=value_fn,
                columns_fn=columns_fn,
                hash_algorithm=sharding.hash_algorithm,
                max_queue_size=max_queue_size,
                max_parallel_shared_memory_bytes=max_parallel_shared_memory_bytes,
                max_parallel_shared_memory_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
                max_writes_per_second=max_writes_per_second,
                max_write_bytes_per_second=max_write_bytes_per_second,
            )
        else:
            attempts, num_dbs = _write_single_process_hash(
                records=records,
                config=config,
                num_dbs=num_dbs,
                hash_algorithm=sharding.hash_algorithm,
                run_id=run_id,
                factory=factory,
                key_fn=key_fn,
                value_fn=value_fn,
                columns_fn=columns_fn,
                vector_fn=vector_fn,
                max_writes_per_second=max_writes_per_second,
                max_write_bytes_per_second=max_write_bytes_per_second,
                max_total_batched_items=max_total_batched_items,
                max_total_batched_bytes=max_total_batched_bytes,
            )

        return _finalize_python_write(
            config=config,
            run_id=run_id,
            started=started,
            sharding=sharding,
            attempts=attempts,
            num_dbs=num_dbs,
            factory=factory,
            mc=mc,
            run_record=run_record,
        )


def write_cel_sharded(
    records: Iterable[T],
    config: CelShardedWriteConfig,
    input: PythonRecordInput[T],
    options: PythonWriteOptions | None = None,
) -> BuildResult:
    """Write an iterable of records into N sharded databases using CEL routing.

    Args:
        records: Iterable of records to write. Each record is passed to
            ``input.key_fn`` and ``input.value_fn`` to extract the key and value
            bytes.
        config: CEL write configuration.
        input: Record extraction callbacks, including ``columns_fn`` for CEL row
            context when needed.
        options: Per-call execution options for multiprocessing and buffering.

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

    options = options or PythonWriteOptions()
    validate_configs(config, input, options)
    key_fn = input.key_fn
    value_fn = input.value_fn
    columns_fn = input.columns_fn
    vector_fn = input.vector_fn
    parallel = options.parallel
    max_queue_size = options.max_queue_size
    max_parallel_shared_memory_bytes = options.shared_memory.max_total_bytes
    max_parallel_shared_memory_bytes_per_worker = (
        options.shared_memory.max_bytes_per_worker
    )
    max_writes_per_second = config.rate_limits.max_writes_per_second
    max_write_bytes_per_second = config.rate_limits.max_write_bytes_per_second
    max_total_batched_items = options.buffering.max_total_batched_items
    max_total_batched_bytes = options.buffering.max_total_batched_bytes

    sharding: ShardingSpec = CelShardingSpec(
        cel_expr=config.cel_expr,
        cel_columns=config.cel_columns,
        routing_values=config.routing_values,
        infer_routing_values_from_data=config.infer_routing_values_from_data,
    )

    if sharding.infer_routing_values_from_data:
        if parallel:
            raise ConfigValidationError(
                "Parallel mode does not support inferred categorical CEL routing. "
                "Materialize records and use single-process mode."
            )
        sharding = _resolve_inferred_categorical_sharding(
            records,
            sharding=sharding,
            key_fn=key_fn,
            columns_fn=columns_fn,
        )

    assert isinstance(sharding, CelShardingSpec)
    num_dbs = (
        resolve_cel_num_dbs(sharding) if sharding.routing_values is not None else None
    )

    _validate_parallel_shared_memory_limit(
        max_parallel_shared_memory_bytes,
        field_name="max_parallel_shared_memory_bytes",
    )
    _validate_parallel_shared_memory_limit(
        max_parallel_shared_memory_bytes_per_worker,
        field_name="max_parallel_shared_memory_bytes_per_worker",
    )

    has_vectors, vector_fn, factory = _resolve_vector_fn_and_factory(
        config=config,
        vector_fn=vector_fn,
        columns_fn=columns_fn,
        key_fn=key_fn,
    )
    if parallel and has_vectors:
        raise ConfigValidationError(
            "Parallel mode is not supported for unified KV+vector writes."
        )

    run_id = config.output.run_id or uuid4().hex
    started = time.perf_counter()
    mc = config.metrics_collector

    with (
        LogContext(run_id=run_id),
        RunRecordLifecycle.start(
            config=config,
            run_id=run_id,
            writer_type="python",
        ) as run_record,
    ):
        log_event(
            "write_started",
            logger=_logger,
            num_dbs=num_dbs,
            s3_prefix=config.s3_prefix,
            strategy="cel",
            key_encoding=config.key_encoding,
            writer_type="python",
        )
        if mc is not None:
            mc.emit(MetricEvent.WRITE_STARTED, {"elapsed_ms": 0})

        if parallel:
            if num_dbs is None:
                raise ConfigValidationError(
                    "Parallel mode requires num_dbs > 0 to spawn workers. "
                    "CEL direct mode (num_dbs discovered from data) "
                    "is only supported in single-process mode."
                )
            attempts = _write_parallel_cel(
                records=records,
                config=config,
                num_dbs=num_dbs,
                run_id=run_id,
                factory=factory,
                key_fn=key_fn,
                value_fn=value_fn,
                columns_fn=columns_fn,
                cel_expr=sharding.cel_expr,
                cel_columns=sharding.cel_columns,
                routing_values=sharding.routing_values,
                max_queue_size=max_queue_size,
                max_parallel_shared_memory_bytes=max_parallel_shared_memory_bytes,
                max_parallel_shared_memory_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
                max_writes_per_second=max_writes_per_second,
                max_write_bytes_per_second=max_write_bytes_per_second,
            )
        else:
            attempts, num_dbs = _write_single_process_cel(
                records=records,
                config=config,
                num_dbs=num_dbs,
                cel_expr=sharding.cel_expr,
                cel_columns=sharding.cel_columns,
                routing_values=sharding.routing_values,
                run_id=run_id,
                factory=factory,
                key_fn=key_fn,
                value_fn=value_fn,
                columns_fn=columns_fn,
                vector_fn=vector_fn,
                max_writes_per_second=max_writes_per_second,
                max_write_bytes_per_second=max_write_bytes_per_second,
                max_total_batched_items=max_total_batched_items,
                max_total_batched_bytes=max_total_batched_bytes,
            )

        assert num_dbs is not None
        return _finalize_python_write(
            config=config,
            run_id=run_id,
            started=started,
            sharding=sharding,
            attempts=attempts,
            num_dbs=num_dbs,
            factory=factory,
            mc=mc,
            run_record=run_record,
        )


def _finalize_python_write(
    *,
    config: BaseShardedWriteConfig,
    run_id: str,
    started: float,
    sharding: ShardingSpec,
    attempts: list[ShardAttemptResult],
    num_dbs: int,
    factory: DbAdapterFactory,
    mc: Any | None,
    run_record: Any,
) -> BuildResult:
    """Select winners, publish manifest, cleanup, and assemble BuildResult."""
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

    winners, num_attempts, all_attempt_urls = select_winners(attempts, num_dbs=num_dbs)

    if config.vector_spec is not None:
        inject_vector_manifest_fields(config, factory)
    inject_sqlite_btreemeta_manifest_field(config, factory)

    manifest_started = time.perf_counter()
    manifest_ref = publish_to_store(
        config=config,
        run_id=run_id,
        resolved_sharding=sharding,
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


def _resolve_vector_fn_and_factory(
    *,
    config: BaseShardedWriteConfig,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None,
    columns_fn: Callable[[Any], dict[str, Any]] | None,
    key_fn: Callable[[Any], KeyInput] | None = None,
) -> tuple[
    bool,
    Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None,
    DbAdapterFactory,
]:
    """Validate vector config, auto-build vector_fn if needed, wrap factory.

    Returns (has_vectors, resolved_vector_fn, factory).
    Raises ConfigValidationError eagerly for invalid combinations.
    """
    has_vectors = vector_fn is not None or config.vector_spec is not None
    if has_vectors:
        if vector_fn is not None and config.vector_spec is None:
            raise ConfigValidationError(
                "vector_fn requires config.vector_spec to be set"
            )
        if config.vector_spec is not None and vector_fn is None:
            if config.vector_spec.vector_col is None:
                raise ConfigValidationError(
                    "config.vector_spec is set but no vector_fn was provided "
                    "and vector_spec.vector_col is None. Either provide vector_fn "
                    "or set vector_spec.vector_col for auto-extraction via columns_fn."
                )
            if columns_fn is None:
                raise ConfigValidationError(
                    "config.vector_spec.vector_col is set but columns_fn is None. "
                    "Provide columns_fn so vectors can be extracted from "
                    f"the {config.vector_spec.vector_col!r} column."
                )
            _vector_col = config.vector_spec.vector_col
            _columns_fn = columns_fn
            _key_fn = key_fn

            def _auto_vector_fn(
                record: Any,
            ) -> tuple[int | str, Any, dict[str, Any] | None]:
                cols = _columns_fn(record)  # type: ignore[misc]
                vec = cols[_vector_col]  # type: ignore[index]
                key = _key_fn(record)  # type: ignore[misc]
                vec_id: int | str = key if isinstance(key, (int, str)) else key.hex()
                return (vec_id, vec, None)

            vector_fn = _auto_vector_fn

    factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory(
        credential_provider=config.credential_provider
    )
    if has_vectors:
        factory = wrap_factory_for_vector(factory, config)

    return has_vectors, vector_fn, factory


def _resolve_num_dbs_before_sharding(
    records: Iterable[Any],
    config: HashShardedWriteConfig,
) -> int:
    """Resolve num_dbs that can be determined before writing.

    Returns ``None`` for CEL (discovered during iteration from data).
    """
    from collections.abc import Sized

    from shardyfusion._writer_core import resolve_num_dbs

    if config.max_keys_per_shard is not None and not isinstance(records, Sized):
        raise ConfigValidationError(
            "max_keys_per_shard requires a Sized input (e.g. list) "
            "so num_dbs can be computed from len(records). "
            "Provide num_dbs explicitly for non-Sized iterables."
        )

    return resolve_num_dbs(
        config,
        lambda: len(records) if isinstance(records, Sized) else 0,
    )


def _resolve_inferred_categorical_sharding(
    records: Iterable[T],
    *,
    sharding: ShardingSpec,
    key_fn: Callable[[T], KeyInput],
    columns_fn: Callable[[T], dict[str, Any]] | None,
) -> ShardingSpec:
    if iter(records) is records:
        raise ConfigValidationError(
            "Inferred categorical CEL routing requires a reiterable input. "
            "Materialize generators before writing."
        )

    from shardyfusion.cel import compile_cel

    assert isinstance(sharding, CelShardingSpec)
    assert sharding.cel_expr is not None and sharding.cel_columns is not None
    compiled = compile_cel(sharding.cel_expr, sharding.cel_columns)
    tokens = []
    for record in records:
        context = (
            columns_fn(record) if columns_fn is not None else {"key": key_fn(record)}
        )
        tokens.append(compiled.evaluate(context))

    return CelShardingSpec(
        cel_expr=sharding.cel_expr,
        cel_columns=dict(sharding.cel_columns),
        routing_values=build_categorical_routing_values(tokens),
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
    state.vector_ids[db_id] = []
    state.vector_vecs[db_id] = []
    state.vector_payloads[db_id] = []


def _ensure_single_process_adapter(
    state: _SingleProcessState,
    *,
    db_id: int,
    stack: contextlib.ExitStack,
    config: BaseShardedWriteConfig,
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
    config: BaseShardedWriteConfig,
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
    adapter = state.adapters[db_id]
    adapter.write_batch(state.batches[db_id])

    # Flush vector batch if present (unified KV+vector mode)
    if state.vector_ids[db_id] and hasattr(adapter, "write_vector_batch"):
        import numpy as np  # pyright: ignore[reportMissingImports]

        ids_arr = np.array(state.vector_ids[db_id])
        vecs_arr = np.array(state.vector_vecs[db_id], dtype=np.float32)
        adapter.write_vector_batch(ids_arr, vecs_arr, state.vector_payloads[db_id])
        state.vector_ids[db_id].clear()
        state.vector_vecs[db_id].clear()
        state.vector_payloads[db_id].clear()

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
    config: BaseShardedWriteConfig,
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
    config: BaseShardedWriteConfig,
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
    config: BaseShardedWriteConfig,
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
    config: BaseShardedWriteConfig,
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
        state.db_bytes_per_shard[db_id] = adapter.db_bytes()


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
            db_bytes=state.db_bytes_per_shard.get(db_id, 0),
        )
        for db_id in sorted(all_db_ids)
    ]


def _write_single_process_impl(
    *,
    records: Iterable[T],
    config: BaseShardedWriteConfig,
    num_dbs: int | None,
    get_db_id: Callable[[KeyInput, T], int],
    routing_values: list[int | str | bytes] | None,
    run_id: str,
    factory: DbAdapterFactory,
    key_fn: Callable[[T], KeyInput],
    value_fn: Callable[[T], bytes],
    columns_fn: Callable[[T], dict[str, Any]] | None = None,
    vector_fn: Callable[[T], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
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
            db_id = get_db_id(key, record)
            key_bytes = key_encoder(key)
            value_bytes = value_fn(record)
            _buffer_single_process_record(
                state,
                db_id=db_id,
                key=key,
                key_bytes=key_bytes,
                value_bytes=value_bytes,
            )
            # Buffer vector data if unified mode
            if vector_fn is not None:
                vec_id, vec_data, vec_payload = vector_fn(record)
                state.vector_ids[db_id].append(vec_id)
                state.vector_vecs[db_id].append(vec_data)
                state.vector_payloads[db_id].append(vec_payload)
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

    if num_dbs is None:
        if routing_values is not None:
            num_dbs = max(1, len(routing_values))
        else:
            from shardyfusion._writer_core import discover_cel_num_dbs

            num_dbs = discover_cel_num_dbs(set(state.row_counts))

    return _build_single_process_results(
        state, attempt=attempt, num_dbs=num_dbs
    ), num_dbs


def _write_single_process_hash(
    *,
    records: Iterable[T],
    config: BaseShardedWriteConfig,
    num_dbs: int,
    hash_algorithm: ShardHashAlgorithm,
    run_id: str,
    factory: DbAdapterFactory,
    key_fn: Callable[[T], KeyInput],
    value_fn: Callable[[T], bytes],
    columns_fn: Callable[[T], dict[str, Any]] | None = None,
    vector_fn: Callable[[T], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    max_total_batched_items: int | None,
    max_total_batched_bytes: int | None,
) -> tuple[list[ShardAttemptResult], int]:
    from shardyfusion.routing import make_hash_router

    # Build the hash router once (resolves enum + skips per-key dispatch);
    # key type is unknown for arbitrary Python iterables so we use the
    # generic 3-branch fallback.
    route = make_hash_router(num_dbs, hash_algorithm)

    def get_db_id(key: KeyInput, record: T) -> int:
        return route(key)

    return _write_single_process_impl(
        records=records,
        config=config,
        num_dbs=num_dbs,
        get_db_id=get_db_id,
        routing_values=None,
        run_id=run_id,
        factory=factory,
        key_fn=key_fn,
        value_fn=value_fn,
        columns_fn=columns_fn,
        vector_fn=vector_fn,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
        max_total_batched_items=max_total_batched_items,
        max_total_batched_bytes=max_total_batched_bytes,
    )


def _write_single_process_cel(
    *,
    records: Iterable[T],
    config: BaseShardedWriteConfig,
    num_dbs: int | None,
    cel_expr: str,
    cel_columns: dict[str, str],
    routing_values: list[int | str | bytes] | None,
    run_id: str,
    factory: DbAdapterFactory,
    key_fn: Callable[[T], KeyInput],
    value_fn: Callable[[T], bytes],
    columns_fn: Callable[[T], dict[str, Any]] | None = None,
    vector_fn: Callable[[T], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
    max_total_batched_items: int | None,
    max_total_batched_bytes: int | None,
) -> tuple[list[ShardAttemptResult], int]:
    # Hoist CEL compile + categorical lookup out of the per-row loop.
    from shardyfusion.cel import (
        UnknownRoutingTokenError,
        compile_cel_cached,
    )
    from shardyfusion.cel import (
        route_cel as _cel_route,
    )

    _compiled_cel = compile_cel_cached(cel_expr, tuple(sorted(cel_columns.items())))
    _cel_lookup = (
        {value: idx for idx, value in enumerate(routing_values)}
        if routing_values is not None
        else None
    )
    _columns_fn = columns_fn  # bind locally so pyright narrows below

    if _columns_fn is None:

        def get_db_id(key: KeyInput, record: T) -> int:
            ctx: dict[str, object] = {"key": key}
            try:
                return _cel_route(_compiled_cel, ctx, routing_values, _cel_lookup)
            except UnknownRoutingTokenError as exc:
                raise ShardAssignmentError(str(exc)) from exc

    else:

        def get_db_id(key: KeyInput, record: T) -> int:
            ctx = _columns_fn(record)
            try:
                return _cel_route(_compiled_cel, ctx, routing_values, _cel_lookup)
            except UnknownRoutingTokenError as exc:
                raise ShardAssignmentError(str(exc)) from exc

    return _write_single_process_impl(
        records=records,
        config=config,
        num_dbs=num_dbs,
        get_db_id=get_db_id,
        routing_values=routing_values,
        run_id=run_id,
        factory=factory,
        key_fn=key_fn,
        value_fn=value_fn,
        columns_fn=columns_fn,
        vector_fn=vector_fn,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
        max_total_batched_items=max_total_batched_items,
        max_total_batched_bytes=max_total_batched_bytes,
    )


# Backward-compatible aliases for tests and internal callers


def _wrap_factory_for_vector(
    factory: DbAdapterFactory, config: BaseShardedWriteConfig
) -> DbAdapterFactory:
    return wrap_factory_for_vector(factory, config)


def _detect_vector_backend(factory: DbAdapterFactory) -> str:
    return detect_vector_backend(factory)


def _detect_kv_backend(factory: DbAdapterFactory) -> str:
    return detect_kv_backend(factory)


def _inject_vector_manifest_fields(
    config: BaseShardedWriteConfig, factory: DbAdapterFactory
) -> None:
    inject_vector_manifest_fields(config, factory)
