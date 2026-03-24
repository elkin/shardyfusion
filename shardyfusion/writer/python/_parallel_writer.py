"""Parallel shared-memory implementation for the Python sharded writer."""

from __future__ import annotations

import contextlib
import inspect
import multiprocessing
import queue
import struct
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from multiprocessing.connection import wait as wait_for_handles
from pathlib import Path
from typing import Any, TypeVar, cast

from shardyfusion._rate_limiter import RateLimiter, TokenBucket
from shardyfusion._writer_core import ShardAttemptResult, route_key, update_min_max
from shardyfusion.config import WriteConfig
from shardyfusion.errors import ConfigValidationError, ShardyfusionError
from shardyfusion.logging import FailureSeverity, get_logger, log_failure
from shardyfusion.manifest import WriterInfo
from shardyfusion.serde import make_key_encoder
from shardyfusion.slatedb_adapter import DbAdapterFactory
from shardyfusion.storage import join_s3
from shardyfusion.type_defs import KeyInput

_logger = get_logger(__name__)

T = TypeVar("T")

_CHUNK_DIVISOR = 10
_WORKER_JOIN_TIMEOUT = timedelta(seconds=60)
_WORKER_TERMINATE_TIMEOUT = timedelta(seconds=5)
_RESULT_COLLECT_TIMEOUT = timedelta(seconds=10)
_SHARED_MEMORY_RECORD_HEADER = struct.Struct("<QQ")
_MAX_SHARED_MEMORY_CHUNK_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_PARALLEL_SHARED_MEMORY_BYTES = 256 * 1024 * 1024
_DEFAULT_MAX_PARALLEL_SHARED_MEMORY_BYTES_PER_WORKER = 32 * 1024 * 1024
_SHARED_MEMORY_SUPPORTS_TRACK = (
    "track" in inspect.signature(shared_memory.SharedMemory).parameters
)


@dataclass(slots=True, frozen=True)
class _SharedMemoryChunkRef:
    name: str
    size: int
    row_count: int


@dataclass(slots=True, frozen=True)
class _SharedMemoryReclaim:
    name: str


@dataclass(slots=True)
class _OutstandingSharedMemory:
    db_id: int
    size: int
    handle: shared_memory.SharedMemory


@dataclass(slots=True)
class _SharedMemoryState:
    outstanding_segments: dict[str, _OutstandingSharedMemory]
    outstanding_bytes: int
    outstanding_bytes_by_worker: list[int]


@dataclass(slots=True)
class _ParallelRuntime:
    queues: list[multiprocessing.Queue[_SharedMemoryChunkRef | None]]
    reclaim_queue: multiprocessing.Queue[_SharedMemoryReclaim]
    result_queue: multiprocessing.Queue[ShardAttemptResult]
    workers: list[multiprocessing.Process]
    chunk_bufs: list[list[tuple[bytes, bytes]]]
    chunk_byte_sizes: list[int]
    min_keys: list[KeyInput | None]
    max_keys: list[KeyInput | None]
    row_counts: list[int]
    key_encoder: Callable[[KeyInput], bytes]
    shared_memory_state: _SharedMemoryState


def _validate_parallel_shared_memory_limit(
    value: int | None, *, field_name: str
) -> None:
    if value is None:
        return
    if value <= 0:
        raise ConfigValidationError(f"{field_name} must be > 0 when set")


def _batch_bytes(batch: list[tuple[bytes, bytes]]) -> int:
    return sum(len(key) + len(value) for key, value in batch)


def _make_db_url(config: WriteConfig, run_id: str, db_id: int, attempt: int) -> str:
    db_rel_path = config.output.db_path_template.format(db_id=db_id)
    return join_s3(
        config.s3_prefix,
        config.output.shard_prefix,
        f"run_id={run_id}",
        db_rel_path,
        f"attempt={attempt:02d}",
    )


def _make_local_dir(config: WriteConfig, run_id: str, db_id: int, attempt: int) -> Path:
    return (
        Path(config.output.local_root)
        / f"run_id={run_id}"
        / f"db={db_id:05d}"
        / f"attempt={attempt:02d}"
    )


def _create_shared_memory_handle(*, size: int) -> shared_memory.SharedMemory:
    ctor = cast(Any, shared_memory.SharedMemory)
    if _SHARED_MEMORY_SUPPORTS_TRACK:
        return ctor(create=True, size=size, track=True)
    return shared_memory.SharedMemory(create=True, size=size)


def _attach_shared_memory_handle(*, name: str) -> shared_memory.SharedMemory:
    ctor = cast(Any, shared_memory.SharedMemory)
    if _SHARED_MEMORY_SUPPORTS_TRACK:
        return ctor(name=name, create=False, track=False)
    return shared_memory.SharedMemory(name=name, create=False)


def _shared_memory_chunk_bytes(batch: list[tuple[bytes, bytes]]) -> int:
    return sum(
        _SHARED_MEMORY_RECORD_HEADER.size + len(key) + len(value)
        for key, value in batch
    )


def _create_shared_memory_chunk(
    batch: list[tuple[bytes, bytes]],
    *,
    size: int,
) -> shared_memory.SharedMemory:
    shm = _create_shared_memory_handle(size=size)
    try:
        buf = shm.buf
        if buf is None:
            raise ShardyfusionError("Shared-memory segment did not expose a buffer")
        offset = 0
        for key, value in batch:
            _SHARED_MEMORY_RECORD_HEADER.pack_into(
                buf,
                offset,
                len(key),
                len(value),
            )
            offset += _SHARED_MEMORY_RECORD_HEADER.size

            key_end = offset + len(key)
            buf[offset:key_end] = key
            offset = key_end

            value_end = offset + len(value)
            buf[offset:value_end] = value
            offset = value_end
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            shm.unlink()
        shm.close()
        raise
    return shm


def _extend_batch_from_shared_memory(
    batch: list[tuple[bytes, bytes]],
    *,
    chunk: _SharedMemoryChunkRef,
) -> None:
    try:
        shm = _attach_shared_memory_handle(name=chunk.name)
    except FileNotFoundError as exc:
        raise ShardyfusionError(
            f"Shared-memory chunk {chunk.name!r} was not available to the worker"
        ) from exc

    try:
        buf = shm.buf
        if buf is None:
            raise ShardyfusionError("Shared-memory segment did not expose a buffer")
        if chunk.size > shm.size:
            raise ShardyfusionError(
                f"Shared-memory chunk {chunk.name!r} declared size {chunk.size} "
                f"but segment size was {shm.size}"
            )

        offset = 0
        for _ in range(chunk.row_count):
            header_end = offset + _SHARED_MEMORY_RECORD_HEADER.size
            if header_end > chunk.size:
                raise ShardyfusionError(
                    f"Shared-memory chunk {chunk.name!r} ended mid-record header"
                )

            key_len, value_len = _SHARED_MEMORY_RECORD_HEADER.unpack_from(buf, offset)
            offset = header_end

            key_end = offset + key_len
            value_end = key_end + value_len
            if value_end > chunk.size:
                raise ShardyfusionError(
                    f"Shared-memory chunk {chunk.name!r} ended mid-record payload"
                )

            batch.append(
                (
                    bytes(buf[offset:key_end]),
                    bytes(buf[key_end:value_end]),
                )
            )
            offset = value_end

        if offset != chunk.size:
            raise ShardyfusionError(
                f"Shared-memory chunk {chunk.name!r} had trailing bytes"
            )
    finally:
        shm.close()


def _check_workers_alive(workers: list[multiprocessing.Process]) -> None:
    exited = [
        (db_id, process.exitcode)
        for db_id, process in enumerate(workers)
        if process.exitcode is not None
    ]
    if not exited:
        return
    failures = [(db_id, code) for db_id, code in exited if code]
    if failures:
        raise ShardyfusionError(f"Worker process(es) exited with errors: {failures}")
    raise ShardyfusionError(
        f"Worker process(es) exited unexpectedly before write completion: {exited}"
    )


def _release_shared_memory_segment(
    state: _SharedMemoryState,
    *,
    name: str,
) -> None:
    segment = state.outstanding_segments.pop(name, None)
    if segment is None:
        return

    state.outstanding_bytes -= segment.size
    state.outstanding_bytes_by_worker[segment.db_id] -= segment.size
    segment.handle.close()
    with contextlib.suppress(FileNotFoundError):
        segment.handle.unlink()


def _drain_reclaimed_segments(
    reclaim_queue: multiprocessing.Queue[_SharedMemoryReclaim],
    state: _SharedMemoryState,
) -> None:
    while True:
        try:
            reclaim = reclaim_queue.get_nowait()
        except queue.Empty:
            return
        _release_shared_memory_segment(state, name=reclaim.name)


def _shared_memory_budget_available(
    *,
    db_id: int,
    required_size: int,
    max_parallel_shared_memory_bytes: int | None,
    max_parallel_shared_memory_bytes_per_worker: int | None,
    state: _SharedMemoryState,
) -> bool:
    exceeds_global = (
        max_parallel_shared_memory_bytes is not None
        and state.outstanding_bytes + required_size > max_parallel_shared_memory_bytes
    )
    exceeds_worker = (
        max_parallel_shared_memory_bytes_per_worker is not None
        and state.outstanding_bytes_by_worker[db_id] + required_size
        > max_parallel_shared_memory_bytes_per_worker
    )
    return not exceeds_global and not exceeds_worker


def _wait_for_reclaim_or_worker_exit(
    *,
    reclaim_queue: multiprocessing.Queue[_SharedMemoryReclaim],
    state: _SharedMemoryState,
    workers: list[multiprocessing.Process],
) -> None:
    reclaim_reader = cast(Any, getattr(reclaim_queue, "_reader", None))
    if reclaim_reader is None:
        reclaim = reclaim_queue.get()
        _release_shared_memory_segment(state, name=reclaim.name)
        return

    worker_sentinels = [cast(Any, worker).sentinel for worker in workers]
    ready = wait_for_handles([reclaim_reader, *worker_sentinels])

    if any(sentinel in ready for sentinel in worker_sentinels):
        _check_workers_alive(workers)
    if reclaim_reader in ready:
        reclaim = reclaim_queue.get()
        _release_shared_memory_segment(state, name=reclaim.name)


def _wait_for_shared_memory_budget(
    *,
    db_id: int,
    required_size: int,
    max_parallel_shared_memory_bytes: int | None,
    max_parallel_shared_memory_bytes_per_worker: int | None,
    reclaim_queue: multiprocessing.Queue[_SharedMemoryReclaim],
    state: _SharedMemoryState,
    workers: list[multiprocessing.Process],
) -> None:
    if (
        max_parallel_shared_memory_bytes is not None
        and required_size > max_parallel_shared_memory_bytes
    ):
        raise ShardyfusionError(
            "Serialized shard chunk exceeds max_parallel_shared_memory_bytes: "
            f"required={required_size}, limit={max_parallel_shared_memory_bytes}"
        )
    if (
        max_parallel_shared_memory_bytes_per_worker is not None
        and required_size > max_parallel_shared_memory_bytes_per_worker
    ):
        raise ShardyfusionError(
            "Serialized shard chunk exceeds "
            "max_parallel_shared_memory_bytes_per_worker: "
            f"required={required_size}, limit={max_parallel_shared_memory_bytes_per_worker}"
        )

    while True:
        _drain_reclaimed_segments(reclaim_queue, state)
        if _shared_memory_budget_available(
            db_id=db_id,
            required_size=required_size,
            max_parallel_shared_memory_bytes=max_parallel_shared_memory_bytes,
            max_parallel_shared_memory_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
            state=state,
        ):
            return
        _wait_for_reclaim_or_worker_exit(
            reclaim_queue=reclaim_queue,
            state=state,
            workers=workers,
        )


def _cleanup_outstanding_segments(state: _SharedMemoryState) -> None:
    for name in list(state.outstanding_segments):
        _release_shared_memory_segment(state, name=name)


def _shard_worker(
    db_id: int,
    queue: multiprocessing.Queue[_SharedMemoryChunkRef | None],
    reclaim_queue: multiprocessing.Queue[_SharedMemoryReclaim],
    result_queue: multiprocessing.Queue[ShardAttemptResult],
    config: WriteConfig,
    run_id: str,
    factory: DbAdapterFactory,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
) -> None:
    attempt = 0
    db_url = _make_db_url(config, run_id, db_id, attempt)
    local_dir = _make_local_dir(config, run_id, db_id, attempt)
    local_dir.mkdir(parents=True, exist_ok=True)

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

    row_count = 0
    min_key: KeyInput | None = None
    max_key: KeyInput | None = None
    checkpoint_id: str | None = None

    try:
        with factory(db_url=db_url, local_dir=local_dir) as adapter:
            batch: list[tuple[bytes, bytes]] = []
            while True:
                chunk = queue.get()
                if chunk is None:
                    break
                _extend_batch_from_shared_memory(batch, chunk=chunk)
                row_count += chunk.row_count
                reclaim_queue.put(_SharedMemoryReclaim(name=chunk.name))

                while len(batch) >= config.batch_size:
                    write_slice = batch[: config.batch_size]
                    if bucket is not None:
                        bucket.acquire(config.batch_size)
                    if byte_bucket is not None:
                        byte_bucket.acquire(_batch_bytes(write_slice))
                    adapter.write_batch(write_slice)
                    batch = batch[config.batch_size :]

            if batch:
                if bucket is not None:
                    bucket.acquire(len(batch))
                if byte_bucket is not None:
                    byte_bucket.acquire(_batch_bytes(batch))
                adapter.write_batch(batch)
                batch.clear()

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
    except Exception as exc:
        log_failure(
            "python_shard_worker_failed",
            severity=FailureSeverity.ERROR,
            logger=_logger,
            error=exc,
            run_id=run_id,
            db_id=db_id,
        )
        raise ShardyfusionError(f"Shard write failed for db_id={db_id}: {exc}") from exc

    result_queue.put(
        ShardAttemptResult(
            db_id=db_id,
            db_url=db_url,
            attempt=attempt,
            row_count=row_count,
            min_key=min_key,
            max_key=max_key,
            checkpoint_id=checkpoint_id,
            writer_info=WriterInfo(attempt=attempt),
        )
    )


def _start_parallel_runtime(
    *,
    config: WriteConfig,
    num_dbs: int,
    run_id: str,
    factory: DbAdapterFactory,
    max_queue_size: int,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
) -> _ParallelRuntime:
    ctx = multiprocessing.get_context("spawn")
    queues: list[multiprocessing.Queue[_SharedMemoryChunkRef | None]] = [
        ctx.Queue(maxsize=max_queue_size) for _ in range(num_dbs)
    ]
    reclaim_queue: multiprocessing.Queue[_SharedMemoryReclaim] = ctx.Queue()
    result_queue: multiprocessing.Queue[ShardAttemptResult] = ctx.Queue()

    workers: list[multiprocessing.Process] = []
    for db_id in range(num_dbs):
        process = ctx.Process(
            target=_shard_worker,
            args=(
                db_id,
                queues[db_id],
                reclaim_queue,
                result_queue,
                config,
                run_id,
                factory,
                max_writes_per_second,
                max_write_bytes_per_second,
            ),
        )
        process.start()
        workers.append(process)  # type: ignore[arg-type]  # SpawnProcess is a Process subclass

    return _ParallelRuntime(
        queues=queues,
        reclaim_queue=reclaim_queue,
        result_queue=result_queue,
        workers=workers,
        chunk_bufs=[[] for _ in range(num_dbs)],
        chunk_byte_sizes=[0] * num_dbs,
        min_keys=[None] * num_dbs,
        max_keys=[None] * num_dbs,
        row_counts=[0] * num_dbs,
        key_encoder=make_key_encoder(config.key_encoding),
        shared_memory_state=_SharedMemoryState(
            outstanding_segments={},
            outstanding_bytes=0,
            outstanding_bytes_by_worker=[0] * num_dbs,
        ),
    )


def _send_chunk(
    runtime: _ParallelRuntime,
    *,
    db_id: int,
    max_parallel_shared_memory_bytes: int | None,
    max_parallel_shared_memory_bytes_per_worker: int | None,
) -> None:
    batch = runtime.chunk_bufs[db_id]
    if not batch:
        return

    required_size = runtime.chunk_byte_sizes[db_id]
    _wait_for_shared_memory_budget(
        db_id=db_id,
        required_size=required_size,
        max_parallel_shared_memory_bytes=max_parallel_shared_memory_bytes,
        max_parallel_shared_memory_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
        reclaim_queue=runtime.reclaim_queue,
        state=runtime.shared_memory_state,
        workers=runtime.workers,
    )

    shm = _create_shared_memory_chunk(batch, size=required_size)
    runtime.shared_memory_state.outstanding_segments[shm.name] = (
        _OutstandingSharedMemory(
            db_id=db_id,
            size=required_size,
            handle=shm,
        )
    )
    runtime.shared_memory_state.outstanding_bytes += required_size
    runtime.shared_memory_state.outstanding_bytes_by_worker[db_id] += required_size

    chunk_ref = _SharedMemoryChunkRef(
        name=shm.name,
        size=required_size,
        row_count=len(batch),
    )
    try:
        runtime.queues[db_id].put(chunk_ref)
    except Exception:
        _release_shared_memory_segment(runtime.shared_memory_state, name=shm.name)
        raise

    runtime.chunk_bufs[db_id] = []
    runtime.chunk_byte_sizes[db_id] = 0


def _route_records_to_workers(
    runtime: _ParallelRuntime,
    *,
    records: Iterable[T],
    config: WriteConfig,
    num_dbs: int,
    key_fn: Callable[[T], KeyInput],
    value_fn: Callable[[T], bytes],
    columns_fn: Callable[[T], dict[str, Any]] | None,
    chunk_size: int,
    max_parallel_shared_memory_bytes: int | None,
    max_parallel_shared_memory_bytes_per_worker: int | None,
) -> None:
    for record in records:
        key = key_fn(record)
        routing_context = columns_fn(record) if columns_fn is not None else None
        db_id = route_key(
            key,
            num_dbs=num_dbs,
            sharding=config.sharding,
            routing_context=routing_context,
        )
        key_bytes = runtime.key_encoder(key)
        value_bytes = value_fn(record)
        pair_bytes = (
            _SHARED_MEMORY_RECORD_HEADER.size + len(key_bytes) + len(value_bytes)
        )
        if (
            runtime.chunk_bufs[db_id]
            and runtime.chunk_byte_sizes[db_id] + pair_bytes
            > _MAX_SHARED_MEMORY_CHUNK_BYTES
        ):
            _send_chunk(
                runtime,
                db_id=db_id,
                max_parallel_shared_memory_bytes=max_parallel_shared_memory_bytes,
                max_parallel_shared_memory_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
            )

        runtime.chunk_bufs[db_id].append((key_bytes, value_bytes))
        runtime.chunk_byte_sizes[db_id] += pair_bytes
        runtime.min_keys[db_id], runtime.max_keys[db_id] = update_min_max(
            runtime.min_keys[db_id], runtime.max_keys[db_id], key
        )
        runtime.row_counts[db_id] += 1

        if (
            len(runtime.chunk_bufs[db_id]) >= chunk_size
            or runtime.chunk_byte_sizes[db_id] >= _MAX_SHARED_MEMORY_CHUNK_BYTES
        ):
            _send_chunk(
                runtime,
                db_id=db_id,
                max_parallel_shared_memory_bytes=max_parallel_shared_memory_bytes,
                max_parallel_shared_memory_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
            )


def _finish_parallel_dispatch(
    runtime: _ParallelRuntime,
    *,
    num_dbs: int,
    max_parallel_shared_memory_bytes: int | None,
    max_parallel_shared_memory_bytes_per_worker: int | None,
) -> None:
    for db_id in range(num_dbs):
        _send_chunk(
            runtime,
            db_id=db_id,
            max_parallel_shared_memory_bytes=max_parallel_shared_memory_bytes,
            max_parallel_shared_memory_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
        )
        runtime.queues[db_id].put(None)


def _send_stop_signals(runtime: _ParallelRuntime, *, num_dbs: int) -> None:
    for db_id in range(num_dbs):
        try:
            runtime.queues[db_id].put_nowait(None)
        except Exception:
            pass


def _cleanup_parallel_runtime(runtime: _ParallelRuntime) -> tuple[str, ...]:
    _join_and_terminate_workers(runtime.workers)
    _drain_reclaimed_segments(runtime.reclaim_queue, runtime.shared_memory_state)
    unreclaimed_segment_names = tuple(runtime.shared_memory_state.outstanding_segments)
    _cleanup_outstanding_segments(runtime.shared_memory_state)
    return unreclaimed_segment_names


def _raise_if_worker_failed(workers: list[multiprocessing.Process]) -> None:
    failed = [
        (db_id, process.exitcode)
        for db_id, process in enumerate(workers)
        if process.exitcode
    ]
    if failed:
        raise ShardyfusionError(
            f"Worker process(es) exited with errors: {[(db_id, code) for db_id, code in failed]}"
        )


def _collect_parallel_results(
    runtime: _ParallelRuntime,
    *,
    num_dbs: int,
) -> list[ShardAttemptResult]:
    results: list[ShardAttemptResult] = []
    deadline = time.monotonic() + _RESULT_COLLECT_TIMEOUT.total_seconds() * num_dbs
    for _ in range(num_dbs):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ShardyfusionError(
                f"Timed out collecting results: got {len(results)}/{num_dbs}"
            )
        result = runtime.result_queue.get(timeout=max(remaining, 0.1))
        result.min_key = runtime.min_keys[result.db_id]
        result.max_key = runtime.max_keys[result.db_id]
        if runtime.row_counts[result.db_id] == 0:
            result.db_url = None
        results.append(result)
    return results


def _write_parallel(
    *,
    records: Iterable[T],
    config: WriteConfig,
    num_dbs: int,
    run_id: str,
    factory: DbAdapterFactory,
    key_fn: Callable[[T], KeyInput],
    value_fn: Callable[[T], bytes],
    columns_fn: Callable[[T], dict[str, Any]] | None = None,
    max_queue_size: int,
    max_parallel_shared_memory_bytes: int | None,
    max_parallel_shared_memory_bytes_per_worker: int | None,
    max_writes_per_second: float | None,
    max_write_bytes_per_second: float | None,
) -> list[ShardAttemptResult]:
    chunk_size = max(1, config.batch_size // _CHUNK_DIVISOR)
    runtime = _start_parallel_runtime(
        config=config,
        num_dbs=num_dbs,
        run_id=run_id,
        factory=factory,
        max_queue_size=max_queue_size,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )

    try:
        _route_records_to_workers(
            runtime,
            records=records,
            config=config,
            num_dbs=num_dbs,
            key_fn=key_fn,
            value_fn=value_fn,
            columns_fn=columns_fn,
            chunk_size=chunk_size,
            max_parallel_shared_memory_bytes=max_parallel_shared_memory_bytes,
            max_parallel_shared_memory_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
        )
        _finish_parallel_dispatch(
            runtime,
            num_dbs=num_dbs,
            max_parallel_shared_memory_bytes=max_parallel_shared_memory_bytes,
            max_parallel_shared_memory_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
        )
    except Exception:
        _send_stop_signals(runtime, num_dbs=num_dbs)
        raise
    finally:
        unreclaimed_segment_names = _cleanup_parallel_runtime(runtime)

    _raise_if_worker_failed(runtime.workers)
    if unreclaimed_segment_names:
        raise ShardyfusionError(
            "Parallel write completed but not all shared-memory segments were reclaimed: "
            f"{sorted(unreclaimed_segment_names)}"
        )
    return _collect_parallel_results(runtime, num_dbs=num_dbs)


def _join_and_terminate_workers(workers: list[multiprocessing.Process]) -> None:
    for process in workers:
        process.join(timeout=_WORKER_JOIN_TIMEOUT.total_seconds())
        if process.is_alive():
            process.terminate()
            process.join(timeout=_WORKER_TERMINATE_TIMEOUT.total_seconds())
            if process.is_alive():
                process.kill()
