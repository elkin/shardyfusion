"""Tests for Python writer shared-memory transport helpers."""

from __future__ import annotations

import multiprocessing
import queue
from pathlib import Path
from uuid import uuid4

import pytest

import shardyfusion.writer.python._parallel_writer as parallel_writer_mod
from shardyfusion._writer_core import ShardAttemptResult
from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
from shardyfusion.errors import ShardyfusionError
from shardyfusion.manifest import WriterInfo
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.writer.python import write_sharded
from shardyfusion.writer.python._parallel_writer import (
    _cleanup_parallel_runtime,
    _collect_parallel_results,
    _create_shared_memory_chunk,
    _extend_batch_from_shared_memory,
    _OutstandingSharedMemory,
    _ParallelRuntime,
    _release_shared_memory_segment,
    _send_chunk,
    _shared_memory_chunk_bytes,
    _SharedMemoryChunkRef,
    _SharedMemoryReclaim,
    _SharedMemoryState,
    _wait_for_reclaim_or_worker_exit,
    _wait_for_shared_memory_budget,
)


class _FakeSegmentHandle:
    def __init__(self) -> None:
        self.closed = False
        self.unlinked = False

    def close(self) -> None:
        self.closed = True

    def unlink(self) -> None:
        self.unlinked = True


class _HealthyWorker:
    exitcode: int | None = None


class _FailingQueue:
    def put(self, item: object) -> None:
        raise RuntimeError("queue explosion")


class _QueueWithReader:
    def __init__(self) -> None:
        self._reader = object()


class _DeadWorker:
    def __init__(self, exitcode: int) -> None:
        self.exitcode = exitcode
        self.sentinel = object()


def _make_parallel_config(tmp_path: Path) -> WriteConfig:
    from shardyfusion.testing import fake_adapter_factory

    return WriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        adapter_factory=fake_adapter_factory,
        output=OutputOptions(
            run_id="shared-memory-test",
            local_root=str(tmp_path / "local"),
        ),
        manifest=ManifestOptions(store=InMemoryManifestStore()),
    )


def test_shared_memory_chunk_round_trip() -> None:
    batch = [
        (b"alpha", b"one"),
        (b"beta", b""),
        (b"gamma", b"three"),
    ]
    size = _shared_memory_chunk_bytes(batch)
    shm = _create_shared_memory_chunk(batch, size=size)

    try:
        decoded: list[tuple[bytes, bytes]] = []
        _extend_batch_from_shared_memory(
            decoded,
            chunk=_SharedMemoryChunkRef(name=shm.name, size=size, row_count=3),
        )
        assert decoded == batch
    finally:
        shm.close()
        shm.unlink()


def test_extend_batch_from_shared_memory_raises_when_segment_missing() -> None:
    with pytest.raises(ShardyfusionError, match="was not available to the worker"):
        _extend_batch_from_shared_memory(
            [],
            chunk=_SharedMemoryChunkRef(
                name=f"missing-{uuid4().hex}",
                size=1,
                row_count=1,
            ),
        )


def test_extend_batch_from_shared_memory_rejects_declared_size_larger_than_segment() -> (
    None
):
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=1)

    try:
        with pytest.raises(ShardyfusionError, match="declared size 2"):
            _extend_batch_from_shared_memory(
                [],
                chunk=_SharedMemoryChunkRef(name=shm.name, size=2, row_count=0),
            )
    finally:
        shm.close()
        shm.unlink()


def test_extend_batch_from_shared_memory_rejects_truncated_header() -> None:
    shm = multiprocessing.shared_memory.SharedMemory(create=True, size=1)

    try:
        with pytest.raises(ShardyfusionError, match="ended mid-record header"):
            _extend_batch_from_shared_memory(
                [],
                chunk=_SharedMemoryChunkRef(name=shm.name, size=1, row_count=1),
            )
    finally:
        shm.close()
        shm.unlink()


def test_extend_batch_from_shared_memory_rejects_truncated_payload() -> None:
    batch = [(b"k", b"value")]
    size = _shared_memory_chunk_bytes(batch)
    shm = _create_shared_memory_chunk(batch, size=size)

    try:
        with pytest.raises(ShardyfusionError, match="ended mid-record payload"):
            _extend_batch_from_shared_memory(
                [],
                chunk=_SharedMemoryChunkRef(
                    name=shm.name,
                    size=size - 1,
                    row_count=1,
                ),
            )
    finally:
        shm.close()
        shm.unlink()


def test_extend_batch_from_shared_memory_rejects_trailing_bytes() -> None:
    batch = [(b"k", b"value")]
    size = _shared_memory_chunk_bytes(batch)
    shm = _create_shared_memory_chunk(batch, size=size)

    try:
        with pytest.raises(ShardyfusionError, match="had trailing bytes"):
            _extend_batch_from_shared_memory(
                [],
                chunk=_SharedMemoryChunkRef(name=shm.name, size=size, row_count=0),
            )
    finally:
        shm.close()
        shm.unlink()


def test_release_shared_memory_segment_updates_budget_counters() -> None:
    handle = _FakeSegmentHandle()
    state = _SharedMemoryState(
        outstanding_segments={
            "seg-1": _OutstandingSharedMemory(
                db_id=0,
                size=128,
                handle=handle,  # type: ignore[arg-type]
            )
        },
        outstanding_bytes=128,
        outstanding_bytes_by_worker=[128],
    )

    _release_shared_memory_segment(state, name="seg-1")

    assert state.outstanding_segments == {}
    assert state.outstanding_bytes == 0
    assert state.outstanding_bytes_by_worker == [0]
    assert handle.closed is True
    assert handle.unlinked is True


def test_send_chunk_releases_shared_memory_when_queue_put_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _FakeSegmentHandle()
    handle.name = "seg-1"  # type: ignore[attr-defined]
    batch = [(b"alpha", b"one")]
    size = _shared_memory_chunk_bytes(batch)

    monkeypatch.setattr(
        parallel_writer_mod,
        "_create_shared_memory_chunk",
        lambda batch, *, size: handle,  # type: ignore[arg-type]
    )

    runtime = _ParallelRuntime(
        queues=[_FailingQueue()],  # type: ignore[list-item]
        reclaim_queue=queue.Queue(),  # type: ignore[arg-type]
        result_queue=queue.Queue(),  # type: ignore[arg-type]
        workers=[_HealthyWorker()],  # type: ignore[list-item]
        chunk_bufs=[batch.copy()],
        chunk_byte_sizes=[size],
        min_keys=[None],
        max_keys=[None],
        row_counts=[1],
        key_encoder=lambda key: b"",
        shared_memory_state=_SharedMemoryState(
            outstanding_segments={},
            outstanding_bytes=0,
            outstanding_bytes_by_worker=[0],
        ),
    )

    with pytest.raises(RuntimeError, match="queue explosion"):
        _send_chunk(
            runtime,
            db_id=0,
            max_parallel_shared_memory_bytes=1024,
            max_parallel_shared_memory_bytes_per_worker=1024,
        )

    assert runtime.shared_memory_state.outstanding_segments == {}
    assert runtime.shared_memory_state.outstanding_bytes == 0
    assert runtime.shared_memory_state.outstanding_bytes_by_worker == [0]
    assert handle.closed is True
    assert handle.unlinked is True
    assert runtime.chunk_bufs[0] == batch
    assert runtime.chunk_byte_sizes[0] == size


def test_wait_for_shared_memory_budget_drains_reclaim_queue() -> None:
    handle = _FakeSegmentHandle()
    state = _SharedMemoryState(
        outstanding_segments={
            "seg-1": _OutstandingSharedMemory(
                db_id=0,
                size=64,
                handle=handle,  # type: ignore[arg-type]
            )
        },
        outstanding_bytes=64,
        outstanding_bytes_by_worker=[64],
    )
    reclaim_queue: queue.Queue[_SharedMemoryReclaim] = queue.Queue()
    reclaim_queue.put(_SharedMemoryReclaim(name="seg-1"))

    _wait_for_shared_memory_budget(
        db_id=0,
        required_size=32,
        max_parallel_shared_memory_bytes=64,
        max_parallel_shared_memory_bytes_per_worker=64,
        reclaim_queue=reclaim_queue,  # type: ignore[arg-type]
        state=state,
        workers=[_HealthyWorker()],  # type: ignore[list-item]
    )

    assert state.outstanding_bytes == 0
    assert state.outstanding_bytes_by_worker == [0]
    assert handle.closed is True
    assert handle.unlinked is True


def test_wait_for_reclaim_or_worker_exit_raises_when_worker_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _DeadWorker(exitcode=1)
    monkeypatch.setattr(
        parallel_writer_mod,
        "wait_for_handles",
        lambda handles: [worker.sentinel],
    )

    with pytest.raises(ShardyfusionError, match="exited with errors"):
        _wait_for_reclaim_or_worker_exit(
            reclaim_queue=_QueueWithReader(),  # type: ignore[arg-type]
            state=_SharedMemoryState(
                outstanding_segments={},
                outstanding_bytes=0,
                outstanding_bytes_by_worker=[0],
            ),
            workers=[worker],  # type: ignore[list-item]
        )


def test_wait_for_shared_memory_budget_is_per_worker() -> None:
    state = _SharedMemoryState(
        outstanding_segments={},
        outstanding_bytes=48,
        outstanding_bytes_by_worker=[48, 0],
    )

    _wait_for_shared_memory_budget(
        db_id=1,
        required_size=16,
        max_parallel_shared_memory_bytes=128,
        max_parallel_shared_memory_bytes_per_worker=32,
        reclaim_queue=queue.Queue(),  # type: ignore[arg-type]
        state=state,
        workers=[_HealthyWorker(), _HealthyWorker()],  # type: ignore[list-item]
    )

    assert state.outstanding_bytes == 48
    assert state.outstanding_bytes_by_worker == [48, 0]


def test_wait_for_shared_memory_budget_rejects_large_global_limit() -> None:
    with pytest.raises(ShardyfusionError, match="max_parallel_shared_memory_bytes"):
        _wait_for_shared_memory_budget(
            db_id=0,
            required_size=65,
            max_parallel_shared_memory_bytes=64,
            max_parallel_shared_memory_bytes_per_worker=128,
            reclaim_queue=queue.Queue(),  # type: ignore[arg-type]
            state=_SharedMemoryState(
                outstanding_segments={},
                outstanding_bytes=0,
                outstanding_bytes_by_worker=[0],
            ),
            workers=[_HealthyWorker()],  # type: ignore[list-item]
        )


def test_cleanup_parallel_runtime_returns_unreclaimed_segment_names() -> None:
    handle = _FakeSegmentHandle()
    state = _SharedMemoryState(
        outstanding_segments={
            "seg-1": _OutstandingSharedMemory(
                db_id=0,
                size=128,
                handle=handle,  # type: ignore[arg-type]
            )
        },
        outstanding_bytes=128,
        outstanding_bytes_by_worker=[128],
    )
    runtime = _ParallelRuntime(
        queues=[],
        reclaim_queue=queue.Queue(),  # type: ignore[arg-type]
        result_queue=queue.Queue(),  # type: ignore[arg-type]
        workers=[],
        chunk_bufs=[],
        chunk_byte_sizes=[],
        min_keys=[],
        max_keys=[],
        row_counts=[],
        key_encoder=lambda key: b"",
        shared_memory_state=state,
    )

    unreclaimed = _cleanup_parallel_runtime(runtime)

    assert unreclaimed == ("seg-1",)
    assert state.outstanding_segments == {}
    assert state.outstanding_bytes == 0
    assert state.outstanding_bytes_by_worker == [0]
    assert handle.closed is True
    assert handle.unlinked is True


def test_collect_parallel_results_patches_min_max_and_marks_empty_shards() -> None:
    result_queue: queue.Queue[ShardAttemptResult] = queue.Queue()
    result_queue.put(
        ShardAttemptResult(
            db_id=0,
            db_url="s3://bucket/db0",
            attempt=0,
            row_count=3,
            min_key=None,
            max_key=None,
            checkpoint_id="cp0",
            writer_info=WriterInfo(attempt=0),
        )
    )
    result_queue.put(
        ShardAttemptResult(
            db_id=1,
            db_url="s3://bucket/db1",
            attempt=0,
            row_count=0,
            min_key=None,
            max_key=None,
            checkpoint_id=None,
            writer_info=WriterInfo(attempt=0),
        )
    )

    runtime = _ParallelRuntime(
        queues=[],
        reclaim_queue=queue.Queue(),  # type: ignore[arg-type]
        result_queue=result_queue,  # type: ignore[arg-type]
        workers=[],
        chunk_bufs=[],
        chunk_byte_sizes=[],
        min_keys=[10, None],
        max_keys=[12, None],
        row_counts=[3, 0],
        key_encoder=lambda key: b"",
        shared_memory_state=_SharedMemoryState(
            outstanding_segments={},
            outstanding_bytes=0,
            outstanding_bytes_by_worker=[0, 0],
        ),
    )

    results = _collect_parallel_results(runtime, num_dbs=2)

    assert [(result.db_id, result.min_key, result.max_key) for result in results] == [
        (0, 10, 12),
        (1, None, None),
    ]
    assert results[0].db_url == "s3://bucket/db0"
    assert results[1].db_url is None


def test_parallel_shared_memory_limit_rejects_large_record(tmp_path: Path) -> None:
    config = _make_parallel_config(tmp_path)

    with pytest.raises(
        ShardyfusionError,
        match="max_parallel_shared_memory_bytes_per_worker",
    ):
        write_sharded(
            [0],
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: b"x" * 256,
            parallel=True,
            max_parallel_shared_memory_bytes=1024,
            max_parallel_shared_memory_bytes_per_worker=64,
        )
