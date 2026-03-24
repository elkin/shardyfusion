"""Tests for Python writer shared-memory transport helpers."""

from __future__ import annotations

import queue
from pathlib import Path

import pytest

from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
from shardyfusion.errors import ShardyfusionError
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.writer.python import write_sharded
from shardyfusion.writer.python._parallel_writer import (
    _create_shared_memory_chunk,
    _extend_batch_from_shared_memory,
    _OutstandingSharedMemory,
    _release_shared_memory_segment,
    _shared_memory_chunk_bytes,
    _SharedMemoryChunkRef,
    _SharedMemoryReclaim,
    _SharedMemoryState,
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
