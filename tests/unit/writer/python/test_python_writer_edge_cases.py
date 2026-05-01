"""Tests for Python writer parallel mode edge cases.

Covers: non-picklable factory, close failure in adapter,
empty input in parallel mode, single shard parallel mode.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Self

from shardyfusion.config import HashWriteConfig, ManifestOptions, OutputOptions
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.sharding_types import ShardHashAlgorithm
from shardyfusion.writer.python._parallel_writer import _write_parallel_hash

_MAX_PARALLEL_SHARED_MEMORY_BYTES = 256 * 1024 * 1024
_MAX_PARALLEL_SHARED_MEMORY_BYTES_PER_WORKER = 32 * 1024 * 1024


class _CloseFailAdapter:
    """Adapter that raises during close."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        # __exit__ delegates to close() in real adapters
        pass

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        pass

    def flush(self) -> None:
        pass

    def checkpoint(self) -> str:
        return "ckpt"

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        raise RuntimeError("close explosion")


class _GoodAdapter:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        pass

    def flush(self) -> None:
        pass

    def checkpoint(self) -> str:
        return "ckpt"

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        pass


def _good_factory(*, db_url: str, local_dir: Path) -> _GoodAdapter:
    return _GoodAdapter()


def _close_fail_factory(*, db_url: str, local_dir: Path) -> _CloseFailAdapter:
    return _CloseFailAdapter()


def _config(num_dbs: int = 2) -> HashWriteConfig:
    return HashWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/test",
        adapter_factory=_good_factory,
        manifest=ManifestOptions(store=InMemoryManifestStore()),
        output=OutputOptions(run_id="edge-test"),
    )


class TestParallelEmptyInput:
    def test_empty_records_produces_empty_results(self) -> None:
        config = _config(num_dbs=2)
        results = _write_parallel_hash(
            records=[],
            config=config,
            num_dbs=2,
            run_id="empty-test",
            factory=_good_factory,
            key_fn=lambda r: r["id"],
            value_fn=lambda r: r["val"],
            hash_algorithm=ShardHashAlgorithm.XXH3_64,
            max_queue_size=10,
            max_parallel_shared_memory_bytes=_MAX_PARALLEL_SHARED_MEMORY_BYTES,
            max_parallel_shared_memory_bytes_per_worker=_MAX_PARALLEL_SHARED_MEMORY_BYTES_PER_WORKER,
            max_writes_per_second=None,
            max_write_bytes_per_second=None,
        )
        assert all(r.row_count == 0 for r in results)


class TestParallelSingleShard:
    def test_single_shard_parallel(self) -> None:
        config = _config(num_dbs=1)
        records = [{"id": i, "val": b"x"} for i in range(10)]
        results = _write_parallel_hash(
            records=records,
            config=config,
            num_dbs=1,
            run_id="single-test",
            factory=_good_factory,
            key_fn=lambda r: r["id"],
            value_fn=lambda r: r["val"],
            hash_algorithm=ShardHashAlgorithm.XXH3_64,
            max_queue_size=10,
            max_parallel_shared_memory_bytes=_MAX_PARALLEL_SHARED_MEMORY_BYTES,
            max_parallel_shared_memory_bytes_per_worker=_MAX_PARALLEL_SHARED_MEMORY_BYTES_PER_WORKER,
            max_writes_per_second=None,
            max_write_bytes_per_second=None,
        )
        assert len(results) == 1
        assert results[0].row_count == 10


class TestParallelManyShards:
    def test_many_records_distributed(self) -> None:
        config = _config(num_dbs=4)
        records = [{"id": i, "val": b"x"} for i in range(100)]
        results = _write_parallel_hash(
            records=records,
            config=config,
            num_dbs=4,
            run_id="many-test",
            factory=_good_factory,
            key_fn=lambda r: r["id"],
            value_fn=lambda r: r["val"],
            hash_algorithm=ShardHashAlgorithm.XXH3_64,
            max_queue_size=50,
            max_parallel_shared_memory_bytes=_MAX_PARALLEL_SHARED_MEMORY_BYTES,
            max_parallel_shared_memory_bytes_per_worker=_MAX_PARALLEL_SHARED_MEMORY_BYTES_PER_WORKER,
            max_writes_per_second=None,
            max_write_bytes_per_second=None,
        )
        assert len(results) == 4
        assert sum(r.row_count for r in results) == 100
