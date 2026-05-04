"""Tests for Python writer parallel mode error paths."""

from __future__ import annotations

import os
import queue
from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import pytest

import shardyfusion.writer.python._parallel_writer as parallel_writer_mod
from shardyfusion.config import (
    HashShardedWriteConfig,
    WriterManifestConfig,
    WriterOutputConfig,
)
from shardyfusion.errors import ShardyfusionError
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.run_registry import InMemoryRunRegistry
from shardyfusion.serde import make_key_encoder
from shardyfusion.sharding_types import ShardHashAlgorithm
from shardyfusion.testing import FileBackedSlateDbAdapter, file_backed_load_db
from shardyfusion.type_defs import RetryConfig
from shardyfusion.writer.python._parallel_writer import (
    _file_shard_worker,
    _FileChunkRef,
    _write_parallel_hash,
)
from tests.helpers.run_record_assertions import (
    assert_success_run_record,
    load_in_memory_run_record,
)
from tests.helpers.tracking import TrackingFactory
from tests.helpers.writer_api import write_python_hash_sharded as write_hash_sharded

_MAX_PARALLEL_SHARED_MEMORY_BYTES = 256 * 1024 * 1024
_MAX_PARALLEL_SHARED_MEMORY_BYTES_PER_WORKER = 32 * 1024 * 1024


class _CrashingAdapter:
    """Adapter that raises after a few writes to simulate worker crash."""

    def __init__(self, crash_after: int = 2) -> None:
        self._count = 0
        self._crash_after = crash_after

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        self._count += 1
        if self._count >= self._crash_after:
            raise RuntimeError("simulated adapter crash")

    def flush(self) -> None:
        pass

    def seal(self) -> None:
        return None

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        pass


class _GoodAdapter:
    """Adapter that works normally."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        pass

    def flush(self) -> None:
        pass

    def seal(self) -> None:
        return None

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        pass


def _crashing_factory(*, db_url: str, local_dir: Path) -> _CrashingAdapter:
    return _CrashingAdapter(crash_after=1)


def _good_factory(*, db_url: str, local_dir: Path) -> _GoodAdapter:
    return _GoodAdapter()


class _FailOnceMarkerAdapter:
    def __init__(self, marker_path: str) -> None:
        self._marker_path = Path(marker_path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        if not self._marker_path.exists():
            self._marker_path.parent.mkdir(parents=True, exist_ok=True)
            self._marker_path.write_text("failed", encoding="utf-8")
            raise RuntimeError("transient retryable failure")

    def flush(self) -> None:
        pass

    def seal(self) -> None:
        return None

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        pass


class _FailOnceMarkerFactory:
    def __init__(self, marker_path: str) -> None:
        self._marker_path = marker_path

    def __call__(self, *, db_url: str, local_dir: Path) -> _FailOnceMarkerAdapter:
        return _FailOnceMarkerAdapter(self._marker_path)


class _ExitOnceMarkerAdapter:
    def __init__(self, marker_path: str) -> None:
        self._marker_path = Path(marker_path)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        pass

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        if not self._marker_path.exists():
            self._marker_path.parent.mkdir(parents=True, exist_ok=True)
            self._marker_path.write_text("exited", encoding="utf-8")
            os._exit(17)

    def flush(self) -> None:
        pass

    def seal(self) -> None:
        return None

    def db_bytes(self) -> int:
        return 0

    def close(self) -> None:
        pass


class _ExitOnceMarkerFactory:
    def __init__(self, marker_path: str) -> None:
        self._marker_path = marker_path

    def __call__(self, *, db_url: str, local_dir: Path) -> _ExitOnceMarkerAdapter:
        return _ExitOnceMarkerAdapter(self._marker_path)


class _FailOnceFileBackedAdapter(FileBackedSlateDbAdapter):
    def __init__(
        self,
        root_dir: str,
        marker_path: str,
        *,
        db_url: str,
        local_dir: Path,
    ) -> None:
        super().__init__(root_dir, db_url=db_url, local_dir=local_dir)
        self._marker_path = Path(marker_path)

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        super().write_batch(pairs)
        if not self._marker_path.exists():
            self._marker_path.parent.mkdir(parents=True, exist_ok=True)
            self._marker_path.write_text("failed", encoding="utf-8")
            raise RuntimeError("transient retryable failure")


class _FailOnceFileBackedFactory:
    def __init__(self, root_dir: str, marker_path: str) -> None:
        self._root_dir = root_dir
        self._marker_path = marker_path

    def __call__(self, *, db_url: str, local_dir: Path) -> _FailOnceFileBackedAdapter:
        return _FailOnceFileBackedAdapter(
            self._root_dir,
            self._marker_path,
            db_url=db_url,
            local_dir=local_dir,
        )


def _make_config(
    num_dbs: int = 2,
    *,
    run_id: str = "test-run",
    local_root: str | None = None,
    shard_retry: RetryConfig | None = None,
    run_registry: InMemoryRunRegistry | None = None,
) -> HashShardedWriteConfig:
    return HashShardedWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/test",
        adapter_factory=_good_factory,
        manifest=WriterManifestConfig(store=InMemoryManifestStore()),
        output=WriterOutputConfig(
            run_id=run_id,
            local_root=local_root or "/tmp/shardyfusion-tests",
        ),
        shard_retry=shard_retry,
        run_registry=run_registry,
    )


def test_worker_crash_raises_error() -> None:
    """Worker process crash is detected and raises ShardyfusionError."""
    config = HashShardedWriteConfig(
        num_dbs=2,
        s3_prefix="s3://bucket/test",
        adapter_factory=_crashing_factory,
        manifest=WriterManifestConfig(store=InMemoryManifestStore()),
        output=WriterOutputConfig(run_id="crash-test"),
    )

    records = [{"id": i, "val": b"x"} for i in range(100)]

    with pytest.raises(ShardyfusionError):
        _write_parallel_hash(
            records=records,
            config=config,
            num_dbs=config.num_dbs,
            run_id="crash-test",
            factory=_crashing_factory,
            key_fn=lambda r: r["id"],
            value_fn=lambda r: r["val"],
            hash_algorithm=ShardHashAlgorithm.XXH3_64,
            max_queue_size=10,
            max_parallel_shared_memory_bytes=_MAX_PARALLEL_SHARED_MEMORY_BYTES,
            max_parallel_shared_memory_bytes_per_worker=_MAX_PARALLEL_SHARED_MEMORY_BYTES_PER_WORKER,
            max_writes_per_second=None,
            max_write_bytes_per_second=None,
        )


def test_parallel_succeeds_with_good_factory() -> None:
    """Parallel mode completes successfully with well-behaved adapters."""
    config = _make_config(num_dbs=2)
    records = [{"id": i, "val": b"x"} for i in range(20)]

    results = _write_parallel_hash(
        records=records,
        config=config,
        num_dbs=config.num_dbs,
        run_id="good-test",
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

    assert len(results) == 2
    assert sum(r.row_count for r in results) == 20


def test_file_shard_worker_returns_empty_result_for_empty_shard(tmp_path: Path) -> None:
    config = _make_config(
        num_dbs=1,
        run_id="file-worker-empty",
        local_root=str(tmp_path),
    )
    work_queue: queue.Queue[_FileChunkRef | None] = queue.Queue()
    result_queue: queue.Queue[object] = queue.Queue()
    failure_queue: queue.Queue[object] = queue.Queue()
    work_queue.put(None)

    _file_shard_worker(
        0,
        work_queue,  # type: ignore[arg-type]
        result_queue,  # type: ignore[arg-type]
        failure_queue,  # type: ignore[arg-type]
        config,
        "file-worker-empty",
        TrackingFactory(),
        0,
        None,
        None,
    )

    result = result_queue.get_nowait()

    assert result.db_id == 0
    assert result.db_url is None
    assert result.row_count == 0
    assert result.checkpoint_id is None
    assert failure_queue.empty()


def test_file_shard_worker_replays_spool_file_and_reports_result(
    tmp_path: Path,
) -> None:
    factory = TrackingFactory()
    config = _make_config(
        num_dbs=1,
        run_id="file-worker-success",
        local_root=str(tmp_path / "local"),
    )
    config.batch_size = 2
    spool_path = tmp_path / "spool.bin"
    batch = [(b"k1", b"v1"), (b"k2", b"v2"), (b"k3", b"v3")]
    spool_path.write_bytes(parallel_writer_mod._serialize_batch(batch))
    work_queue: queue.Queue[_FileChunkRef | None] = queue.Queue()
    result_queue: queue.Queue[object] = queue.Queue()
    failure_queue: queue.Queue[object] = queue.Queue()
    work_queue.put(
        _FileChunkRef(
            path=str(spool_path),
            offset=0,
            size=spool_path.stat().st_size,
            row_count=len(batch),
        )
    )
    work_queue.put(None)

    _file_shard_worker(
        0,
        work_queue,  # type: ignore[arg-type]
        result_queue,  # type: ignore[arg-type]
        failure_queue,  # type: ignore[arg-type]
        config,
        "file-worker-success",
        factory,
        0,
        None,
        None,
    )

    result = result_queue.get_nowait()
    adapter = factory.adapters[0]

    assert [len(call) for call in adapter.write_calls] == [2, 1]
    assert adapter.flushed is True
    assert result.row_count == 3
    assert result.checkpoint_id is not None
    # Writer now stamps a shardyfusion-generated uuid4 hex (32 chars).
    assert (
        isinstance(result.checkpoint_id, str)
        and len(result.checkpoint_id) == 32
        and all(c in "0123456789abcdef" for c in result.checkpoint_id)
    )
    assert (
        result.db_url
        == "s3://bucket/test/shards/run_id=file-worker-success/db=00000/attempt=00"
    )
    assert failure_queue.empty()


def test_parallel_retry_replays_to_new_attempt_and_cleans_spool(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "retry-once.marker"
    local_root = tmp_path / "local"
    config = _make_config(
        num_dbs=1,
        run_id="retry-file-spool",
        local_root=str(local_root),
        shard_retry=RetryConfig(
            max_retries=1,
            initial_backoff=timedelta(seconds=0),
        ),
    )
    records = [{"id": i, "val": b"x"} for i in range(10)]

    results = _write_parallel_hash(
        records=records,
        config=config,
        num_dbs=1,
        run_id="retry-file-spool",
        factory=_FailOnceMarkerFactory(str(marker_path)),
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
    assert results[0].attempt == 1
    assert results[0].row_count == 10
    assert results[0].all_attempt_urls == (
        "s3://bucket/test/shards/run_id=retry-file-spool/db=00000/attempt=00",
        "s3://bucket/test/shards/run_id=retry-file-spool/db=00000/attempt=01",
    )
    assert not (local_root / "run_id=retry-file-spool" / "_parallel_spool").exists()


def test_write_sharded_parallel_retry_records_succeeded_run_record(
    tmp_path: Path,
) -> None:
    registry = InMemoryRunRegistry()
    marker_path = tmp_path / "retry-run-record.marker"
    local_root = tmp_path / "local"
    config = _make_config(
        num_dbs=1,
        run_id="retry-run-record",
        local_root=str(local_root),
        shard_retry=RetryConfig(
            max_retries=1,
            initial_backoff=timedelta(seconds=0),
        ),
        run_registry=registry,
    )
    config.adapter_factory = _FailOnceMarkerFactory(str(marker_path))

    result = write_hash_sharded(
        [{"id": i, "val": b"x"} for i in range(10)],
        config,
        key_fn=lambda r: r["id"],
        value_fn=lambda r: r["val"],
        parallel=True,
    )

    assert len(result.winners) == 1
    assert result.winners[0].attempt == 1

    run_record = load_in_memory_run_record(registry, result)
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="python",
        s3_prefix=config.s3_prefix,
    )


def test_parallel_retry_replay_preserves_final_shard_contents(tmp_path: Path) -> None:
    marker_path = tmp_path / "retry-contract.marker"
    file_backed_root = str(tmp_path / "file-backed")
    local_root = tmp_path / "local"
    config = _make_config(
        num_dbs=1,
        run_id="retry-contract",
        local_root=str(local_root),
        shard_retry=RetryConfig(
            max_retries=1,
            initial_backoff=timedelta(seconds=0),
        ),
    )
    records = [{"id": i, "val": f"value-{i}".encode()} for i in range(12)]

    results = _write_parallel_hash(
        records=records,
        config=config,
        num_dbs=1,
        run_id="retry-contract",
        factory=_FailOnceFileBackedFactory(file_backed_root, str(marker_path)),
        key_fn=lambda r: r["id"],
        value_fn=lambda r: r["val"],
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
        max_queue_size=10,
        max_parallel_shared_memory_bytes=_MAX_PARALLEL_SHARED_MEMORY_BYTES,
        max_parallel_shared_memory_bytes_per_worker=_MAX_PARALLEL_SHARED_MEMORY_BYTES_PER_WORKER,
        max_writes_per_second=None,
        max_write_bytes_per_second=None,
    )

    result = results[0]
    encoder = make_key_encoder(config.key_encoding)
    expected = {encoder(record["id"]): record["val"] for record in records}

    assert result.attempt == 1
    assert result.row_count == len(records)
    assert result.all_attempt_urls == (
        "s3://bucket/test/shards/run_id=retry-contract/db=00000/attempt=00",
        "s3://bucket/test/shards/run_id=retry-contract/db=00000/attempt=01",
    )
    assert result.db_url is not None
    assert file_backed_load_db(file_backed_root, result.db_url) == expected


def test_parallel_retry_respawns_worker_on_unexpected_exit(tmp_path: Path) -> None:
    marker_path = tmp_path / "exit-once.marker"
    config = _make_config(
        num_dbs=1,
        run_id="retry-worker-exit",
        local_root=str(tmp_path / "local"),
        shard_retry=RetryConfig(
            max_retries=1,
            initial_backoff=timedelta(seconds=0),
        ),
    )
    records = [{"id": i, "val": b"x"} for i in range(10)]

    results = _write_parallel_hash(
        records=records,
        config=config,
        num_dbs=1,
        run_id="retry-worker-exit",
        factory=_ExitOnceMarkerFactory(str(marker_path)),
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
    assert results[0].attempt == 1
    assert results[0].row_count == 10
    assert results[0].all_attempt_urls == (
        "s3://bucket/test/shards/run_id=retry-worker-exit/db=00000/attempt=00",
        "s3://bucket/test/shards/run_id=retry-worker-exit/db=00000/attempt=01",
    )


def test_send_retry_chunk_waits_to_record_chunk_until_queue_accepts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(
        num_dbs=1,
        run_id="queue-after-record",
        local_root=str(tmp_path / "local"),
        shard_retry=RetryConfig(
            max_retries=1,
            initial_backoff=timedelta(seconds=0),
        ),
    )
    runtime = SimpleNamespace(
        chunk_bufs=[[(b"k", b"v")]],
        chunk_byte_sizes=[
            parallel_writer_mod._SHARED_MEMORY_RECORD_HEADER.size
            + len(b"k")
            + len(b"v")
        ],
        spool_paths=[tmp_path / "spool.bin"],
        chunk_refs_by_db=[[]],
        attempt_urls_by_db=[[]],
        config=config,
        run_id="queue-after-record",
        attempts=[0],
    )

    call_count = 0

    def _fake_put_retry_message(
        runtime_obj: object,
        *,
        db_id: int,
        message: _FileChunkRef | None,
        expected_attempt: int | None = None,
    ) -> bool:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            assert runtime_obj.chunk_refs_by_db[db_id] == []
            assert runtime_obj.attempt_urls_by_db[db_id] == []
            assert message is not None
            return False
        return True

    monkeypatch.setattr(
        parallel_writer_mod,
        "_put_retry_message",
        _fake_put_retry_message,
    )

    parallel_writer_mod._send_retry_chunk(runtime, db_id=0)

    assert call_count == 2
    assert len(runtime.chunk_refs_by_db[0]) == 1
    assert runtime.attempt_urls_by_db[0] == [
        "s3://bucket/test/shards/run_id=queue-after-record/db=00000/attempt=00"
    ]


class _ReplayQueueShouldNotBlock:
    def put(self, item: object, timeout: float | None = None) -> None:
        raise AssertionError("replay should route through _put_retry_message")


class _ReplayProcess:
    exitcode: int | None = None

    def start(self) -> None:
        pass


class _ReplayContext:
    def Queue(self, maxsize: int) -> _ReplayQueueShouldNotBlock:
        return _ReplayQueueShouldNotBlock()

    def Process(self, *, target: object, args: tuple[object, ...]) -> _ReplayProcess:
        return _ReplayProcess()


def test_spawn_retry_worker_stops_replay_when_newer_attempt_takes_over(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _make_config(
        num_dbs=1,
        run_id="replay-takeover",
        local_root=str(tmp_path / "local"),
        shard_retry=RetryConfig(
            max_retries=2,
            initial_backoff=timedelta(seconds=0),
        ),
    )
    runtime = SimpleNamespace(
        ctx=_ReplayContext(),
        max_queue_size=1,
        result_queue=object(),
        failure_queue=object(),
        config=config,
        run_id="replay-takeover",
        factory=_good_factory,
        attempts=[1],
        max_writes_per_second=None,
        max_write_bytes_per_second=None,
        queues=[None],
        workers=[None],
        chunk_refs_by_db=[
            [
                _FileChunkRef(path="spool.bin", offset=0, size=1, row_count=1),
                _FileChunkRef(path="spool.bin", offset=1, size=1, row_count=1),
            ]
        ],
        attempt_urls_by_db=[[]],
        dispatch_finished=[True],
    )

    calls: list[_FileChunkRef | None] = []

    def _fake_put_retry_message(
        runtime_obj: object,
        *,
        db_id: int,
        message: _FileChunkRef | None,
        expected_attempt: int | None = None,
    ) -> bool:
        calls.append(message)
        return False

    monkeypatch.setattr(
        parallel_writer_mod,
        "_put_retry_message",
        _fake_put_retry_message,
    )

    parallel_writer_mod._spawn_retry_worker(runtime, db_id=0)

    assert len(calls) == 1
    assert runtime.attempt_urls_by_db[0] == []


def test_retry_result_timeout_includes_retry_backoff_budget() -> None:
    timeout_s = parallel_writer_mod._retry_result_timeout_seconds(
        retry_config=RetryConfig(
            max_retries=4,
            initial_backoff=timedelta(seconds=1),
            backoff_multiplier=2.0,
        ),
        num_dbs=1,
    )

    assert timeout_s == pytest.approx(25.0)
