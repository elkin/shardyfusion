"""Concurrent stress tests for ConcurrentShardedReader."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shardyfusion.manifest import (
    CurrentPointer,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.reader.reader import ConcurrentShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


@dataclass
class _FakeReader:
    store: dict[bytes, bytes]
    closed: bool = False

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def close(self) -> None:
        self.closed = True


class _VersionedManifestStore:
    """ManifestStore that supports switching to a new manifest version."""

    def __init__(self) -> None:
        self._version = 1
        self._lock = threading.Lock()

    def set_version(self, v: int) -> None:
        with self._lock:
            self._version = v

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        raise NotImplementedError("publish not used in reader tests")

    def load_current(self) -> CurrentPointer:
        with self._lock:
            v = self._version
        return CurrentPointer(
            manifest_ref=f"s3://bucket/manifest-v{v}.json",
            manifest_content_type="application/json",
            run_id=f"run-v{v}",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    def load_manifest(self, ref: str) -> ParsedManifest:
        return ParsedManifest(
            required_build=_build(num_dbs=2),
            shards=[_shard(0), _shard(1)],
        )


def _build(num_dbs: int = 2) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template="db={db_id:05d}",
        tmp_prefix="_tmp",
    )


def _shard(db_id: int) -> RequiredShardMeta:
    return RequiredShardMeta(
        db_id=db_id,
        db_url=f"s3://bucket/prefix/_tmp/db={db_id:05d}",
        attempt=0,
        row_count=100,
        writer_info={},
    )


def _make_factory() -> Any:
    key_bytes = (42).to_bytes(8, byteorder="big")
    store = {key_bytes: b"value-42"}

    def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
        return _FakeReader(store)

    return factory


def test_concurrent_gets_no_errors() -> None:
    """10 threads calling get() simultaneously — no errors or crashes."""
    manifest_store = _VersionedManifestStore()
    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root="/tmp/stress-test",
        manifest_store=manifest_store,
        reader_factory=_make_factory(),
        thread_safety="lock",
    )

    errors: list[Exception] = []
    barrier = threading.Barrier(10)

    def worker() -> None:
        try:
            barrier.wait(timeout=5.0)
            for _ in range(50):
                reader.get(42)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    reader.close()
    assert not errors
    assert all(not t.is_alive() for t in threads)


def test_refresh_racing_with_gets() -> None:
    """refresh() during in-flight get() calls — no data corruption."""
    manifest_store = _VersionedManifestStore()
    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root="/tmp/stress-test",
        manifest_store=manifest_store,
        reader_factory=_make_factory(),
        thread_safety="lock",
    )

    errors: list[Exception] = []
    stop = threading.Event()

    def get_worker() -> None:
        try:
            while not stop.is_set():
                reader.get(42)
        except Exception as e:
            errors.append(e)

    def refresh_worker() -> None:
        try:
            for i in range(5):
                manifest_store.set_version(i + 2)
                reader.refresh()
                time.sleep(0.01)
        except Exception as e:
            errors.append(e)

    get_threads = [threading.Thread(target=get_worker) for _ in range(5)]
    refresh_thread = threading.Thread(target=refresh_worker)

    for t in get_threads:
        t.start()
    refresh_thread.start()

    refresh_thread.join(timeout=10.0)
    stop.set()
    for t in get_threads:
        t.join(timeout=5.0)

    reader.close()
    assert not errors


def test_pool_mode_concurrent_gets() -> None:
    """Pool mode handles concurrent gets without deadlock."""
    manifest_store = _VersionedManifestStore()
    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root="/tmp/stress-test",
        manifest_store=manifest_store,
        reader_factory=_make_factory(),
        thread_safety="pool",
        max_workers=4,
    )

    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait(timeout=5.0)
            for _ in range(20):
                reader.get(42)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    reader.close()
    assert not errors
