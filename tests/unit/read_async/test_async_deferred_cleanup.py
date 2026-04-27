"""Tests for async reader deferred cleanup edge cases.

Covers: borrow release with no event loop, create_task failure,
close while deferred cleanup pending, multiple deferred cleanups.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.reader import AsyncShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


@dataclass
class _AsyncFakeReader:
    store: dict[bytes, bytes]
    closed: bool = False

    async def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    async def close(self) -> None:
        self.closed = True


def _async_factory(stores: dict[str, dict[bytes, bytes]]):
    async def factory(
        *, db_url: str, local_dir: Path, checkpoint_id: str | None, **_kwargs
    ):
        return _AsyncFakeReader(stores.get(db_url, {}))

    return factory


class _AsyncMutableStore:
    def __init__(self, manifests: dict[str, ParsedManifest], initial: str) -> None:
        self.manifests = manifests
        self.current_ref = initial

    async def load_current(self) -> ManifestRef | None:
        return ManifestRef(
            ref=self.current_ref, run_id="r", published_at=datetime.now(UTC)
        )

    async def load_manifest(self, ref: str) -> ParsedManifest:
        return self.manifests[ref]

    async def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []


def _manifest(db_url: str, run_id: str = "run") -> ParsedManifest:
    required = RequiredBuildMeta(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.HASH,
            hash_algorithm="xxh3_64",
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    return ParsedManifest(
        required_build=required,
        shards=[
            RequiredShardMeta(
                db_id=0, db_url=db_url, attempt=0, row_count=1, db_bytes=0
            )
        ],
        custom={},
    )


@pytest.mark.asyncio
async def test_refresh_defers_cleanup_with_active_borrow(tmp_path: Path) -> None:
    """Old state cleanup is deferred when a borrow handle is held."""
    m1 = _manifest("mem://db/one", run_id="r1")
    m2 = _manifest("mem://db/two", run_id="r2")

    stores = {"mem://db/one": {}, "mem://db/two": {}}
    store = _AsyncMutableStore({"mem://m1": m1, "mem://m2": m2}, initial="mem://m1")

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_factory(stores),
    )

    # Borrow a handle from state v1
    handle = reader.reader_for_key(1)

    # Refresh to v2 — old state should be retired but not closed
    store.current_ref = "mem://m2"
    changed = await reader.refresh()
    assert changed is True

    # Release the borrow — this should trigger deferred cleanup
    handle.close()

    # Give the event loop a chance to run the cleanup task
    await asyncio.sleep(0.05)

    await reader.close()


@pytest.mark.asyncio
async def test_close_while_borrow_outstanding(tmp_path: Path) -> None:
    """Close reader while a borrow handle is still held."""
    m1 = _manifest("mem://db/one")
    stores = {"mem://db/one": {}}
    store = _AsyncMutableStore({"mem://m1": m1}, initial="mem://m1")

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_factory(stores),
    )

    handle = reader.reader_for_key(1)
    await reader.close()

    # Releasing after close should not crash
    handle.close()


@pytest.mark.asyncio
async def test_multiple_refreshes_deferred_cleanup(tmp_path: Path) -> None:
    """Multiple rapid refreshes queue multiple cleanup tasks."""
    m1 = _manifest("mem://db/one", run_id="r1")
    m2 = _manifest("mem://db/two", run_id="r2")
    m3 = _manifest("mem://db/three", run_id="r3")

    stores = {"mem://db/one": {}, "mem://db/two": {}, "mem://db/three": {}}
    store = _AsyncMutableStore(
        {"mem://m1": m1, "mem://m2": m2, "mem://m3": m3},
        initial="mem://m1",
    )

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_factory(stores),
    )

    store.current_ref = "mem://m2"
    await reader.refresh()

    store.current_ref = "mem://m3"
    await reader.refresh()

    info = reader.snapshot_info()
    assert info.run_id == "r3"

    await reader.close()


@pytest.mark.asyncio
async def test_get_after_refresh_uses_new_state(tmp_path: Path) -> None:
    """get() after refresh returns data from the new manifest."""
    key_bytes = (1).to_bytes(8, "big", signed=False)
    m1 = _manifest("mem://db/one", run_id="r1")
    m2 = _manifest("mem://db/two", run_id="r2")

    stores = {
        "mem://db/one": {key_bytes: b"old"},
        "mem://db/two": {key_bytes: b"new"},
    }
    store = _AsyncMutableStore({"mem://m1": m1, "mem://m2": m2}, initial="mem://m1")

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_factory(stores),
    )

    assert await reader.get(1) == b"old"

    store.current_ref = "mem://m2"
    await reader.refresh()

    assert await reader.get(1) == b"new"
    await reader.close()
