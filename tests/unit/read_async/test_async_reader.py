"""Unit tests for the async reader capability bucket."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from shardyfusion.errors import ReaderStateError
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.metrics import MetricEvent
from shardyfusion.reader import AsyncShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.testing import ListMetricsCollector

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _AsyncFakeReader:
    store: dict[bytes, bytes]
    closed: bool = False

    async def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    async def close(self) -> None:
        self.closed = True


def _async_fake_reader_factory(stores: dict[str, dict[bytes, bytes]]):
    """Return an AsyncShardReaderFactory that routes db_url → in-memory store."""

    async def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
        _ = (local_dir, checkpoint_id)
        return _AsyncFakeReader(stores[db_url])

    return factory


class _MutableManifestStore:
    """Sync manifest store whose current_ref can be swapped for refresh tests."""

    def __init__(self, manifests: dict[str, ParsedManifest], initial_ref: str) -> None:
        self.manifests = manifests
        self.current_ref = initial_ref

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        raise NotImplementedError("publish not used in reader tests")

    def load_current(self) -> ManifestRef | None:
        return ManifestRef(
            ref=self.current_ref,
            run_id="run",
            published_at=datetime.now(UTC),
        )

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self.manifests[ref]

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []

    def set_current(self, ref: str) -> None:
        pass


class _AsyncMutableManifestStore:
    """Native async manifest store for tests (no wrapping needed)."""

    def __init__(self, manifests: dict[str, ParsedManifest], initial_ref: str) -> None:
        self.manifests = manifests
        self.current_ref = initial_ref

    async def load_current(self) -> ManifestRef | None:
        return ManifestRef(
            ref=self.current_ref,
            run_id="run",
            published_at=datetime.now(UTC),
        )

    async def load_manifest(self, ref: str) -> ParsedManifest:
        return self.manifests[ref]


class _NullAsyncManifestStore:
    async def load_current(self) -> ManifestRef | None:
        return None

    async def load_manifest(self, ref: str) -> ParsedManifest:
        raise AssertionError("load_manifest should not be called")


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _required_build() -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )


def _manifest(db_url: str) -> ParsedManifest:
    return ParsedManifest(
        required_build=_required_build(),
        shards=[
            RequiredShardMeta(
                db_id=0,
                db_url=db_url,
                attempt=0,
                row_count=1,
                min_key=None,
                max_key=None,
                checkpoint_id=None,
                writer_info={},
            )
        ],
        custom={},
    )


def _manifest_2shard(db_url_0: str, db_url_1: str) -> ParsedManifest:
    """Two-shard hash manifest: key=1 -> shard 0, key=6 -> shard 1 (via xxh3)."""
    required = RequiredBuildMeta(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=2,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    return ParsedManifest(
        required_build=required,
        shards=[
            RequiredShardMeta(
                db_id=0,
                db_url=db_url_0,
                attempt=0,
                row_count=1,
                min_key=None,
                max_key=None,
                checkpoint_id=None,
                writer_info={},
            ),
            RequiredShardMeta(
                db_id=1,
                db_url=db_url_1,
                attempt=0,
                row_count=1,
                min_key=None,
                max_key=None,
                checkpoint_id=None,
                writer_info={},
            ),
        ],
        custom={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_and_get(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
    )
    assert await reader.get(1) == b"val"
    assert await reader.get(999) is None
    await reader.close()


@pytest.mark.asyncio
async def test_multi_get(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/zero": {(1).to_bytes(8, "big", signed=False): b"a"},
        "mem://db/one": {(6).to_bytes(8, "big", signed=False): b"b"},
    }

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
    ) as reader:
        got = await reader.multi_get([1, 6])
        assert got[1] == b"a"
        assert got[6] == b"b"


@pytest.mark.asyncio
async def test_multi_get_with_concurrency_limit(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/zero": {(1).to_bytes(8, "big", signed=False): b"a"},
        "mem://db/one": {(6).to_bytes(8, "big", signed=False): b"b"},
    }

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
        max_concurrency=1,
    ) as reader:
        got = await reader.multi_get([1, 6])
        assert got[1] == b"a"
        assert got[6] == b"b"


@pytest.mark.asyncio
async def test_refresh(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest("mem://db/one"),
        "mem://manifest/two": _manifest("mem://db/two"),
    }
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"one"},
        "mem://db/two": {(1).to_bytes(8, "big", signed=False): b"two"},
    }

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
    ) as reader:
        assert await reader.get(1) == b"one"
        store.current_ref = "mem://manifest/two"

        changed = await reader.refresh()
        assert changed is True
        assert await reader.get(1) == b"two"

        unchanged = await reader.refresh()
        assert unchanged is False


@pytest.mark.asyncio
async def test_close_prevents_get(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
    )
    await reader.close()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        await reader.get(1)


@pytest.mark.asyncio
async def test_close_prevents_refresh(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
    )
    await reader.close()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        await reader.refresh()


@pytest.mark.asyncio
async def test_context_manager(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
    )

    async with reader as ctx:
        assert ctx is reader

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        await reader.get(1)


@pytest.mark.asyncio
async def test_missing_current_raises() -> None:
    with pytest.raises(ReaderStateError, match="CURRENT pointer not found"):
        await AsyncShardedReader.open(
            s3_prefix="s3://bucket/prefix",
            local_root="/tmp/unused",
            manifest_store=_NullAsyncManifestStore(),
            reader_factory=_async_fake_reader_factory({}),
        )


# ---------------------------------------------------------------------------
# Borrow / handle tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reader_for_key(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/zero": {(1).to_bytes(8, "big", signed=False): b"a"},
        "mem://db/one": {(6).to_bytes(8, "big", signed=False): b"b"},
    }

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
    ) as reader:
        handle = reader.reader_for_key(1)
        key_bytes = (1).to_bytes(8, "big", signed=False)
        assert await handle.get(key_bytes) == b"a"
        handle.close()


@pytest.mark.asyncio
async def test_readers_for_keys(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/zero": {(1).to_bytes(8, "big", signed=False): b"a"},
        "mem://db/one": {(6).to_bytes(8, "big", signed=False): b"b"},
    }

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
    ) as reader:
        handles = reader.readers_for_keys([1, 6])
        assert await handles[1].get((1).to_bytes(8, "big", signed=False)) == b"a"
        assert await handles[6].get((6).to_bytes(8, "big", signed=False)) == b"b"
        for h in handles.values():
            h.close()


@pytest.mark.asyncio
async def test_borrow_count(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
    ) as reader:
        assert reader._state.borrow_count == 0
        h1 = reader.reader_for_key(1)
        assert reader._state.borrow_count == 1
        h2 = reader.reader_for_key(1)
        assert reader._state.borrow_count == 2
        h1.close()
        assert reader._state.borrow_count == 1
        h2.close()
        assert reader._state.borrow_count == 0


@pytest.mark.asyncio
async def test_handle_double_close_is_noop(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
    ) as reader:
        handle = reader.reader_for_key(1)
        handle.close()
        handle.close()
        assert reader._state.borrow_count == 0


@pytest.mark.asyncio
async def test_handle_context_manager(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
    ) as reader:
        async with reader.reader_for_key(1) as handle:
            assert await handle.get((1).to_bytes(8, "big", signed=False)) == b"val"
        assert reader._state.borrow_count == 0


@pytest.mark.asyncio
async def test_reader_for_key_raises_when_closed(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
    )
    await reader.close()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.reader_for_key(1)


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_key_encoding(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
    ) as reader:
        assert reader.key_encoding == KeyEncoding.U64BE


@pytest.mark.asyncio
async def test_snapshot_info(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
    ) as reader:
        info = reader.snapshot_info()
        assert info.run_id == "run"
        assert info.num_dbs == 1
        assert info.sharding == ShardingStrategy.HASH
        assert info.manifest_ref == "mem://manifest/one"


@pytest.mark.asyncio
async def test_shard_details(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
    ) as reader:
        details = reader.shard_details()
        assert len(details) == 1
        assert details[0].db_id == 0
        assert details[0].db_url == "mem://db/one"


@pytest.mark.asyncio
async def test_route_key(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(
            {"mem://db/zero": {}, "mem://db/one": {}}
        ),
    ) as reader:
        assert reader.route_key(1) == 0
        assert reader.route_key(6) == 1


@pytest.mark.asyncio
async def test_shard_for_key(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(
            {"mem://db/zero": {}, "mem://db/one": {}}
        ),
    ) as reader:
        meta_0 = reader.shard_for_key(1)
        assert meta_0.db_id == 0
        meta_1 = reader.shard_for_key(6)
        assert meta_1.db_id == 1


@pytest.mark.asyncio
async def test_shards_for_keys(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(
            {"mem://db/zero": {}, "mem://db/one": {}}
        ),
    ) as reader:
        mapping = reader.shards_for_keys([1, 6])
        assert mapping[1].db_id == 0
        assert mapping[6].db_id == 1


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_lifecycle(tmp_path) -> None:
    mc = ListMetricsCollector()
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
        metrics_collector=mc,
    ) as reader:
        result = await reader.get(1)
        assert result == b"val"

    event_names = [e[0] for e in mc.events]
    assert MetricEvent.READER_INITIALIZED in event_names
    assert MetricEvent.READER_GET in event_names
    assert MetricEvent.READER_CLOSED in event_names

    get_payload = next(p for e, p in mc.events if e is MetricEvent.READER_GET)
    assert "duration_ms" in get_payload
    assert get_payload["found"] is True


@pytest.mark.asyncio
async def test_metrics_get_not_found(tmp_path) -> None:
    mc = ListMetricsCollector()
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
        metrics_collector=mc,
    ) as reader:
        result = await reader.get(1)
        assert result is None

    get_payload = next(p for e, p in mc.events if e is MetricEvent.READER_GET)
    assert get_payload["found"] is False


@pytest.mark.asyncio
async def test_metrics_multi_get(tmp_path) -> None:
    mc = ListMetricsCollector()
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
        metrics_collector=mc,
    ) as reader:
        await reader.multi_get([1, 2])

    multi_payload = next(p for e, p in mc.events if e is MetricEvent.READER_MULTI_GET)
    assert multi_payload["num_keys"] == 2
    assert "duration_ms" in multi_payload


@pytest.mark.asyncio
async def test_metrics_refresh(tmp_path) -> None:
    mc = ListMetricsCollector()
    manifests = {
        "mem://manifest/one": _manifest("mem://db/one"),
        "mem://manifest/two": _manifest("mem://db/two"),
    }
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {},
        "mem://db/two": {},
    }

    async with await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
        metrics_collector=mc,
    ) as reader:
        store.current_ref = "mem://manifest/two"
        await reader.refresh()

    refresh_events = [p for e, p in mc.events if e is MetricEvent.READER_REFRESHED]
    assert any(p["changed"] is True for p in refresh_events)


# ---------------------------------------------------------------------------
# Close behavior tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_double_close_is_noop(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory({"mem://db/one": {}}),
    )
    await reader.close()
    await reader.close()  # should not raise


@pytest.mark.asyncio
async def test_close_closes_shard_readers(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {"mem://db/one": {}}

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
    )
    shard_reader = reader._state.readers[0]
    assert isinstance(shard_reader, _AsyncFakeReader)
    assert not shard_reader.closed

    await reader.close()
    assert shard_reader.closed


# ---------------------------------------------------------------------------
# Concurrent borrow-count tests
# ---------------------------------------------------------------------------


class _SlowAsyncFakeReader:
    def __init__(
        self, store: dict[bytes, bytes], delay: timedelta = timedelta(seconds=0.1)
    ):
        self.store = store
        self.delay = delay
        self.closed = False

    async def get(self, key: bytes) -> bytes | None:
        await asyncio.sleep(self.delay.total_seconds())
        return self.store.get(key)

    async def close(self) -> None:
        self.closed = True


def _slow_async_fake_reader_factory(
    stores: dict[str, dict[bytes, bytes]], delay: timedelta = timedelta(seconds=0.1)
):
    async def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
        _ = (local_dir, checkpoint_id)
        return _SlowAsyncFakeReader(stores[db_url], delay=delay)

    return factory


@pytest.mark.asyncio
async def test_get_increments_borrow_during_read(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_slow_async_fake_reader_factory(
            stores, delay=timedelta(seconds=0.1)
        ),
    )

    assert reader._state.borrow_count == 0

    # Start get() but don't await it yet
    task = asyncio.create_task(reader.get(1))

    # Yield control so the task enters get() and increments borrow_count
    await asyncio.sleep(0.01)
    assert reader._state.borrow_count > 0

    result = await task
    assert result == b"val"
    assert reader._state.borrow_count == 0

    await reader.close()


@pytest.mark.asyncio
async def test_refresh_waits_for_inflight_get(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest("mem://db/one"),
        "mem://manifest/two": _manifest("mem://db/two"),
    }
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"one"},
        "mem://db/two": {(1).to_bytes(8, "big", signed=False): b"two"},
    }

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_slow_async_fake_reader_factory(
            stores, delay=timedelta(seconds=0.15)
        ),
    )

    old_state = reader._state

    # Start a slow get
    get_task = asyncio.create_task(reader.get(1))
    await asyncio.sleep(0.01)

    # Now refresh while the get is in-flight
    store.current_ref = "mem://manifest/two"
    changed = await reader.refresh()
    assert changed is True

    # Old state is retired but NOT closed yet because the get holds a borrow
    assert old_state.retired is True
    assert not all(r.closed for r in old_state.readers.values()), (
        "Old readers should not be closed while get is in-flight"
    )

    # Complete the get — this releases the borrow and schedules deferred cleanup
    result = await get_task
    assert result == b"one"

    # Let the deferred cleanup task run
    await asyncio.sleep(0.01)
    assert all(r.closed for r in old_state.readers.values())

    await reader.close()


@pytest.mark.asyncio
async def test_close_waits_for_inflight_get(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    store = _AsyncMutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_slow_async_fake_reader_factory(
            stores, delay=timedelta(seconds=0.15)
        ),
    )

    state = reader._state

    # Start a slow get
    get_task = asyncio.create_task(reader.get(1))
    await asyncio.sleep(0.01)

    # Close while the get is in-flight
    await reader.close()

    # State is retired but NOT closed because the get holds a borrow
    assert state.retired is True
    assert not all(r.closed for r in state.readers.values()), (
        "Readers should not be closed while get is in-flight"
    )

    # Complete the get — this releases the borrow and schedules deferred cleanup
    result = await get_task
    assert result == b"val"

    # Let the deferred cleanup task run
    await asyncio.sleep(0.01)
    assert all(r.closed for r in state.readers.values())


# ---------------------------------------------------------------------------
# Null shard reader tests (db_url=None)
# ---------------------------------------------------------------------------


def _manifest_with_empty_shard() -> ParsedManifest:
    """Manifest with shard 0 = real data, shard 1 = empty (omitted from manifest).

    The SnapshotRouter synthesizes a metadata-only entry for shard 1.
    """
    build = RequiredBuildMeta(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=2,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    return ParsedManifest(
        required_build=build,
        shards=[
            RequiredShardMeta(
                db_id=0,
                db_url="mem://db/shard0",
                attempt=0,
                row_count=5,
                min_key=0,
                max_key=4,
                checkpoint_id="ckpt-0",
            ),
        ],
        custom={},
    )


@pytest.mark.asyncio
async def test_async_reader_null_shard_get(tmp_path: Path) -> None:
    """Keys routed to an empty shard return None via null async reader.

    With xxh3 and num_dbs=2: key=1 -> shard 0 (real), key=0 -> shard 1 (empty).
    """
    manifest = _manifest_with_empty_shard()
    stores = {"mem://db/shard0": {(1).to_bytes(8, "big"): b"val1"}}
    store = _AsyncMutableManifestStore(
        {"mem://manifest/v1": manifest}, "mem://manifest/v1"
    )
    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
    )
    async with reader:
        assert await reader.get(1) == b"val1"
        assert await reader.get(0) is None


@pytest.mark.asyncio
async def test_async_reader_null_shard_multi_get(tmp_path: Path) -> None:
    """multi_get across real and empty shards returns mixed results.

    With xxh3 and num_dbs=2: key=1 -> shard 0 (real), key=0 -> shard 1 (empty).
    """
    manifest = _manifest_with_empty_shard()
    stores = {"mem://db/shard0": {(1).to_bytes(8, "big"): b"val1"}}
    store = _AsyncMutableManifestStore(
        {"mem://manifest/v1": manifest}, "mem://manifest/v1"
    )
    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=_async_fake_reader_factory(stores),
    )
    async with reader:
        results = await reader.multi_get([1, 0])
        assert results[1] == b"val1"
        assert results[0] is None


@pytest.mark.asyncio
async def test_async_reader_factory_not_called_for_null_shard(tmp_path: Path) -> None:
    """Async reader factory is never invoked for shards with db_url=None."""
    manifest = _manifest_with_empty_shard()
    factory_calls: list[str] = []

    async def tracking_factory(
        *, db_url: str, local_dir: Path, checkpoint_id: str | None
    ):
        factory_calls.append(db_url)
        return _AsyncFakeReader({})

    store = _AsyncMutableManifestStore(
        {"mem://manifest/v1": manifest}, "mem://manifest/v1"
    )
    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=store,
        reader_factory=tracking_factory,
    )
    assert factory_calls == ["mem://db/shard0"]
    await reader.close()
