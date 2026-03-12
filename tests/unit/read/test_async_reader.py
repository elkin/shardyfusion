from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from shardyfusion.async_manifest_store import _SyncManifestStoreAdapter
from shardyfusion.errors import ReaderStateError
from shardyfusion.manifest import (
    CurrentPointer,
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

    def load_current(self) -> CurrentPointer | None:
        return CurrentPointer(
            manifest_ref=self.current_ref,
            manifest_content_type="application/json",
            run_id="run",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self.manifests[ref]


class _AsyncMutableManifestStore:
    """Native async manifest store for tests (no wrapping needed)."""

    def __init__(self, manifests: dict[str, ParsedManifest], initial_ref: str) -> None:
        self.manifests = manifests
        self.current_ref = initial_ref

    async def load_current(self) -> CurrentPointer | None:
        return CurrentPointer(
            manifest_ref=self.current_ref,
            manifest_content_type="application/json",
            run_id="run",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    async def load_manifest(self, ref: str) -> ParsedManifest:
        return self.manifests[ref]


class _NullAsyncManifestStore:
    async def load_current(self) -> CurrentPointer | None:
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
        tmp_prefix="_tmp",
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
    required = RequiredBuildMeta(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=2,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.RANGE),
        db_path_template="db={db_id:05d}",
        tmp_prefix="_tmp",
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
                max_key=5,
                checkpoint_id=None,
                writer_info={},
            ),
            RequiredShardMeta(
                db_id=1,
                db_url=db_url_1,
                attempt=0,
                row_count=1,
                min_key=6,
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
async def test_open_with_sync_manifest_store(tmp_path) -> None:
    """Sync ManifestStore should be auto-wrapped in _SyncManifestStoreAdapter."""
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    sync_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    reader = await AsyncShardedReader.open(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=sync_store,
        reader_factory=_async_fake_reader_factory(stores),
    )
    assert isinstance(reader._manifest_store, _SyncManifestStoreAdapter)
    assert await reader.get(1) == b"val"
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
