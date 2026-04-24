from __future__ import annotations

import sys
import threading
import time
import types
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from shardyfusion.errors import (
    ConfigValidationError,
    DbAdapterError,
    ManifestParseError,
    ReaderStateError,
)
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.metrics import MetricEvent
from shardyfusion.reader import (
    ConcurrentShardedReader,
    ShardedReader,
    SlateDbReaderFactory,
)
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.testing import ListMetricsCollector


class _MutableManifestStore:
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


@dataclass
class _FakeReader:
    store: dict[bytes, bytes]

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def close(self) -> None:
        return None


def _fake_reader_factory(stores: dict[str, dict[bytes, bytes]]):
    """Return a ShardReaderFactory that routes db_url → in-memory store."""

    def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
        _ = (local_dir, checkpoint_id)
        return _FakeReader(stores[db_url])

    return factory


def _required_build() -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run",
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


def _vector_only_manifest(db_url: str) -> ParsedManifest:
    manifest = _manifest(db_url)
    return ParsedManifest(
        required_build=manifest.required_build,
        shards=manifest.shards,
        custom={
            "vector": {
                "dim": 4,
                "metric": "cosine",
                "backend": "lancedb",
                "kv_backend": "slatedb",
            }
        },
    )


def test_refresh_swaps_manifest_ref_and_readers(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest("mem://db/one"),
        "mem://manifest/two": _manifest("mem://db/two"),
    }
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"one"},
        "mem://db/two": {(1).to_bytes(8, "big", signed=False): b"two"},
    }

    with ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
    ) as reader:
        assert reader.get(1) == b"one"
        manifest_store.current_ref = "mem://manifest/two"

        changed = reader.refresh()
        assert changed is True
        assert reader.get(1) == b"two"

        unchanged = reader.refresh()
        assert unchanged is False


def test_slate_db_reader_factory_uses_official_slatedbreader_signature(
    monkeypatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel = _FakeReader({})

    def fake_reader_ctor(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    fake_module = types.ModuleType("slatedb")
    fake_module.SlateDBReader = fake_reader_ctor
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    factory = SlateDbReaderFactory(env_file="slatedb.env")
    reader = factory(
        db_url="s3://bucket/db",
        local_dir=Path("/tmp/local"),
        checkpoint_id="ckpt-1",
    )

    assert reader is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("/tmp/local",)
    assert kwargs["url"] == "s3://bucket/db"
    assert kwargs["checkpoint_id"] == "ckpt-1"
    assert kwargs["env_file"] == "slatedb.env"


class _NullManifestStore:
    """Always returns None for CURRENT (simulates missing pointer)."""

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
        return None

    def load_manifest(self, ref: str) -> ParsedManifest:
        raise AssertionError("load_manifest should not be called")

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []

    def set_current(self, ref: str) -> None:
        pass


def test_load_initial_state_raises_reader_state_error_when_no_current() -> None:
    with pytest.raises(ReaderStateError, match="CURRENT pointer not found"):
        ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root="/tmp/unused",
            manifest_store=_NullManifestStore(),
        )


def test_closed_reader_get_raises_reader_state_error(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _FakeReader({}),
    )
    reader.close()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.get(1)


def test_closed_reader_refresh_raises_reader_state_error(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _FakeReader({}),
    )
    reader.close()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.refresh()


def test_context_manager_returns_self_and_calls_close(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _FakeReader({}),
    )

    with reader as ctx:
        assert ctx is reader

    # After exiting the context, the reader should be closed.
    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.get(1)


def _manifest_2shard(db_url_0: str, db_url_1: str) -> ParsedManifest:
    """Two-shard hash manifest: key=1 → shard 0, key=6 → shard 1 (via xxh3)."""
    required = RequiredBuildMeta(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=2,
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


def test_multi_get_shard_failure_raises_slate_db_api_error(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    class _BrokenReader:
        def get(self, key: bytes) -> bytes | None:
            raise OSError("disk error")

        def close(self) -> None:
            pass

    with ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _BrokenReader(),
        max_workers=2,
    ) as reader:
        # key=1 routes to shard 0, key=6 routes to shard 1 via xxh3 → both shards in executor
        with pytest.raises(DbAdapterError, match="db_id="):
            reader.multi_get([1, 6])


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def test_metrics_emitted_on_reader_lifecycle(tmp_path) -> None:
    mc = ListMetricsCollector()
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    with ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
        metrics_collector=mc,
    ) as reader:
        result = reader.get(1)
        assert result == b"val"

    event_names = [e[0] for e in mc.events]
    assert MetricEvent.READER_INITIALIZED in event_names
    assert MetricEvent.READER_GET in event_names
    assert MetricEvent.READER_CLOSED in event_names

    # Check READER_GET payload
    get_payload = next(p for e, p in mc.events if e is MetricEvent.READER_GET)
    assert "duration_ms" in get_payload
    assert get_payload["found"] is True


def test_metrics_reader_get_not_found(tmp_path) -> None:
    mc = ListMetricsCollector()
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    stores: dict[str, dict[bytes, bytes]] = {"mem://db/one": {}}

    with ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
        metrics_collector=mc,
    ) as reader:
        result = reader.get(1)
        assert result is None

    get_payload = next(p for e, p in mc.events if e is MetricEvent.READER_GET)
    assert get_payload["found"] is False


# ---------------------------------------------------------------------------
# ShardedReader tests (non-thread-safe variant)
# ---------------------------------------------------------------------------


def test_sharded_reader_get_basic(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    with ShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
    ) as reader:
        assert reader.get(1) == b"val"
        assert reader.get(999) is None


def test_sharded_reader_multi_get_sequential(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/zero": {(1).to_bytes(8, "big", signed=False): b"a"},
        "mem://db/one": {(6).to_bytes(8, "big", signed=False): b"b"},
    }

    with ShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
    ) as reader:
        got = reader.multi_get([1, 6])
        assert got[1] == b"a"
        assert got[6] == b"b"


def test_sharded_reader_multi_get_parallel(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/zero": {(1).to_bytes(8, "big", signed=False): b"a"},
        "mem://db/one": {(6).to_bytes(8, "big", signed=False): b"b"},
    }

    with ShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
        max_workers=2,
    ) as reader:
        got = reader.multi_get([1, 6])
        assert got[1] == b"a"
        assert got[6] == b"b"


def test_sharded_reader_refresh(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest("mem://db/one"),
        "mem://manifest/two": _manifest("mem://db/two"),
    }
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"one"},
        "mem://db/two": {(1).to_bytes(8, "big", signed=False): b"two"},
    }

    with ShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
    ) as reader:
        assert reader.get(1) == b"one"
        manifest_store.current_ref = "mem://manifest/two"

        changed = reader.refresh()
        assert changed is True
        assert reader.get(1) == b"two"


def test_sharded_reader_refresh_unchanged(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    with ShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
    ) as reader:
        assert reader.refresh() is False


def test_sharded_reader_close_prevents_get(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    reader = ShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _FakeReader({}),
    )
    reader.close()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.get(1)


def test_sharded_reader_close_prevents_refresh(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    reader = ShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _FakeReader({}),
    )
    reader.close()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.refresh()


def test_sharded_reader_context_manager(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    reader = ShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _FakeReader({}),
    )

    with reader as ctx:
        assert ctx is reader

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.get(1)


def test_sharded_reader_missing_current_raises() -> None:
    with pytest.raises(ReaderStateError, match="CURRENT pointer not found"):
        ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root="/tmp/unused",
            manifest_store=_NullManifestStore(),
        )


def test_sharded_reader_rejects_vector_only_manifest(tmp_path) -> None:
    manifests = {"mem://manifest/one": _vector_only_manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"one"},
    }

    with pytest.raises(
        ConfigValidationError, match="unsupported or incomplete vector metadata"
    ):
        ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_store,
            reader_factory=_fake_reader_factory(stores),
        )


def test_concurrent_reader_rejects_vector_only_manifest(tmp_path) -> None:
    manifests = {"mem://manifest/one": _vector_only_manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"one"},
    }

    with pytest.raises(
        ConfigValidationError, match="unsupported or incomplete vector metadata"
    ):
        ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_store,
            reader_factory=_fake_reader_factory(stores),
        )


def test_sharded_reader_metrics_lifecycle(tmp_path) -> None:
    mc = ListMetricsCollector()
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    with ShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
        metrics_collector=mc,
    ) as reader:
        reader.get(1)

    event_names = [e[0] for e in mc.events]
    assert MetricEvent.READER_INITIALIZED in event_names
    assert MetricEvent.READER_GET in event_names
    assert MetricEvent.READER_CLOSED in event_names


def test_sharded_reader_multi_get_shard_failure(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    class _BrokenReader:
        def get(self, key: bytes) -> bytes | None:
            raise OSError("disk error")

        def close(self) -> None:
            pass

    with ShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _BrokenReader(),
        max_workers=2,
    ) as reader:
        with pytest.raises(DbAdapterError, match="db_id="):
            reader.multi_get([1, 6])


# ---------------------------------------------------------------------------
# Executor reuse test
# ---------------------------------------------------------------------------


def test_thread_safe_reader_executor_created_at_init(tmp_path) -> None:
    """Verify ConcurrentShardedReader creates _executor at init, not per multi_get."""
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    with ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
        max_workers=2,
    ) as reader:
        assert reader._executor is not None
        executor_id = id(reader._executor)
        reader.multi_get([1])
        assert id(reader._executor) == executor_id


# ---------------------------------------------------------------------------
# shard_for_key / shards_for_keys / reader_for_key / readers_for_keys tests
# ---------------------------------------------------------------------------


def _2shard_setup(tmp_path):
    """Shared helper: 2-shard hash reader (key=1 → shard 0, key=6 → shard 1 via xxh3)."""
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    manifest_reader = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/zero": {(1).to_bytes(8, "big", signed=False): b"a"},
        "mem://db/one": {(6).to_bytes(8, "big", signed=False): b"b"},
    }
    return manifest_reader, stores


class TestShardForKey:
    """Tests for shard_for_key / shards_for_keys on both reader variants."""

    def test_simple_reader_shard_for_key(self, tmp_path) -> None:
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            meta_0 = reader.shard_for_key(1)
            assert meta_0.db_id == 0
            assert meta_0.db_url == "mem://db/zero"

            meta_1 = reader.shard_for_key(6)
            assert meta_1.db_id == 1
            assert meta_1.db_url == "mem://db/one"

    def test_concurrent_reader_shard_for_key(self, tmp_path) -> None:
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            meta_0 = reader.shard_for_key(1)
            assert meta_0.db_id == 0

            meta_1 = reader.shard_for_key(6)
            assert meta_1.db_id == 1

    def test_simple_reader_shards_for_keys(self, tmp_path) -> None:
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            mapping = reader.shards_for_keys([1, 6])
            assert mapping[1].db_id == 0
            assert mapping[6].db_id == 1

    def test_concurrent_reader_shards_for_keys(self, tmp_path) -> None:
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            mapping = reader.shards_for_keys([1, 6])
            assert mapping[1].db_id == 0
            assert mapping[6].db_id == 1


class TestReaderForKey:
    """Tests for reader_for_key / readers_for_keys on both reader variants."""

    def test_simple_reader_for_key_reads_value(self, tmp_path) -> None:
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            handle = reader.reader_for_key(1)
            key_bytes = (1).to_bytes(8, "big", signed=False)
            assert handle.get(key_bytes) == b"a"
            handle.close()

    def test_simple_readers_for_keys_reads_values(self, tmp_path) -> None:
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            handles = reader.readers_for_keys([1, 6])
            assert handles[1].get((1).to_bytes(8, "big", signed=False)) == b"a"
            assert handles[6].get((6).to_bytes(8, "big", signed=False)) == b"b"
            for h in handles.values():
                h.close()

    def test_close_does_not_close_underlying(self, tmp_path) -> None:
        """Closing borrowed handle should not close the parent reader's shard."""
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            handle = reader.reader_for_key(1)
            handle.close()
            # Parent reader should still work
            assert reader.get(1) == b"a"

    def test_reader_for_key_raises_when_closed(self, tmp_path) -> None:
        manifest_reader, stores = _2shard_setup(tmp_path)
        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        )
        reader.close()
        with pytest.raises(ReaderStateError, match="Reader is closed"):
            reader.reader_for_key(1)

    def test_readers_for_keys_raises_when_closed(self, tmp_path) -> None:
        manifest_reader, stores = _2shard_setup(tmp_path)
        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        )
        reader.close()
        with pytest.raises(ReaderStateError, match="Reader is closed"):
            reader.readers_for_keys([1])

    def test_concurrent_reader_for_key_reads_value(self, tmp_path) -> None:
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            handle = reader.reader_for_key(6)
            assert handle.get((6).to_bytes(8, "big", signed=False)) == b"b"
            handle.close()

    def test_concurrent_reader_for_key_refcount(self, tmp_path) -> None:
        """Verify refcount goes up on borrow and back down on close."""
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            assert reader._state.refcount == 0
            h1 = reader.reader_for_key(1)
            assert reader._state.refcount == 1
            h2 = reader.reader_for_key(6)
            assert reader._state.refcount == 2
            h1.close()
            assert reader._state.refcount == 1
            h2.close()
            assert reader._state.refcount == 0

    def test_concurrent_readers_for_keys_dedup(self, tmp_path) -> None:
        """Duplicate keys should be deduped; refcount = number of unique keys."""
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            handles = reader.readers_for_keys([1, 1, 6])
            # Only 2 unique keys
            assert len(handles) == 2
            assert reader._state.refcount == 2
            for h in handles.values():
                h.close()
            assert reader._state.refcount == 0

    def test_concurrent_readers_for_keys_empty(self, tmp_path) -> None:
        """Empty key list returns empty dict without touching refcount."""
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            handles = reader.readers_for_keys([])
            assert handles == {}
            assert reader._state.refcount == 0

    def test_concurrent_reader_for_key_raises_when_closed(self, tmp_path) -> None:
        manifest_reader, stores = _2shard_setup(tmp_path)
        reader = ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        )
        reader.close()
        with pytest.raises(ReaderStateError, match="Reader is closed"):
            reader.reader_for_key(1)

    def test_shard_reader_handle_double_close_is_noop(self, tmp_path) -> None:
        """Calling close() twice on ShardReaderHandle should not double-release."""
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            handle = reader.reader_for_key(1)
            assert reader._state.refcount == 1
            handle.close()
            assert reader._state.refcount == 0
            handle.close()  # no-op
            assert reader._state.refcount == 0

    def test_shard_reader_handle_context_manager(self, tmp_path) -> None:
        """Verify ``with`` releases refcount on exit."""
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            with reader.reader_for_key(1) as handle:
                assert reader._state.refcount == 1
                assert handle.get((1).to_bytes(8, "big", signed=False)) == b"a"
            assert reader._state.refcount == 0

    def test_shard_reader_handle_multi_get(self, tmp_path) -> None:
        """Verify batch reads on a single shard handle."""
        manifest_reader, stores = _2shard_setup(tmp_path)
        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            with reader.reader_for_key(1) as handle:
                key_bytes = (1).to_bytes(8, "big", signed=False)
                missing_bytes = (999).to_bytes(8, "big", signed=False)
                result = handle.multi_get([key_bytes, missing_bytes])
                assert result[key_bytes] == b"a"
                assert result[missing_bytes] is None


class TestSimpleReaderBorrowSafety:
    """Tests for borrow safety on ShardedReader (deferred close)."""

    def test_simple_reader_refresh_defers_close_with_borrow(self, tmp_path) -> None:
        """Borrow a handle, refresh, verify handle still works, close handle."""
        manifests = {
            "mem://manifest/one": _manifest("mem://db/one"),
            "mem://manifest/two": _manifest("mem://db/two"),
        }
        manifest_reader = _MutableManifestStore(manifests, "mem://manifest/one")
        stores: dict[str, dict[bytes, bytes]] = {
            "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"one"},
            "mem://db/two": {(1).to_bytes(8, "big", signed=False): b"two"},
        }

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        )
        key_bytes = (1).to_bytes(8, "big", signed=False)

        # Borrow a handle from the old state
        handle = reader.reader_for_key(1)
        old_state = reader._state
        assert old_state.borrow_count == 1

        # Refresh — old state should be retired but NOT closed
        manifest_reader.current_ref = "mem://manifest/two"
        reader.refresh()
        assert old_state.retired is True
        # Handle should still work (old shard readers not yet closed)
        assert handle.get(key_bytes) == b"one"

        # New state should serve new data
        assert reader.get(1) == b"two"

        # Close handle — old state should now be cleaned up
        handle.close()
        assert old_state.borrow_count == 0

        reader.close()

    def test_simple_reader_close_defers_with_borrow(self, tmp_path) -> None:
        """Borrow a handle, close parent, verify handle still works."""
        manifests = {"mem://manifest/one": _manifest("mem://db/one")}
        manifest_reader = _MutableManifestStore(manifests, "mem://manifest/one")
        stores: dict[str, dict[bytes, bytes]] = {
            "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
        }

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=manifest_reader,
            reader_factory=_fake_reader_factory(stores),
        )
        key_bytes = (1).to_bytes(8, "big", signed=False)

        handle = reader.reader_for_key(1)
        state = reader._state
        assert state.borrow_count == 1

        # Close parent — state should be retired but not closed
        reader.close()
        assert state.retired is True

        # Handle should still work
        assert handle.get(key_bytes) == b"val"

        # Release handle — triggers deferred cleanup
        handle.close()
        assert state.borrow_count == 0


# ---------------------------------------------------------------------------
# Commit 1: Superseded refresh cleans up discarded state
# ---------------------------------------------------------------------------


def test_concurrent_reader_serialized_refresh_cleans_up(tmp_path) -> None:
    """Two threads call refresh(); _refresh_lock serializes them, old state is closed."""
    closed_urls: list[str] = []

    class _TrackingReader:
        def __init__(self, url: str) -> None:
            self.url = url

        def get(self, key: bytes) -> bytes | None:
            return None

        def close(self) -> None:
            closed_urls.append(self.url)

    manifests = {
        "mem://manifest/v1": _manifest("mem://db/v1"),
        "mem://manifest/v2": _manifest("mem://db/v2"),
    }
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/v1")

    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _TrackingReader(
            db_url
        ),
    )
    assert reader._state.manifest_ref == "mem://manifest/v1"

    manifest_store.current_ref = "mem://manifest/v2"

    # Slow down load_manifest so thread 2 queues behind _refresh_lock.
    entered_load = threading.Event()
    proceed = threading.Event()
    original_load = manifest_store.load_manifest
    slow_done = threading.Event()

    def slow_once_load_manifest(ref: str) -> ParsedManifest:
        if not slow_done.is_set():
            slow_done.set()
            entered_load.set()
            proceed.wait(timeout=5)
        return original_load(ref)

    manifest_store.load_manifest = slow_once_load_manifest  # type: ignore[assignment]

    results: list[bool | None] = [None, None]
    errors: list[Exception] = []

    def first_refresh() -> None:
        try:
            results[0] = reader.refresh()
        except Exception as e:
            errors.append(e)

    def second_refresh() -> None:
        # Wait until first thread is inside load_manifest (holding _refresh_lock)
        entered_load.wait(timeout=5)
        proceed.set()  # Unblock the slow load
        try:
            results[1] = reader.refresh()
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=first_refresh)
    t2 = threading.Thread(target=second_refresh)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not errors
    # First thread updates v1 → v2
    assert results[0] is True
    # Second thread sees v2 already active, short-circuits
    assert results[1] is False
    assert reader._state.manifest_ref == "mem://manifest/v2"

    # Old v1 state was properly closed
    v1_closed = [u for u in closed_urls if u == "mem://db/v1"]
    assert len(v1_closed) == 1

    reader.close()


# ---------------------------------------------------------------------------
# Commit 2: Metadata methods raise when closed
# ---------------------------------------------------------------------------


def test_concurrent_metadata_methods_raise_when_closed(tmp_path) -> None:
    """snapshot_info, shard_details, route_key, key_encoding raise on closed reader."""
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _FakeReader({}),
    )
    reader.close()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.snapshot_info()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.shard_details()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.route_key(1)

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        _ = reader.key_encoding


# ---------------------------------------------------------------------------
# Commit 3: Pool checkout timeout
# ---------------------------------------------------------------------------


def test_pool_checkout_timeout_raises(tmp_path) -> None:
    """Pool checkout raises PoolExhaustedError after timeout."""
    in_get = threading.Event()

    class _SlowReader:
        def get(self, key: bytes) -> bytes | None:
            in_get.set()
            time.sleep(2)
            return None

        def close(self) -> None:
            pass

    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")

    with ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _SlowReader(),
        thread_safety="pool",
        pool_checkout_timeout=timedelta(seconds=0.1),
        max_workers=1,
    ) as reader:
        errors: list[Exception] = []

        def hold_pool() -> None:
            try:
                reader.get(1)
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=hold_pool)
        t.start()
        in_get.wait(timeout=5)

        # The single pool reader is held; this checkout should time out.
        from shardyfusion.errors import PoolExhaustedError

        with pytest.raises(PoolExhaustedError, match="timed out"):
            reader.get(1)

        t.join(timeout=5)


# ---------------------------------------------------------------------------
# Commit 4: Borrowed handle survives refresh, refresh failure
# ---------------------------------------------------------------------------


def test_concurrent_reader_borrowed_handle_survives_refresh(tmp_path) -> None:
    """Borrow handle, refresh, verify handle reads old data, reader reads new data."""
    manifests = {
        "mem://manifest/one": _manifest("mem://db/one"),
        "mem://manifest/two": _manifest("mem://db/two"),
    }
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"one"},
        "mem://db/two": {(1).to_bytes(8, "big", signed=False): b"two"},
    }

    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
    )
    key_bytes = (1).to_bytes(8, "big", signed=False)

    # Borrow handle from old state
    handle = reader.reader_for_key(1)
    old_state = reader._state
    assert old_state.refcount == 1

    # Refresh to new manifest
    manifest_store.current_ref = "mem://manifest/two"
    reader.refresh()
    assert old_state.retired is True

    # Borrowed handle still reads old data
    assert handle.get(key_bytes) == b"one"
    # Reader reads new data
    assert reader.get(1) == b"two"

    # Release handle — triggers deferred cleanup
    handle.close()
    assert old_state.refcount == 0

    reader.close()


def test_concurrent_reader_refresh_fails_reader_stays_on_old_manifest(
    tmp_path,
) -> None:
    """If load_manifest raises during refresh, reader keeps serving from old manifest."""
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_store = _MutableManifestStore(manifests, "mem://manifest/one")
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=manifest_store,
        reader_factory=_fake_reader_factory(stores),
    )

    # Point to a new manifest that will fail to load
    manifest_store.current_ref = "mem://manifest/broken"
    original_load = manifest_store.load_manifest

    def failing_load(ref: str) -> ParsedManifest:
        if ref == "mem://manifest/broken":
            raise ManifestParseError("corrupt manifest")
        return original_load(ref)

    manifest_store.load_manifest = failing_load  # type: ignore[assignment]

    # Refresh should return False (no change) and log the error, not raise
    assert reader.refresh() is False

    # Reader should still serve from old manifest
    assert reader.get(1) == b"val"
    assert reader._state.manifest_ref == "mem://manifest/one"

    reader.close()


# ---------------------------------------------------------------------------
# Lock ordering validation
# ---------------------------------------------------------------------------


class TestOrderedLock:
    """Verify _OrderedLock detects lock ordering violations."""

    def test_correct_order_succeeds(self) -> None:
        """Acquiring locks in ascending level order raises no error."""
        from shardyfusion.reader._state import _OrderedLock

        lock_a = _OrderedLock(0, "first")
        lock_b = _OrderedLock(1, "second")

        with lock_a:
            with lock_b:
                assert lock_a.locked()
                assert lock_b.locked()

    def test_reverse_order_raises(self) -> None:
        """Acquiring a lower-level lock while holding a higher one raises."""
        from shardyfusion.reader._state import _OrderedLock

        lock_a = _OrderedLock(0, "first")
        lock_b = _OrderedLock(1, "second")

        with lock_b:
            with pytest.raises(AssertionError, match="Lock ordering violation"):
                lock_a.__enter__()

    def test_same_level_raises(self) -> None:
        """Acquiring two locks at the same level raises (prevents self-deadlock)."""
        from shardyfusion.reader._state import _OrderedLock

        lock_x = _OrderedLock(0, "x")
        lock_y = _OrderedLock(0, "y")

        with lock_x:
            with pytest.raises(AssertionError, match="Lock ordering violation"):
                lock_y.__enter__()

    def test_standalone_acquisition_succeeds(self) -> None:
        """Acquiring a higher-level lock without holding any other lock works."""
        from shardyfusion.reader._state import _OrderedLock

        lock_b = _OrderedLock(1, "second")

        with lock_b:
            assert lock_b.locked()

    def test_reader_lock_ordering_holds(self, tmp_path) -> None:
        """ConcurrentShardedReader operations don't trigger ordering violations."""
        manifests = {"mem://manifest/one": _manifest("mem://db/one")}
        stores: dict[str, dict[bytes, bytes]] = {
            "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
        }

        reader = ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_MutableManifestStore(manifests, "mem://manifest/one"),
            reader_factory=_fake_reader_factory(stores),
        )

        # Exercise all code paths that touch locks
        reader.get(1)
        reader.snapshot_info()
        reader.shard_details()
        reader.route_key(1)
        _ = reader.key_encoding
        reader.shard_for_key(1)
        reader.shards_for_keys([1])
        with reader.reader_for_key(1) as h:
            h.get((1).to_bytes(8, "big", signed=False))
        reader.refresh()
        reader.close()


def test_do_refresh_without_refresh_lock_raises(tmp_path) -> None:
    """Calling _do_refresh directly without _refresh_lock raises AssertionError."""
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    reader = ConcurrentShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_store=_MutableManifestStore(manifests, "mem://manifest/one"),
        reader_factory=_fake_reader_factory(stores),
    )

    with pytest.raises(AssertionError, match="_refresh_lock"):
        reader._do_refresh()

    reader.close()


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestReaderParameterValidation:
    """Validate that invalid constructor parameters are rejected eagerly."""

    def test_max_workers_zero_raises(self, tmp_path) -> None:
        manifests = {"mem://manifest/one": _manifest("mem://db/one")}
        stores: dict[str, dict[bytes, bytes]] = {
            "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
        }
        with pytest.raises(ValueError, match="positive integer"):
            ConcurrentShardedReader(
                s3_prefix="s3://bucket/prefix",
                local_root=str(tmp_path),
                manifest_store=_MutableManifestStore(manifests, "mem://manifest/one"),
                reader_factory=_fake_reader_factory(stores),
                max_workers=0,
            )

    def test_max_workers_negative_raises(self, tmp_path) -> None:
        manifests = {"mem://manifest/one": _manifest("mem://db/one")}
        stores: dict[str, dict[bytes, bytes]] = {
            "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
        }
        with pytest.raises(ValueError, match="positive integer"):
            ConcurrentShardedReader(
                s3_prefix="s3://bucket/prefix",
                local_root=str(tmp_path),
                manifest_store=_MutableManifestStore(manifests, "mem://manifest/one"),
                reader_factory=_fake_reader_factory(stores),
                max_workers=-3,
            )

    def test_max_workers_one_is_valid(self, tmp_path) -> None:
        manifests = {"mem://manifest/one": _manifest("mem://db/one")}
        stores: dict[str, dict[bytes, bytes]] = {
            "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
        }
        reader = ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_MutableManifestStore(manifests, "mem://manifest/one"),
            reader_factory=_fake_reader_factory(stores),
            max_workers=1,
        )
        reader.close()

    def test_pool_checkout_timeout_zero_raises(self, tmp_path) -> None:
        manifests = {"mem://manifest/one": _manifest("mem://db/one")}
        stores: dict[str, dict[bytes, bytes]] = {
            "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
        }
        with pytest.raises(ValueError, match="must be > 0"):
            ConcurrentShardedReader(
                s3_prefix="s3://bucket/prefix",
                local_root=str(tmp_path),
                manifest_store=_MutableManifestStore(manifests, "mem://manifest/one"),
                reader_factory=_fake_reader_factory(stores),
                pool_checkout_timeout=timedelta(seconds=0),
            )

    def test_pool_checkout_timeout_negative_raises(self, tmp_path) -> None:
        manifests = {"mem://manifest/one": _manifest("mem://db/one")}
        stores: dict[str, dict[bytes, bytes]] = {
            "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
        }
        with pytest.raises(ValueError, match="must be > 0"):
            ConcurrentShardedReader(
                s3_prefix="s3://bucket/prefix",
                local_root=str(tmp_path),
                manifest_store=_MutableManifestStore(manifests, "mem://manifest/one"),
                reader_factory=_fake_reader_factory(stores),
                pool_checkout_timeout=timedelta(seconds=-5.0),
            )

    def test_sharded_reader_max_workers_zero_raises(self, tmp_path) -> None:
        """Validation applies to ShardedReader too (inherited from base)."""
        manifests = {"mem://manifest/one": _manifest("mem://db/one")}
        stores: dict[str, dict[bytes, bytes]] = {
            "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
        }
        with pytest.raises(ValueError, match="positive integer"):
            ShardedReader(
                s3_prefix="s3://bucket/prefix",
                local_root=str(tmp_path),
                manifest_store=_MutableManifestStore(manifests, "mem://manifest/one"),
                reader_factory=_fake_reader_factory(stores),
                max_workers=0,
            )


class TestReaderPoolInternals:
    """Verify defensive invariants on _ReaderPool."""

    def test_reader_pool_queue_bounded(self) -> None:
        from shardyfusion.reader._state import _ReaderPool

        readers = [_FakeReader({}), _FakeReader({}), _FakeReader({})]
        pool = _ReaderPool(readers)
        assert pool._indexes.maxsize == 3

    def test_reader_pool_checkout_timeout_zero_raises(self) -> None:
        from shardyfusion.reader._state import _ReaderPool

        with pytest.raises(ValueError, match="must be > 0"):
            _ReaderPool([_FakeReader({})], checkout_timeout=timedelta(seconds=0))

    def test_reader_pool_readers_is_tuple(self) -> None:
        from shardyfusion.reader._state import _ReaderPool

        readers = [_FakeReader({}), _FakeReader({})]
        pool = _ReaderPool(readers)
        assert isinstance(pool._readers, tuple)
