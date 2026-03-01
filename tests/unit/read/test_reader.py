from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from slatedb_spark_sharded.errors import ReaderStateError, SlateDbApiError
from slatedb_spark_sharded.manifest import (
    CurrentPointer,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from slatedb_spark_sharded.metrics import MetricEvent
from slatedb_spark_sharded.reader import SlateDbReaderFactory, SlateShardedReader
from slatedb_spark_sharded.sharding_types import KeyEncoding, ShardingStrategy
from slatedb_spark_sharded.testing import ListMetricsCollector


class _MutableManifestReader:
    def __init__(self, manifests: dict[str, ParsedManifest], initial_ref: str) -> None:
        self.manifests = manifests
        self.current_ref = initial_ref

    def load_current(self) -> CurrentPointer | None:
        return CurrentPointer(
            manifest_ref=self.current_ref,
            manifest_content_type="application/json",
            run_id="run",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    def load_manifest(
        self, ref: str, content_type: str | None = None
    ) -> ParsedManifest:
        _ = content_type
        return self.manifests[ref]


@dataclass
class _FakeReader:
    store: dict[bytes, bytes]

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def close(self) -> None:
        return None


def _fake_reader_factory(stores: dict[str, dict[bytes, bytes]]):
    """Return a ShardReaderFactory that routes db_url → in-memory store."""

    def factory(*, db_url: str, local_dir: str, checkpoint_id: str | None):
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


def test_refresh_swaps_manifest_ref_and_readers(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest("mem://db/one"),
        "mem://manifest/two": _manifest("mem://db/two"),
    }
    manifest_reader = _MutableManifestReader(manifests, "mem://manifest/one")

    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"one"},
        "mem://db/two": {(1).to_bytes(8, "big", signed=False): b"two"},
    }

    with SlateShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_reader=manifest_reader,
        reader_factory=_fake_reader_factory(stores),
    ) as reader:
        assert reader.get(1) == b"one"
        manifest_reader.current_ref = "mem://manifest/two"

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
        local_dir="/tmp/local",
        checkpoint_id="ckpt-1",
    )

    assert reader is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("/tmp/local",)
    assert kwargs["url"] == "s3://bucket/db"
    assert kwargs["checkpoint_id"] == "ckpt-1"
    assert kwargs["env_file"] == "slatedb.env"


class _NullManifestReader:
    """Always returns None for CURRENT (simulates missing pointer)."""

    def load_current(self) -> CurrentPointer | None:
        return None

    def load_manifest(
        self, ref: str, content_type: str | None = None
    ) -> ParsedManifest:
        raise AssertionError("load_manifest should not be called")


def test_load_initial_state_raises_reader_state_error_when_no_current() -> None:
    with pytest.raises(ReaderStateError, match="CURRENT pointer not found"):
        SlateShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root="/tmp/unused",
            manifest_reader=_NullManifestReader(),
        )


def test_closed_reader_get_raises_reader_state_error(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_reader = _MutableManifestReader(manifests, "mem://manifest/one")

    reader = SlateShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_reader=manifest_reader,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _FakeReader({}),
    )
    reader.close()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.get(1)


def test_closed_reader_refresh_raises_reader_state_error(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_reader = _MutableManifestReader(manifests, "mem://manifest/one")

    reader = SlateShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_reader=manifest_reader,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _FakeReader({}),
    )
    reader.close()

    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.refresh()


def test_context_manager_returns_self_and_calls_close(tmp_path) -> None:
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_reader = _MutableManifestReader(manifests, "mem://manifest/one")

    reader = SlateShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_reader=manifest_reader,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _FakeReader({}),
    )

    with reader as ctx:
        assert ctx is reader

    # After exiting the context, the reader should be closed.
    with pytest.raises(ReaderStateError, match="Reader is closed"):
        reader.get(1)


def _manifest_2shard(db_url_0: str, db_url_1: str) -> ParsedManifest:
    """Two-shard range manifest: keys <=5 → shard 0, keys >=6 → shard 1."""
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


def test_multi_get_shard_failure_raises_slate_db_api_error(tmp_path) -> None:
    manifests = {
        "mem://manifest/one": _manifest_2shard("mem://db/zero", "mem://db/one")
    }
    manifest_reader = _MutableManifestReader(manifests, "mem://manifest/one")

    class _BrokenReader:
        def get(self, key: bytes) -> bytes | None:
            raise OSError("disk error")

        def close(self) -> None:
            pass

    with SlateShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_reader=manifest_reader,
        reader_factory=lambda *, db_url, local_dir, checkpoint_id: _BrokenReader(),
        max_workers=2,
    ) as reader:
        # key=1 routes to shard 0, key=6 routes to shard 1 → both shards in executor
        with pytest.raises(SlateDbApiError, match="db_id="):
            reader.multi_get([1, 6])


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


def test_metrics_emitted_on_reader_lifecycle(tmp_path) -> None:
    mc = ListMetricsCollector()
    manifests = {"mem://manifest/one": _manifest("mem://db/one")}
    manifest_reader = _MutableManifestReader(manifests, "mem://manifest/one")

    stores: dict[str, dict[bytes, bytes]] = {
        "mem://db/one": {(1).to_bytes(8, "big", signed=False): b"val"},
    }

    with SlateShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_reader=manifest_reader,
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
    manifest_reader = _MutableManifestReader(manifests, "mem://manifest/one")

    stores: dict[str, dict[bytes, bytes]] = {"mem://db/one": {}}

    with SlateShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root=str(tmp_path),
        manifest_reader=manifest_reader,
        reader_factory=_fake_reader_factory(stores),
        metrics_collector=mc,
    ) as reader:
        result = reader.get(1)
        assert result is None

    get_payload = next(p for e, p in mc.events if e is MetricEvent.READER_GET)
    assert get_payload["found"] is False
