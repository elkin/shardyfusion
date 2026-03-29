"""Tests for null shard reader handling when manifests contain empty shards (db_url=None)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.reader import ConcurrentShardedReader, ShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeReader:
    store: dict[bytes, bytes]

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def close(self) -> None:
        pass


class _StaticManifestStore:
    def __init__(
        self, manifest: ParsedManifest, ref: str = "mem://manifest/v1"
    ) -> None:
        self._manifest = manifest
        self._ref = ref

    def publish(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def load_current(self) -> ManifestRef | None:
        return ManifestRef(ref=self._ref, run_id="run", published_at=datetime.now(UTC))

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifest

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []

    def set_current(self, ref: str) -> None:
        pass


def _required_build(num_dbs: int = 2) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )


def _manifest_with_empty_shard() -> ParsedManifest:
    """Manifest with shard 0 = real data, shard 1 = empty (omitted from manifest).

    The SnapshotRouter will synthesize a metadata-only entry for shard 1
    with db_url=None and row_count=0.
    """
    return ParsedManifest(
        required_build=_required_build(num_dbs=2),
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


def _fake_reader_factory(stores: dict[str, dict[bytes, bytes]]):
    def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
        return _FakeReader(stores[db_url])

    return factory


# ---------------------------------------------------------------------------
# ShardedReader tests
# ---------------------------------------------------------------------------


class TestShardedReaderNullShards:
    def test_get_returns_none_for_empty_shard(self, tmp_path: Path) -> None:
        """Keys routed to an empty shard return None without opening a real reader."""
        manifest = _manifest_with_empty_shard()
        # key=1 routes to shard 0 under xxh3_64, key=0 routes to shard 1
        stores = {"mem://db/shard0": {(1).to_bytes(8, "big"): b"val1"}}

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(manifest),
            reader_factory=_fake_reader_factory(stores),
        )

        # Shard 0 has data — key=1 routes to shard 0
        assert reader.get(1) == b"val1"

        # Shard 1 is empty — key=0 routes to shard 1, should return None
        result = reader.get(0)
        assert result is None

        reader.close()

    def test_shard_details_includes_empty_shard(self, tmp_path: Path) -> None:
        manifest = _manifest_with_empty_shard()
        stores = {"mem://db/shard0": {}}

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(manifest),
            reader_factory=_fake_reader_factory(stores),
        )

        details = reader.shard_details()
        assert len(details) == 2
        empty = [d for d in details if d.db_url is None]
        assert len(empty) == 1
        assert empty[0].db_id == 1
        assert empty[0].row_count == 0

        reader.close()

    def test_factory_not_called_for_empty_shard(self, tmp_path: Path) -> None:
        """Reader factory is never invoked for shards with db_url=None."""
        manifest = _manifest_with_empty_shard()
        factory_calls: list[str] = []

        def tracking_factory(
            *, db_url: str, local_dir: Path, checkpoint_id: str | None
        ):
            factory_calls.append(db_url)
            return _FakeReader({})

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(manifest),
            reader_factory=tracking_factory,
        )

        # Only shard 0 should have been opened
        assert factory_calls == ["mem://db/shard0"]
        reader.close()


# ---------------------------------------------------------------------------
# ConcurrentShardedReader tests
# ---------------------------------------------------------------------------


class TestConcurrentReaderNullShards:
    def test_lock_mode(self, tmp_path: Path) -> None:
        manifest = _manifest_with_empty_shard()
        stores = {"mem://db/shard0": {(1).to_bytes(8, "big"): b"val1"}}

        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(manifest),
            reader_factory=_fake_reader_factory(stores),
            thread_safety="lock",
        ) as reader:
            assert reader.get(1) == b"val1"
            assert reader.get(0) is None

    def test_pool_mode(self, tmp_path: Path) -> None:
        manifest = _manifest_with_empty_shard()
        stores = {"mem://db/shard0": {(1).to_bytes(8, "big"): b"val1"}}

        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(manifest),
            reader_factory=_fake_reader_factory(stores),
            thread_safety="pool",
        ) as reader:
            assert reader.get(1) == b"val1"
            assert reader.get(0) is None

    def test_multi_get_mixed_shards(self, tmp_path: Path) -> None:
        """multi_get across real and empty shards returns mixed results."""
        manifest = _manifest_with_empty_shard()
        stores = {"mem://db/shard0": {(1).to_bytes(8, "big"): b"val1"}}

        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(manifest),
            reader_factory=_fake_reader_factory(stores),
        ) as reader:
            results = reader.multi_get([1, 0])
            assert results[1] == b"val1"
            assert results[0] is None


# ---------------------------------------------------------------------------
# Manifest round-trip test
# ---------------------------------------------------------------------------


class TestManifestSparseShards:
    def test_sparse_manifest_round_trip(self) -> None:
        """Manifest with only non-empty shards round-trips; router fills gaps."""
        from shardyfusion.manifest import SqliteManifestBuilder
        from shardyfusion.manifest_store import parse_sqlite_manifest
        from shardyfusion.routing import SnapshotRouter

        builder = SqliteManifestBuilder()
        build = _required_build(num_dbs=3)
        # Only shard 0 and 2 have data; shard 1 is omitted (empty)
        shards = [
            RequiredShardMeta(
                db_id=0,
                db_url="s3://bucket/prefix/db=00000",
                attempt=0,
                row_count=5,
            ),
            RequiredShardMeta(
                db_id=2,
                db_url="s3://bucket/prefix/db=00002",
                attempt=0,
                row_count=3,
            ),
        ]

        artifact = builder.build(
            required_build=build,
            shards=shards,
            custom_fields={},
        )

        parsed = parse_sqlite_manifest(artifact.payload)
        assert len(parsed.shards) == 2

        router = SnapshotRouter(parsed.required_build, parsed.shards)
        assert len(router.shards) == 3
        assert router.shards[0].db_url == "s3://bucket/prefix/db=00000"
        assert router.shards[1].db_url is None  # synthesized by router
        assert router.shards[1].row_count == 0
        assert router.shards[2].db_url == "s3://bucket/prefix/db=00002"
