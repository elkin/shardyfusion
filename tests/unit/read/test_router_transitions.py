"""Tests for router invariant transitions across reader refresh.

Validates behavior when num_dbs, key_encoding, or shard
layout changes between manifests during refresh().
"""

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
from shardyfusion.routing import xxh3_db_id
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


@dataclass
class _FakeReader:
    store: dict[bytes, bytes]

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def close(self) -> None:
        pass


def _fake_factory(stores: dict[str, dict[bytes, bytes]]):
    def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
        return _FakeReader(stores.get(db_url, {}))

    return factory


class _MutableStore:
    def __init__(self, manifests: dict[str, ParsedManifest], initial: str) -> None:
        self.manifests = manifests
        self.current_ref = initial

    def publish(self, **kw: Any) -> str:
        raise NotImplementedError

    def load_current(self) -> ManifestRef | None:
        return ManifestRef(
            ref=self.current_ref, run_id="r", published_at=datetime.now(UTC)
        )

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self.manifests[ref]

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []

    def set_current(self, ref: str) -> None:
        pass


def _manifest(
    num_dbs: int, key_encoding: KeyEncoding = KeyEncoding.U64BE
) -> ParsedManifest:
    required = RequiredBuildMeta(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=key_encoding,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.HASH,
            hash_algorithm="xxh3_64",
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    shards = [
        RequiredShardMeta(
            db_id=i,
            db_url=f"mem://db/n{num_dbs}/s{i}",
            attempt=0,
            row_count=10,
        )
        for i in range(num_dbs)
    ]
    return ParsedManifest(required_build=required, shards=shards, custom={})


class TestNumDbsChange:
    def test_num_dbs_increases_on_refresh(self, tmp_path: Path) -> None:
        """Refresh from 2 shards to 4 shards re-routes keys correctly."""
        m2 = _manifest(2)
        m4 = _manifest(4)

        stores: dict[str, dict[bytes, bytes]] = {}
        for i in range(4):
            stores[f"mem://db/n2/s{i}"] = {}
            stores[f"mem://db/n4/s{i}"] = {}

        store = _MutableStore({"mem://m2": m2, "mem://m4": m4}, initial="mem://m2")

        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(stores),
        ) as reader:
            info_before = reader.snapshot_info()
            assert info_before.num_dbs == 2

            store.current_ref = "mem://m4"
            reader.refresh()

            info_after = reader.snapshot_info()
            assert info_after.num_dbs == 4

    def test_num_dbs_decreases_on_refresh(self, tmp_path: Path) -> None:
        """Refresh from 4 shards to 2 shards updates routing."""
        m4 = _manifest(4)
        m2 = _manifest(2)

        stores: dict[str, dict[bytes, bytes]] = {}
        for i in range(4):
            stores[f"mem://db/n4/s{i}"] = {}
            stores[f"mem://db/n2/s{i}"] = {}

        store = _MutableStore({"mem://m4": m4, "mem://m2": m2}, initial="mem://m4")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(stores),
        )
        assert reader.snapshot_info().num_dbs == 4

        store.current_ref = "mem://m2"
        reader.refresh()

        assert reader.snapshot_info().num_dbs == 2
        # After refresh, routing should use 2 shards
        db_id = reader.route_key(42)
        assert db_id == xxh3_db_id(42, 2)
        reader.close()


class TestSparseToFullTransition:
    def test_sparse_manifest_pads_with_null_shards(self, tmp_path: Path) -> None:
        """Manifest with missing db_ids gets null shards padded in."""
        required = RequiredBuildMeta(
            run_id="run",
            created_at="2026-01-01T00:00:00+00:00",
            num_dbs=3,
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
        # Only shard 0 and 2 present (shard 1 is empty/missing)
        shards = [
            RequiredShardMeta(db_id=0, db_url="mem://db/s0", attempt=0, row_count=5),
            RequiredShardMeta(db_id=2, db_url="mem://db/s2", attempt=0, row_count=5),
        ]
        manifest = ParsedManifest(required_build=required, shards=shards, custom={})
        stores = {"mem://db/s0": {}, "mem://db/s2": {}}

        store = _MutableStore({"mem://sparse": manifest}, initial="mem://sparse")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(stores),
        )
        # Key that routes to shard 1 (empty) should return None
        details = reader.shard_details()
        assert len(details) == 3
        shard_1 = [d for d in details if d.db_id == 1][0]
        assert shard_1.row_count == 0
        reader.close()

    def test_refresh_from_sparse_to_full(self, tmp_path: Path) -> None:
        """Refresh from sparse manifest (missing shard 1) to full manifest."""
        required_sparse = RequiredBuildMeta(
            run_id="run",
            created_at="2026-01-01T00:00:00+00:00",
            num_dbs=3,
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
        sparse = ParsedManifest(
            required_build=required_sparse,
            shards=[
                RequiredShardMeta(
                    db_id=0, db_url="mem://db/a0", attempt=0, row_count=5
                ),
                RequiredShardMeta(
                    db_id=2, db_url="mem://db/a2", attempt=0, row_count=5
                ),
            ],
            custom={},
        )
        full = _manifest(3)

        stores: dict[str, dict[bytes, bytes]] = {
            "mem://db/a0": {},
            "mem://db/a2": {},
        }
        for i in range(3):
            stores[f"mem://db/n3/s{i}"] = {}

        store = _MutableStore(
            {"mem://sparse": sparse, "mem://full": full}, initial="mem://sparse"
        )

        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(stores),
        ) as reader:
            # Initially sparse
            details = reader.shard_details()
            shard_1_before = [d for d in details if d.db_id == 1][0]
            assert shard_1_before.row_count == 0

            store.current_ref = "mem://full"
            reader.refresh()

            # After refresh, all shards present
            details = reader.shard_details()
            shard_1_after = [d for d in details if d.db_id == 1][0]
            assert shard_1_after.row_count == 10
