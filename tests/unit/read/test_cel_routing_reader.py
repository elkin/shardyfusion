"""Tests for CEL routing through the reader path.

Validates that ShardedReader and ConcurrentShardedReader correctly
route keys when using CEL sharding with routing_context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.reader import ConcurrentShardedReader, ShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

cel_expr_python = pytest.importorskip("cel_expr_python")

pytestmark = pytest.mark.cel


@dataclass
class _FakeReader:
    store: dict[bytes, bytes]

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def close(self) -> None:
        pass


def _fake_factory(stores: dict[str, dict[bytes, bytes]]):
    def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
        return _FakeReader(stores[db_url])

    return factory


class _FixedManifestStore:
    def __init__(self, manifest: ParsedManifest, ref: str) -> None:
        self._manifest = manifest
        self._ref = ref

    def publish(self, **kw: Any) -> str:
        raise NotImplementedError

    def load_current(self) -> ManifestRef | None:
        return ManifestRef(ref=self._ref, run_id="run", published_at=datetime.now(UTC))

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifest

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []

    def set_current(self, ref: str) -> None:
        pass


def _cel_key_only_manifest(num_dbs: int = 4) -> ParsedManifest:
    """CEL key-only mode: key % num_dbs."""
    required = RequiredBuildMeta(
        run_id="run-cel",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="key",
        key_encoding=KeyEncoding.UTF8,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr=f"key % {num_dbs}",
            cel_columns={"key": "int"},
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    shards = [
        RequiredShardMeta(
            db_id=i,
            db_url=f"mem://db/{i}",
            attempt=0,
            row_count=10,
        )
        for i in range(num_dbs)
    ]
    return ParsedManifest(required_build=required, shards=shards, custom={})


def _cel_multi_column_manifest(num_dbs: int = 3) -> ParsedManifest:
    """CEL multi-column direct mode using explicit integer routing."""
    required = RequiredBuildMeta(
        run_id="run-cel-mc",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="key",
        key_encoding=KeyEncoding.UTF8,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr='region == "ap" ? 0 : region == "eu" ? 1 : 2',
            cel_columns={"region": "string"},
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    shards = [
        RequiredShardMeta(
            db_id=i,
            db_url=f"mem://db/{i}",
            attempt=0,
            row_count=10,
        )
        for i in range(num_dbs)
    ]
    return ParsedManifest(required_build=required, shards=shards, custom={})


def _cel_categorical_manifest(num_dbs: int = 3) -> ParsedManifest:
    required = RequiredBuildMeta(
        run_id="run-cel-cat",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="key",
        key_encoding=KeyEncoding.UTF8,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="region",
            cel_columns={"region": "string"},
            routing_values=["ap", "eu", "us"],
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
        format_version=3,
    )
    shards = [
        RequiredShardMeta(
            db_id=i,
            db_url=f"mem://db/{i}",
            attempt=0,
            row_count=10,
        )
        for i in range(num_dbs)
    ]
    return ParsedManifest(required_build=required, shards=shards, custom={})


class TestCelKeyOnlyReaderRouting:
    """CEL key-only mode: reader routes via key % num_dbs."""

    def _stores(self, num_dbs: int = 4) -> dict[str, dict[bytes, bytes]]:
        """Each shard has data for keys that route to it."""
        stores: dict[str, dict[bytes, bytes]] = {}
        for i in range(num_dbs):
            stores[f"mem://db/{i}"] = {
                f"shard-{i}".encode(): f"val-{i}".encode(),
            }
        return stores

    def test_simple_reader_get_routes_correctly(self, tmp_path: Path) -> None:
        manifest = _cel_key_only_manifest(4)
        stores = self._stores(4)
        store = _FixedManifestStore(manifest, "mem://manifest/cel")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(stores),
        )
        # Key 0 → shard 0, key 1 → shard 1, etc.
        assert reader.route_key(0) == 0
        assert reader.route_key(1) == 1
        assert reader.route_key(5) == 1  # 5 % 4 = 1
        reader.close()

    def test_concurrent_reader_get_routes_correctly(self, tmp_path: Path) -> None:
        manifest = _cel_key_only_manifest(4)
        stores = self._stores(4)
        store = _FixedManifestStore(manifest, "mem://manifest/cel")

        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(stores),
        ) as reader:
            assert reader.route_key(0) == 0
            assert reader.route_key(3) == 3


class TestCelMultiColumnReaderRouting:
    """CEL multi-column mode: reader routes via routing_context."""

    def _stores(self) -> dict[str, dict[bytes, bytes]]:
        return {
            "mem://db/0": {b"k": b"ap-shard"},
            "mem://db/1": {b"k": b"eu-shard"},
            "mem://db/2": {b"k": b"us-shard"},
        }

    def test_get_with_routing_context(self, tmp_path: Path) -> None:
        manifest = _cel_multi_column_manifest(3)
        stores = self._stores()
        store = _FixedManifestStore(manifest, "mem://manifest/cel-mc")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(stores),
        )
        val = reader.get("k", routing_context={"region": "ap"})
        assert val == b"ap-shard"
        reader.close()

    def test_multi_get_with_routing_context(self, tmp_path: Path) -> None:
        manifest = _cel_multi_column_manifest(3)
        stores = self._stores()
        store = _FixedManifestStore(manifest, "mem://manifest/cel-mc")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(stores),
        )
        # routing_context groups ALL keys to one shard
        result = reader.multi_get(["k"], routing_context={"region": "ap"})
        assert result["k"] == b"ap-shard"
        reader.close()

    def test_route_key_with_routing_context(self, tmp_path: Path) -> None:
        manifest = _cel_multi_column_manifest(3)
        stores = self._stores()
        store = _FixedManifestStore(manifest, "mem://manifest/cel-mc")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(stores),
        )
        assert reader.route_key("k", routing_context={"region": "ap"}) == 0
        assert reader.route_key("k", routing_context={"region": "eu"}) == 1
        assert reader.route_key("k", routing_context={"region": "us"}) == 2
        reader.close()

    def test_route_one_without_context_raises(self, tmp_path: Path) -> None:
        """Calling route_key without routing_context on multi-column CEL raises."""
        manifest = _cel_multi_column_manifest(3)
        stores = self._stores()
        store = _FixedManifestStore(manifest, "mem://manifest/cel-mc")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(stores),
        )
        with pytest.raises(ValueError, match="routing_context"):
            reader.route_key("k")
        reader.close()


class TestCelCategoricalReaderRouting:
    def _stores(self) -> dict[str, dict[bytes, bytes]]:
        return {
            "mem://db/0": {b"k": b"ap-shard"},
            "mem://db/1": {b"k": b"eu-shard"},
            "mem://db/2": {b"k": b"us-shard"},
        }

    def test_get_unknown_categorical_token_returns_miss(self, tmp_path: Path) -> None:
        manifest = _cel_categorical_manifest(3)
        store = _FixedManifestStore(manifest, "mem://manifest/cel-cat")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(self._stores()),
        )
        assert reader.get("k", routing_context={"region": "jp"}) is None
        reader.close()

    def test_multi_get_unknown_categorical_token_returns_misses(
        self,
        tmp_path: Path,
    ) -> None:
        manifest = _cel_categorical_manifest(3)
        store = _FixedManifestStore(manifest, "mem://manifest/cel-cat")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(self._stores()),
        )
        result = reader.multi_get(["k", "missing"], routing_context={"region": "jp"})
        assert result == {"k": None, "missing": None}
        reader.close()

    def test_route_key_unknown_categorical_token_raises(self, tmp_path: Path) -> None:
        manifest = _cel_categorical_manifest(3)
        store = _FixedManifestStore(manifest, "mem://manifest/cel-cat")

        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory(self._stores()),
        ) as reader:
            with pytest.raises(ValueError, match="not present"):
                reader.route_key("k", routing_context={"region": "jp"})
