"""Tests for multi_get() partial failure across shards.

Validates behavior when one shard read fails while others succeed,
for both ShardedReader and ConcurrentShardedReader.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from shardyfusion.errors import DbAdapterError
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.reader import ConcurrentShardedReader, ShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy


@dataclass
class _FakeReader:
    store: dict[bytes, bytes]

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def close(self) -> None:
        pass


@dataclass
class _FailingReader:
    """Reader that always raises on get()."""

    error: Exception

    def get(self, key: bytes) -> bytes | None:
        raise self.error

    def close(self) -> None:
        pass


class _FixedStore:
    def __init__(self, manifest: ParsedManifest) -> None:
        self._manifest = manifest

    def publish(self, **kw: Any) -> str:
        raise NotImplementedError

    def load_current(self) -> ManifestRef | None:
        return ManifestRef(ref="mem://m", run_id="r", published_at=datetime.now(UTC))

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifest

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []

    def set_current(self, ref: str) -> None:
        pass


def _2shard_manifest() -> ParsedManifest:
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
            RequiredShardMeta(db_id=0, db_url="mem://db/0", attempt=0, row_count=10),
            RequiredShardMeta(db_id=1, db_url="mem://db/1", attempt=0, row_count=10),
        ],
        custom={},
    )


def _find_keys_for_shards(num_dbs: int = 2) -> dict[int, int]:
    """Find one key that routes to each shard."""
    from shardyfusion.routing import xxh3_db_id

    found: dict[int, int] = {}
    for i in range(1000):
        db_id = xxh3_db_id(i, num_dbs)
        if db_id not in found:
            found[db_id] = i
        if len(found) == num_dbs:
            break
    return found


class TestMultiGetPartialFailure:
    def test_shard_failure_raises_for_simple_reader(self, tmp_path: Path) -> None:
        """ShardedReader multi_get raises when one shard fails."""
        manifest = _2shard_manifest()
        keys_by_shard = _find_keys_for_shards(2)

        def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
            if db_url == "mem://db/1":
                return _FailingReader(error=RuntimeError("shard 1 down"))
            key = keys_by_shard[0]
            encoded = key.to_bytes(8, "big", signed=False)
            return _FakeReader({encoded: b"val"})

        store = _FixedStore(manifest)
        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=factory,
        )

        # multi_get with keys spanning both shards should raise
        keys = [keys_by_shard[0], keys_by_shard[1]]
        with pytest.raises((RuntimeError, DbAdapterError)):
            reader.multi_get(keys)
        reader.close()

    def test_shard_failure_raises_for_concurrent_reader(self, tmp_path: Path) -> None:
        """ConcurrentShardedReader multi_get raises when one shard fails."""
        manifest = _2shard_manifest()
        keys_by_shard = _find_keys_for_shards(2)

        def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
            if db_url == "mem://db/1":
                return _FailingReader(error=RuntimeError("shard 1 down"))
            return _FakeReader({})

        store = _FixedStore(manifest)
        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=factory,
        ) as reader:
            keys = [keys_by_shard[0], keys_by_shard[1]]
            with pytest.raises((RuntimeError, DbAdapterError)):
                reader.multi_get(keys)

    def test_all_shards_fail_raises(self, tmp_path: Path) -> None:
        """multi_get raises when all shards fail."""
        manifest = _2shard_manifest()
        keys_by_shard = _find_keys_for_shards(2)

        def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
            return _FailingReader(error=DbAdapterError("all down"))

        store = _FixedStore(manifest)
        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=factory,
        )
        with pytest.raises(DbAdapterError):
            reader.multi_get([keys_by_shard[0], keys_by_shard[1]])
        reader.close()

    def test_single_shard_multi_get_failure(self, tmp_path: Path) -> None:
        """multi_get with all keys on the same failing shard raises."""
        manifest = _2shard_manifest()
        keys_by_shard = _find_keys_for_shards(2)

        def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
            if db_url == "mem://db/0":
                return _FailingReader(error=DbAdapterError("shard 0 down"))
            return _FakeReader({})

        store = _FixedStore(manifest)
        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=factory,
        )
        # Only keys for shard 0 — the failing one
        with pytest.raises(DbAdapterError, match="shard 0 down"):
            reader.multi_get([keys_by_shard[0]])
        reader.close()

    def test_concurrent_reader_with_executor_partial_failure(
        self, tmp_path: Path
    ) -> None:
        """ConcurrentShardedReader with max_workers > 1 raises on shard failure."""
        manifest = _2shard_manifest()
        keys_by_shard = _find_keys_for_shards(2)

        def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None):
            if db_url == "mem://db/1":
                return _FailingReader(error=DbAdapterError("shard 1 read error"))
            return _FakeReader({})

        store = _FixedStore(manifest)
        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=factory,
            max_workers=2,
        ) as reader:
            keys = [keys_by_shard[0], keys_by_shard[1]]
            with pytest.raises(DbAdapterError, match="shard 1"):
                reader.multi_get(keys)
