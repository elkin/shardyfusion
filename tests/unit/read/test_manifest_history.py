"""Tests for manifest history, listing, set_current, and reader fallback."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from shardyfusion.errors import ManifestParseError, ReaderStateError
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.reader import ConcurrentShardedReader, ShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_meta(run_id: str = "run-1") -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id=run_id,
        created_at=datetime.now(UTC),
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


def _one_shard() -> list[RequiredShardMeta]:
    return [
        RequiredShardMeta(
            db_id=0,
            db_url="s3://bucket/prefix/db=00000",
            attempt=0,
            row_count=1,
            db_bytes=0,
        )
    ]


class _FakeReader:
    def get(self, key: bytes) -> bytes | None:
        return None

    def close(self) -> None:
        pass


def _fake_factory(
    *, db_url: str, local_dir: Path, checkpoint_id: str | None, **_kwargs
):
    return _FakeReader()


# ---------------------------------------------------------------------------
# Fake store that can serve malformed manifests
# ---------------------------------------------------------------------------


class _FallbackTestStore:
    """Store that can return malformed manifest payloads for specific refs."""

    def __init__(self) -> None:
        self._manifests: dict[str, ParsedManifest | None] = {}
        self._history: list[ManifestRef] = []
        self._current_ref: str | None = None

    def add_manifest(self, run_id: str, *, malformed: bool = False) -> str:
        ref = f"mem://manifests/run_id={run_id}/manifest"
        if malformed:
            self._manifests[ref] = None  # None = will raise ManifestParseError
        else:
            self._manifests[ref] = ParsedManifest(
                required_build=_build_meta(run_id),
                shards=_one_shard(),
                custom={},
            )
        mr = ManifestRef(ref=ref, run_id=run_id, published_at=datetime.now(UTC))
        self._history.append(mr)
        self._current_ref = ref
        return ref

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        raise NotImplementedError

    def load_current(self) -> ManifestRef | None:
        if self._current_ref is None:
            return None
        for entry in reversed(self._history):
            if entry.ref == self._current_ref:
                return entry
        return None

    def load_manifest(self, ref: str) -> ParsedManifest:
        if ref not in self._manifests:
            raise ManifestParseError(f"Not found: {ref}")
        manifest = self._manifests[ref]
        if manifest is None:
            raise ManifestParseError(f"Malformed manifest: {ref}")
        return manifest

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return list(reversed(self._history))[:limit]

    def set_current(self, ref: str) -> None:
        self._current_ref = ref


# ---------------------------------------------------------------------------
# InMemoryManifestStore history tests
# ---------------------------------------------------------------------------


class TestInMemoryManifestStoreHistory:
    def test_list_manifests_returns_newest_first(self) -> None:
        store = InMemoryManifestStore()
        store.publish(
            run_id="v1",
            required_build=_build_meta("v1"),
            shards=_one_shard(),
            custom={},
        )
        store.publish(
            run_id="v2",
            required_build=_build_meta("v2"),
            shards=_one_shard(),
            custom={},
        )
        store.publish(
            run_id="v3",
            required_build=_build_meta("v3"),
            shards=_one_shard(),
            custom={},
        )

        refs = store.list_manifests(limit=10)
        assert len(refs) == 3
        assert refs[0].run_id == "v3"
        assert refs[1].run_id == "v2"
        assert refs[2].run_id == "v1"

    def test_list_manifests_respects_limit(self) -> None:
        store = InMemoryManifestStore()
        for i in range(5):
            store.publish(
                run_id=f"v{i}",
                required_build=_build_meta(f"v{i}"),
                shards=_one_shard(),
                custom={},
            )

        refs = store.list_manifests(limit=2)
        assert len(refs) == 2
        assert refs[0].run_id == "v4"

    def test_list_manifests_empty_store(self) -> None:
        store = InMemoryManifestStore()
        assert store.list_manifests() == []

    def test_set_current_changes_pointer(self) -> None:
        store = InMemoryManifestStore()
        ref_v1 = store.publish(
            run_id="v1",
            required_build=_build_meta("v1"),
            shards=_one_shard(),
            custom={},
        )
        store.publish(
            run_id="v2",
            required_build=_build_meta("v2"),
            shards=_one_shard(),
            custom={},
        )

        current = store.load_current()
        assert current is not None
        assert current.run_id == "v2"

        store.set_current(ref_v1)
        current = store.load_current()
        assert current is not None
        assert current.run_id == "v1"

    def test_set_current_rejects_unknown_ref(self) -> None:
        store = InMemoryManifestStore()
        with pytest.raises(KeyError):
            store.set_current("nonexistent://ref")

    def test_load_current_returns_manifest_ref(self) -> None:
        store = InMemoryManifestStore()
        store.publish(
            run_id="v1",
            required_build=_build_meta("v1"),
            shards=_one_shard(),
            custom={},
        )
        current = store.load_current()
        assert current is not None
        assert isinstance(current, ManifestRef)
        assert current.run_id == "v1"
        assert "run_id=v1" in current.ref


# ---------------------------------------------------------------------------
# Reader cold-start fallback tests
# ---------------------------------------------------------------------------


class TestReaderColdStartFallback:
    def test_reader_falls_back_when_latest_manifest_malformed(self, tmp_path) -> None:
        store = _FallbackTestStore()
        store.add_manifest("v1")
        store.add_manifest("v2", malformed=True)  # latest is malformed

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory,
        )

        info = reader.snapshot_info()
        assert info.run_id == "v1"
        reader.close()

    def test_reader_falls_back_skips_multiple_malformed(self, tmp_path) -> None:
        store = _FallbackTestStore()
        store.add_manifest("v1")
        store.add_manifest("v2", malformed=True)
        store.add_manifest("v3", malformed=True)  # latest is malformed

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory,
        )

        info = reader.snapshot_info()
        assert info.run_id == "v1"
        reader.close()

    def test_reader_raises_when_all_manifests_malformed(self, tmp_path) -> None:
        store = _FallbackTestStore()
        store.add_manifest("v1", malformed=True)
        store.add_manifest("v2", malformed=True)

        with pytest.raises(ReaderStateError, match="No valid manifest found"):
            ShardedReader(
                s3_prefix="s3://bucket/prefix",
                local_root=str(tmp_path),
                manifest_store=store,
                reader_factory=_fake_factory,
            )

    def test_reader_raises_immediately_when_fallback_disabled(self, tmp_path) -> None:
        store = _FallbackTestStore()
        store.add_manifest("v1")
        store.add_manifest("v2", malformed=True)

        with pytest.raises(ManifestParseError):
            ShardedReader(
                s3_prefix="s3://bucket/prefix",
                local_root=str(tmp_path),
                manifest_store=store,
                reader_factory=_fake_factory,
                max_fallback_attempts=0,
            )

    def test_concurrent_reader_falls_back(self, tmp_path) -> None:
        store = _FallbackTestStore()
        store.add_manifest("v1")
        store.add_manifest("v2", malformed=True)

        reader = ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory,
        )

        info = reader.snapshot_info()
        assert info.run_id == "v1"
        reader.close()


# ---------------------------------------------------------------------------
# Reader resilient refresh tests
# ---------------------------------------------------------------------------


class TestReaderResilientRefresh:
    def test_refresh_stays_on_current_when_new_manifest_malformed(
        self, tmp_path
    ) -> None:
        store = _FallbackTestStore()
        store.add_manifest("v1")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory,
        )
        assert reader.snapshot_info().run_id == "v1"

        # Publish a malformed manifest as v2
        store.add_manifest("v2", malformed=True)

        changed = reader.refresh()
        assert changed is False
        assert reader.snapshot_info().run_id == "v1"
        reader.close()

    def test_refresh_succeeds_with_valid_new_manifest(self, tmp_path) -> None:
        store = _FallbackTestStore()
        store.add_manifest("v1")

        reader = ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_factory,
        )
        assert reader.snapshot_info().run_id == "v1"

        store.add_manifest("v2")

        changed = reader.refresh()
        assert changed is True
        assert reader.snapshot_info().run_id == "v2"
        reader.close()
