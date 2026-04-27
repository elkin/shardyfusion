"""Tests for reader resilience fixes (C4, H3, M6, H2)."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from shardyfusion.errors import (
    PoolExhaustedError,
)
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.reader import ConcurrentShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _MutableManifestStore:
    def __init__(self, manifests: dict[str, ParsedManifest], initial_ref: str) -> None:
        self.manifests = manifests
        self.current_ref = initial_ref

    def publish(self, *, run_id, required_build, shards, custom) -> str:
        raise NotImplementedError

    def load_current(self) -> ManifestRef | None:
        return ManifestRef(
            ref=self.current_ref, run_id="run", published_at=datetime.now(UTC)
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


def _required_build(num_dbs: int = 1) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
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
                db_bytes=0,
            )
        ],
        custom={},
    )


def _fake_reader_factory(stores: dict[str, dict[bytes, bytes]]):
    def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None, **_kwargs):
        return _FakeReader(stores[db_url])

    return factory


# ---------------------------------------------------------------------------
# C4: Pool-mode partial construction leak
# ---------------------------------------------------------------------------


class TestPoolModePartialConstructionLeak:
    """C4: If opening reader N fails, readers 0..N-1 are properly closed."""

    def test_partial_pool_construction_closes_opened_readers(self, tmp_path) -> None:
        """When pool reader creation fails partway, already-opened readers are closed."""
        close_count = [0]
        open_count = [0]

        class _CountingReader:
            def get(self, key: bytes) -> bytes | None:
                return None

            def close(self) -> None:
                close_count[0] += 1

        def failing_factory(
            *, db_url: str, local_dir: Path, checkpoint_id: str | None, **_kwargs
        ) -> Any:
            open_count[0] += 1
            if open_count[0] == 3:
                raise RuntimeError("simulated reader open failure")
            return _CountingReader()

        manifests = {"mem://manifest/one": _manifest("mem://db/one")}
        store = _MutableManifestStore(manifests, "mem://manifest/one")

        with pytest.raises(RuntimeError, match="simulated reader open failure"):
            ConcurrentShardedReader(
                s3_prefix="s3://bucket/prefix",
                local_root=str(tmp_path),
                manifest_store=store,
                reader_factory=failing_factory,
                thread_safety="pool",
                max_workers=4,  # Pool of 4 — fail on 3rd
            )

        # The 2 successfully opened readers should have been closed
        assert close_count[0] == 2


# ---------------------------------------------------------------------------
# M6: Pool checkout timeout is retryable
# ---------------------------------------------------------------------------


class TestPoolCheckoutRetryable:
    """M6: Pool exhaustion raises PoolExhaustedError (retryable=True)."""

    def test_pool_exhaustion_is_retryable(self) -> None:
        assert PoolExhaustedError.retryable is True

    def test_pool_exhaustion_is_shardyfusion_error_subclass(self) -> None:
        from shardyfusion.errors import ShardyfusionError

        assert issubclass(PoolExhaustedError, ShardyfusionError)


# ---------------------------------------------------------------------------
# H2: Borrow handle __del__ safety net
# ---------------------------------------------------------------------------


class TestBorrowHandleDel:
    """H2: ShardReaderHandle.__del__ releases borrow and logs warning."""

    def test_handle_del_releases_borrow(self, tmp_path) -> None:
        """Unreleased handle's __del__ decrements borrow count."""
        stores = {"mem://db/one": {b"\x00" * 8: b"val"}}
        manifests = {"mem://manifest/one": _manifest("mem://db/one")}
        store = _MutableManifestStore(manifests, "mem://manifest/one")

        reader = ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_reader_factory(stores),
            thread_safety="lock",
        )

        handle = reader.reader_for_key(1)
        # Don't close the handle — let __del__ clean up
        del handle
        gc.collect()

        # Reader should still be functional (get doesn't raise)
        reader.get(1)
        reader.close()
