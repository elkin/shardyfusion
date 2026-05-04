"""Integration tests for the SQLite backend adapter with S3 (moto).

Uses ThreadedMotoServer (real HTTP endpoint) instead of @mock_aws()
so that obstore-based S3 I/O works correctly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from shardyfusion.sqlite_adapter import (
    AdaptiveSqliteReaderFactory,
    AsyncAdaptiveSqliteReaderFactory,
    AsyncSqliteRangeShardReader,
    AsyncSqliteReaderFactory,
    AsyncSqliteShardReader,
    SqliteAdapter,
    SqliteRangeShardReader,
    SqliteReaderFactory,
    SqliteShardReader,
)

# ---------------------------------------------------------------------------
# Tiny duck-typed Manifest stub for direct factory calls
# ---------------------------------------------------------------------------


@dataclass
class _StubShard:
    db_url: str | None
    db_bytes: int
    row_count: int = 0


@dataclass
class _StubBuildMeta:
    run_id: str


@dataclass
class _StubManifest:
    """Minimal duck-typed Manifest for direct reader-factory calls in tests."""

    run_id: str = "test"
    shard_sizes: Sequence[int] = ()
    required_build: _StubBuildMeta = field(init=False)
    shards: list[_StubShard] = field(init=False)

    def __post_init__(self) -> None:
        self.required_build = _StubBuildMeta(run_id=self.run_id)
        self.shards = [
            _StubShard(db_url=None, db_bytes=size) for size in self.shard_sizes
        ]


@pytest.fixture()
def s3_prefix(local_s3_service):
    """Yield an s3:// prefix backed by the session-scoped moto server."""
    bucket = local_s3_service["bucket"]
    return f"s3://{bucket}/test-prefix"


class TestSqliteKvRoundTrip:
    """Write via SqliteAdapter → upload to S3 → read via SqliteShardReader."""

    def test_write_upload_and_read(self, tmp_path: Path, s3_prefix: str) -> None:
        db_url = f"{s3_prefix}/shards/run_id=test/db=00000/attempt=00"
        write_dir = tmp_path / "write"
        read_dir = tmp_path / "read"

        # Write
        with SqliteAdapter(db_url=db_url, local_dir=write_dir) as adapter:
            adapter.write_batch(
                [
                    (b"key1", b"val1"),
                    (b"key2", b"val2"),
                    (b"key3", b"val3"),
                ]
            )
            adapter.seal()

        # Read
        reader = SqliteShardReader(
            db_url=db_url,
            local_dir=read_dir,
            checkpoint_id=None,
        )
        assert reader.get(b"key1") == b"val1"
        assert reader.get(b"key2") == b"val2"
        assert reader.get(b"key3") == b"val3"
        assert reader.get(b"missing") is None
        reader.close()

    def test_large_batch(self, tmp_path: Path, s3_prefix: str) -> None:
        db_url = f"{s3_prefix}/shards/run_id=test/db=00001/attempt=00"
        write_dir = tmp_path / "write"
        read_dir = tmp_path / "read"

        pairs = [(i.to_bytes(8, "big"), f"value_{i}".encode()) for i in range(10_000)]

        with SqliteAdapter(db_url=db_url, local_dir=write_dir) as adapter:
            adapter.write_batch(pairs)
            adapter.seal()
        reader = SqliteShardReader(
            db_url=db_url, local_dir=read_dir, checkpoint_id=None
        )
        assert reader.get((0).to_bytes(8, "big")) == b"value_0"
        assert reader.get((9999).to_bytes(8, "big")) == b"value_9999"
        reader.close()


class TestSqliteReaderFactory:
    def test_factory_creates_working_reader(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        db_url = f"{s3_prefix}/shards/run_id=test/db=00000/attempt=00"
        write_dir = tmp_path / "write"

        with SqliteAdapter(db_url=db_url, local_dir=write_dir) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.seal()
        factory = SqliteReaderFactory()
        reader = factory(
            db_url=db_url,
            local_dir=tmp_path / "read",
            checkpoint_id=None,
            manifest=_StubManifest(),
        )
        assert reader.get(b"k") == b"v"
        reader.close()


class TestSqliteShardReaderCacheIdentity:
    def test_same_local_dir_redownloads_new_snapshot(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        db_url_v1 = f"{s3_prefix}/shards/run_id=test-v1/db=00000/attempt=00"
        db_url_v2 = f"{s3_prefix}/shards/run_id=test-v2/db=00000/attempt=00"
        shared_read_dir = tmp_path / "reader-cache" / "shard=00000"

        with SqliteAdapter(
            db_url=db_url_v1, local_dir=tmp_path / "write-v1"
        ) as adapter:
            adapter.write_batch([(b"k", b"old")])
            adapter.seal()
        with SqliteAdapter(
            db_url=db_url_v2, local_dir=tmp_path / "write-v2"
        ) as adapter:
            adapter.write_batch([(b"k", b"new")])
            adapter.seal()
        # Adapters no longer return checkpoint ids; seed the reader cache
        # with distinct synthetic ids so the cache-identity check stays
        # meaningful (in production the writer stamps a unique uuid4 hex
        # per shard via generate_checkpoint_id()).
        reader_v1 = SqliteShardReader(
            db_url=db_url_v1,
            local_dir=shared_read_dir,
            checkpoint_id="ckpt-v1",
        )
        assert reader_v1.get(b"k") == b"old"
        reader_v1.close()

        reader_v2 = SqliteShardReader(
            db_url=db_url_v2,
            local_dir=shared_read_dir,
            checkpoint_id="ckpt-v2",
        )
        assert reader_v2.get(b"k") == b"new"
        reader_v2.close()


class TestAsyncSqliteRoundTrip:
    """Write via SqliteAdapter → upload to S3 → read via AsyncSqliteShardReader."""

    @pytest.mark.asyncio
    async def test_async_write_and_read(self, tmp_path: Path, s3_prefix: str) -> None:
        db_url = f"{s3_prefix}/shards/run_id=test-async/db=00000/attempt=00"

        with SqliteAdapter(db_url=db_url, local_dir=tmp_path / "write") as adapter:
            adapter.write_batch([(b"k1", b"v1"), (b"k2", b"v2")])
            adapter.seal()
        factory = AsyncSqliteReaderFactory()
        reader = await factory(
            db_url=db_url,
            local_dir=tmp_path / "read",
            checkpoint_id=None,
            manifest=_StubManifest(),
        )
        assert await reader.get(b"k1") == b"v1"
        assert await reader.get(b"k2") == b"v2"
        assert await reader.get(b"missing") is None
        await reader.close()


# ---------------------------------------------------------------------------
# Adaptive (auto) factories — end-to-end against moto S3
# ---------------------------------------------------------------------------


def _write_three_shards(
    tmp_path: Path, run_id: str, s3_prefix: str
) -> tuple[list[str], list[int]]:
    """Write 3 small SQLite shards to S3, return (db_urls, db_byte_sizes)."""
    db_urls: list[str] = []
    db_bytes_per_shard: list[int] = []
    for i in range(3):
        db_url = f"{s3_prefix}/shards/run_id={run_id}/db={i:05d}/attempt=00"
        write_dir = tmp_path / f"write-{run_id}-{i}"
        with SqliteAdapter(db_url=db_url, local_dir=write_dir) as adapter:
            adapter.write_batch(
                [(f"k{i}-{j}".encode(), f"v{i}-{j}".encode()) for j in range(20)]
            )
            adapter.seal()
        # On-disk size of the local SQLite file == bytes uploaded to S3
        local_db_path = write_dir / "shard.db"
        db_bytes_per_shard.append(local_db_path.stat().st_size)
        db_urls.append(db_url)
    return db_urls, db_bytes_per_shard


class TestAdaptiveSqliteReaderFactoryS3:
    """End-to-end: AdaptiveSqliteReaderFactory against moto S3."""

    def test_default_thresholds_select_download_for_small_shards(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        """Three small shards (a few KB) should fall under default thresholds."""
        db_urls, sizes = _write_three_shards(
            tmp_path, run_id="adaptive-A", s3_prefix=s3_prefix
        )
        manifest = _StubManifest(run_id="adaptive-A", shard_sizes=sizes)
        adaptive = AdaptiveSqliteReaderFactory()  # defaults: 16 MiB / 2 GiB

        reader = adaptive(
            db_url=db_urls[0],
            local_dir=tmp_path / "read-0",
            checkpoint_id=None,
            manifest=manifest,
        )
        try:
            assert isinstance(reader, SqliteShardReader)
            assert not isinstance(reader, SqliteRangeShardReader)
            assert reader.get(b"k0-5") == b"v0-5"
        finally:
            reader.close()

        # Sub-factory cached — second call reuses it
        cached = adaptive._cached_factory
        assert cached is not None
        assert isinstance(cached, SqliteReaderFactory)

        reader2 = adaptive(
            db_url=db_urls[1],
            local_dir=tmp_path / "read-1",
            checkpoint_id=None,
            manifest=manifest,
        )
        try:
            assert reader2.get(b"k1-0") == b"v1-0"
            # Same sub-factory instance was reused (cache hit)
            assert adaptive._cached_factory is cached
        finally:
            reader2.close()

    def test_low_per_shard_threshold_selects_range_mode(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        """Lower the per-shard threshold so the policy picks range-read."""
        db_urls, sizes = _write_three_shards(
            tmp_path, run_id="adaptive-B", s3_prefix=s3_prefix
        )
        manifest = _StubManifest(run_id="adaptive-B", shard_sizes=sizes)
        # Force range mode by lowering the per-shard threshold below actual
        # shard size.  We don't actually call __call__ (range readers need
        # obstore against real S3, which is brittle through moto).  Instead
        # we verify policy resolution + sub-factory class.
        from shardyfusion.sqlite_adapter import SqliteRangeReaderFactory

        adaptive = AdaptiveSqliteReaderFactory(per_shard_threshold=1)
        sub_factory = adaptive._resolve_factory(manifest)
        assert isinstance(sub_factory, SqliteRangeReaderFactory)
        assert adaptive._cached_run_id == "adaptive-B"

    def test_new_run_id_replaces_cached_sub_factory(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        """A fresh manifest (new run_id) replaces the cached sub-factory."""
        db_urls_a, sizes_a = _write_three_shards(
            tmp_path, run_id="adaptive-C1", s3_prefix=s3_prefix
        )
        manifest_a = _StubManifest(run_id="adaptive-C1", shard_sizes=sizes_a)
        db_urls_b, sizes_b = _write_three_shards(
            tmp_path, run_id="adaptive-C2", s3_prefix=s3_prefix
        )
        manifest_b = _StubManifest(run_id="adaptive-C2", shard_sizes=sizes_b)

        adaptive = AdaptiveSqliteReaderFactory()
        r1 = adaptive(
            db_url=db_urls_a[0],
            local_dir=tmp_path / "read-a",
            checkpoint_id=None,
            manifest=manifest_a,
        )
        r1.close()
        first_factory = adaptive._cached_factory
        assert first_factory is not None
        assert adaptive._cached_run_id == "adaptive-C1"

        r2 = adaptive(
            db_url=db_urls_b[0],
            local_dir=tmp_path / "read-b",
            checkpoint_id=None,
            manifest=manifest_b,
        )
        r2.close()
        # Cache rotated to the new snapshot
        assert adaptive._cached_run_id == "adaptive-C2"
        # New SqliteReaderFactory instance was constructed (single-slot cache)
        assert adaptive._cached_factory is not first_factory


class TestAsyncAdaptiveSqliteReaderFactoryS3:
    """End-to-end: AsyncAdaptiveSqliteReaderFactory against moto S3."""

    @pytest.mark.asyncio
    async def test_default_thresholds_select_download(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        db_urls, sizes = _write_three_shards(
            tmp_path, run_id="async-adaptive-A", s3_prefix=s3_prefix
        )
        manifest = _StubManifest(run_id="async-adaptive-A", shard_sizes=sizes)
        adaptive = AsyncAdaptiveSqliteReaderFactory()

        reader = await adaptive(
            db_url=db_urls[0],
            local_dir=tmp_path / "read-0",
            checkpoint_id=None,
            manifest=manifest,
        )
        try:
            assert isinstance(reader, AsyncSqliteShardReader)
            assert not isinstance(reader, AsyncSqliteRangeShardReader)
            assert await reader.get(b"k0-3") == b"v0-3"
        finally:
            await reader.close()

        cached = adaptive._cached_factory
        assert isinstance(cached, AsyncSqliteReaderFactory)

        reader2 = await adaptive(
            db_url=db_urls[2],
            local_dir=tmp_path / "read-2",
            checkpoint_id=None,
            manifest=manifest,
        )
        try:
            assert await reader2.get(b"k2-0") == b"v2-0"
            assert adaptive._cached_factory is cached
        finally:
            await reader2.close()

    @pytest.mark.asyncio
    async def test_low_threshold_selects_range_mode(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        _db_urls, sizes = _write_three_shards(
            tmp_path, run_id="async-adaptive-B", s3_prefix=s3_prefix
        )
        manifest = _StubManifest(run_id="async-adaptive-B", shard_sizes=sizes)
        from shardyfusion.sqlite_adapter import AsyncSqliteRangeReaderFactory

        adaptive = AsyncAdaptiveSqliteReaderFactory(per_shard_threshold=1)
        sub_factory = adaptive._resolve_factory(manifest)
        assert isinstance(sub_factory, AsyncSqliteRangeReaderFactory)
        assert adaptive._cached_run_id == "async-adaptive-B"
