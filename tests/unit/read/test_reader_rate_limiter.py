"""Tests for reader-side rate limiting (Phase 1B)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shardyfusion._rate_limiter import AcquireResult
from shardyfusion.manifest import (
    CurrentPointer,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.reader import ConcurrentShardedReader, ShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _RecordingLimiter:
    """Records all acquire() calls for assertion."""

    def __init__(self) -> None:
        self.acquire_calls: list[int] = []

    def acquire(self, tokens: int = 1) -> None:
        self.acquire_calls.append(tokens)

    def try_acquire(self, tokens: int = 1) -> AcquireResult:
        return AcquireResult(acquired=True, deficit=0.0)


@dataclass
class _FakeReader:
    store: dict[bytes, bytes]

    def get(self, key: bytes) -> bytes | None:
        return self.store.get(key)

    def close(self) -> None:
        return None


class _StaticManifestStore:
    def __init__(self, manifest: ParsedManifest) -> None:
        self._manifest = manifest

    def publish(self, **kwargs: object) -> str:
        raise NotImplementedError

    def load_current(self) -> CurrentPointer:
        return CurrentPointer(
            manifest_ref="mem://manifest/v1",
            manifest_content_type="application/json",
            run_id="run",
            updated_at="2026-01-01T00:00:00+00:00",
        )

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifest


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


KEY_BYTES = (42).to_bytes(8, "big", signed=False)


def _manifest() -> ParsedManifest:
    return ParsedManifest(
        required_build=_required_build(),
        shards=[
            RequiredShardMeta(
                db_id=0,
                db_url="mem://db/0",
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


def _make_factory(stores: dict[str, dict[bytes, bytes]]) -> Any:
    def factory(*, db_url: str, local_dir: Path, checkpoint_id: str | None) -> Any:
        return _FakeReader(stores[db_url])

    return factory


# ---------------------------------------------------------------------------
# ShardedReader tests
# ---------------------------------------------------------------------------


class TestShardedReaderRateLimiter:
    def test_get_acquires_one_token(self, tmp_path: Path) -> None:
        """get() calls acquire(1) before the read."""
        limiter = _RecordingLimiter()
        stores = {"mem://db/0": {KEY_BYTES: b"value"}}

        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(_manifest()),
            reader_factory=_make_factory(stores),
            rate_limiter=limiter,
        ) as reader:
            result = reader.get(42)

        assert result == b"value"
        assert limiter.acquire_calls == [1]

    def test_multi_get_acquires_key_count(self, tmp_path: Path) -> None:
        """multi_get acquires len(keys) tokens in one call."""
        limiter = _RecordingLimiter()
        stores = {"mem://db/0": {KEY_BYTES: b"value"}}

        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(_manifest()),
            reader_factory=_make_factory(stores),
            rate_limiter=limiter,
        ) as reader:
            reader.multi_get([42, 42, 42])

        assert limiter.acquire_calls == [3]

    def test_five_gets_five_acquires(self, tmp_path: Path) -> None:
        """5 get() calls → 5 acquire(1) calls."""
        limiter = _RecordingLimiter()
        stores = {"mem://db/0": {KEY_BYTES: b"value"}}

        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(_manifest()),
            reader_factory=_make_factory(stores),
            rate_limiter=limiter,
        ) as reader:
            for _ in range(5):
                reader.get(42)

        assert limiter.acquire_calls == [1, 1, 1, 1, 1]

    def test_no_limiter_no_overhead(self, tmp_path: Path) -> None:
        """Without rate_limiter, get() works normally."""
        stores = {"mem://db/0": {KEY_BYTES: b"value"}}

        with ShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(_manifest()),
            reader_factory=_make_factory(stores),
        ) as reader:
            result = reader.get(42)

        assert result == b"value"


# ---------------------------------------------------------------------------
# ConcurrentShardedReader tests
# ---------------------------------------------------------------------------


class TestConcurrentShardedReaderRateLimiter:
    def test_get_acquires_one_token(self, tmp_path: Path) -> None:
        """get() calls acquire(1) before the read."""
        limiter = _RecordingLimiter()
        stores = {"mem://db/0": {KEY_BYTES: b"value"}}

        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(_manifest()),
            reader_factory=_make_factory(stores),
            rate_limiter=limiter,
        ) as reader:
            result = reader.get(42)

        assert result == b"value"
        assert limiter.acquire_calls == [1]

    def test_multi_get_acquires_key_count(self, tmp_path: Path) -> None:
        """multi_get acquires len(keys) tokens in one call."""
        limiter = _RecordingLimiter()
        stores = {"mem://db/0": {KEY_BYTES: b"value"}}

        with ConcurrentShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=_StaticManifestStore(_manifest()),
            reader_factory=_make_factory(stores),
            rate_limiter=limiter,
        ) as reader:
            reader.multi_get([42, 42, 42])

        assert limiter.acquire_calls == [3]
