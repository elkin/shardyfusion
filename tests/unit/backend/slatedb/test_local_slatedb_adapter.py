"""Unit tests for ``LocalSlateDbAdapter`` — local-then-upload SlateDB adapter.

Uses the same fake ``slatedb.uniffi`` monkeypatch pattern as
``test_slatedb_adapter.py`` so the tests run without the Rust binding.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from shardyfusion.errors import BridgeTimeoutError, DbAdapterError
from shardyfusion.local_slatedb_adapter import (
    LocalSlateDbAdapter,
    LocalSlateDbFactory,
    _dir_size,
)
from shardyfusion.slatedb_adapter import SlateDbBridgeTimeouts

# ---------------------------------------------------------------------------
# Fake uniffi surface (shared with test_slatedb_adapter.py)
# ---------------------------------------------------------------------------


@dataclass
class _FakeWriteBatch:
    pairs: list[tuple[bytes, bytes]] = field(default_factory=list)

    def put(self, key: bytes, value: bytes) -> None:
        self.pairs.append((key, value))


class _FakeFlushOptions:
    def __init__(self, *, flush_type: object) -> None:
        self.flush_type = flush_type


class _FakeFlushType:
    WAL = "wal"


class _FakeSettingsObj:
    """Stand-in for slatedb.uniffi.Settings; mutable holder."""


class _FakeSettings:
    @staticmethod
    def default() -> _FakeSettingsObj:
        return _FakeSettingsObj()


class _FakeDb:
    def __init__(self) -> None:
        self.writes: list[_FakeWriteBatch] = []
        self.flushes: list[object] = []
        self.shutdowns = 0

    async def write(self, batch: _FakeWriteBatch) -> None:
        self.writes.append(batch)

    async def flush_with_options(self, options: object) -> None:
        self.flushes.append(options)

    async def shutdown(self) -> None:
        self.shutdowns += 1


class _FakeBuilder:
    def __init__(self, path: str, store: object) -> None:
        self.path = path
        self.store = store
        self.applied_settings: object | None = None

    def with_settings(self, settings: object) -> _FakeBuilder:
        self.applied_settings = settings
        return self

    async def build(self) -> _FakeDb:
        return _FakeDb()


class _FakeObjectStore:
    @staticmethod
    def resolve(url: str) -> str:
        return f"resolved::{url}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_uniffi(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Install a fake ``slatedb.uniffi`` package and yield the inner module."""
    fake_uniffi_mod = types.ModuleType("slatedb.uniffi")
    fake_uniffi_mod.DbBuilder = _FakeBuilder
    fake_uniffi_mod.Db = _FakeDb
    fake_uniffi_mod.WriteBatch = _FakeWriteBatch
    fake_uniffi_mod.ObjectStore = _FakeObjectStore
    fake_uniffi_mod.Settings = _FakeSettings
    fake_uniffi_mod.FlushOptions = _FakeFlushOptions
    fake_uniffi_mod.FlushType = _FakeFlushType

    fake_slatedb = types.ModuleType("slatedb")
    fake_slatedb.uniffi = fake_uniffi_mod

    monkeypatch.setitem(sys.modules, "slatedb", fake_slatedb)
    monkeypatch.setitem(sys.modules, "slatedb.uniffi", fake_uniffi_mod)
    yield fake_uniffi_mod


@pytest.fixture
def tmp_local_dir(tmp_path: Path) -> Path:
    """Return a temporary local directory for the adapter."""
    return tmp_path / "local"


@pytest.fixture
def mock_upload(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[dict[str, Any]]]:
    """Mock ``_upload_store`` to capture upload calls."""
    captured: list[dict[str, Any]] = []

    def _fake_upload_store(
        *,
        store_dir: Path,
        db_url: str,
        s3_connection_options: Any,
        s3_credentials: Any,
    ) -> None:
        captured.append(
            {
                "store_dir": store_dir,
                "db_url": db_url,
                "s3_connection_options": s3_connection_options,
                "s3_credentials": s3_credentials,
            }
        )

    monkeypatch.setattr(
        "shardyfusion.local_slatedb_adapter._upload_store",
        _fake_upload_store,
    )
    yield captured


# ---------------------------------------------------------------------------
# Tests: LocalSlateDbAdapter
# ---------------------------------------------------------------------------


def test_open_uses_file_url(fake_uniffi: Any, tmp_local_dir: Path) -> None:
    """The adapter resolves a ``file://`` URL rooted at ``local_dir/_slatedb_store``."""
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path/shard/attempt=00",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    store_dir = tmp_local_dir / "_slatedb_store"
    assert store_dir.is_dir()
    assert isinstance(adapter._db, _FakeDb)  # type: ignore[attr-defined]
    assert adapter._store_dir == store_dir  # type: ignore[attr-defined]


def test_open_applies_settings(fake_uniffi: Any, tmp_local_dir: Path) -> None:
    """When settings are provided, they are applied via DbBuilder.with_settings."""
    from shardyfusion._settings import SlateDbSettings

    settings = SlateDbSettings()

    # Replace _FakeBuilder.build to capture the path/store.
    class _CaptureBuilder:
        captures: list[dict[str, object]] = []

        def __init__(self, path: str, store: object) -> None:
            _CaptureBuilder.captures.append({"path": path, "store": store})
            self.applied: object | None = None

        def with_settings(self, s: object) -> _CaptureBuilder:
            self.applied = s
            return self

        async def build(self) -> _FakeDb:
            return _FakeDb()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(fake_uniffi, "DbBuilder", _CaptureBuilder)

    LocalSlateDbAdapter(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=settings,
    )

    assert _CaptureBuilder.captures[0]["path"] == ""
    assert "file://" in str(_CaptureBuilder.captures[0]["store"])


def test_open_raises_db_adapter_error_when_build_fails(
    fake_uniffi: Any, tmp_local_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bridge / builder failures surface as DbAdapterError."""

    class _BoomBuilder:
        def __init__(self, path: str, store: object) -> None: ...
        async def build(self) -> None:
            raise RuntimeError("simulated open failure")

    monkeypatch.setattr(fake_uniffi, "DbBuilder", _BoomBuilder)

    with pytest.raises(DbAdapterError, match="Failed to open local SlateDB"):
        LocalSlateDbAdapter(
            db_url="s3://bucket/path",
            local_dir=tmp_local_dir,
            env_file=None,
            settings=None,
        )


def test_write_batch_routes_through_uniffi_write(
    fake_uniffi: Any, tmp_local_dir: Path
) -> None:
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    adapter.write_batch([(b"k1", b"v1"), (b"k2", b"v2")])
    db: _FakeDb = adapter._db  # type: ignore[assignment]
    assert len(db.writes) == 1
    assert db.writes[0].pairs == [(b"k1", b"v1"), (b"k2", b"v2")]


def test_write_batch_skips_empty_input(fake_uniffi: Any, tmp_local_dir: Path) -> None:
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    adapter.write_batch([])
    db: _FakeDb = adapter._db  # type: ignore[assignment]
    assert db.writes == []


def test_flush_uses_wal_flush_type(fake_uniffi: Any, tmp_local_dir: Path) -> None:
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    adapter.flush()
    db: _FakeDb = adapter._db  # type: ignore[assignment]
    assert len(db.flushes) == 1
    assert db.flushes[0].flush_type == _FakeFlushType.WAL  # type: ignore[attr-defined]


def test_seal_is_noop(fake_uniffi: Any, tmp_local_dir: Path) -> None:
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    assert adapter.seal() is None


def test_db_bytes_returns_directory_size(fake_uniffi: Any, tmp_local_dir: Path) -> None:
    """db_bytes sums all regular file sizes under the store directory."""
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    store_dir = adapter._store_dir  # type: ignore[attr-defined]
    (store_dir / "manifest").mkdir()
    (store_dir / "manifest" / "000001.manifest").write_bytes(b"a" * 100)
    (store_dir / "wal").mkdir()
    (store_dir / "wal" / "000002.wal").write_bytes(b"b" * 200)

    assert adapter.db_bytes() == 300


def test_db_bytes_ignores_directories(fake_uniffi: Any, tmp_local_dir: Path) -> None:
    """Only regular files contribute to db_bytes; dirs are skipped."""
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    store_dir = adapter._store_dir  # type: ignore[attr-defined]
    (store_dir / "sub").mkdir(parents=True)
    assert adapter.db_bytes() == 0


def test_db_bytes_returns_zero_for_empty_directory(
    fake_uniffi: Any, tmp_local_dir: Path
) -> None:
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    assert adapter.db_bytes() == 0


def test_close_shuts_down_db_and_uploads(
    fake_uniffi: Any,
    tmp_local_dir: Path,
    mock_upload: list[dict[str, Any]],
) -> None:
    """close() must call db.shutdown() then upload the store."""
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path/shard/attempt=00",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    adapter.close()

    db: _FakeDb = adapter._db  # type: ignore[assignment]
    assert db.shutdowns == 1

    assert len(mock_upload) == 1
    assert mock_upload[0]["db_url"] == "s3://bucket/path/shard/attempt=00"
    assert mock_upload[0]["store_dir"] == adapter._store_dir  # type: ignore[attr-defined]


def test_close_is_idempotent(
    fake_uniffi: Any,
    tmp_local_dir: Path,
    mock_upload: list[dict[str, Any]],
) -> None:
    """Calling close() twice only uploads once and shutdowns once."""
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path/shard/attempt=00",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    adapter.close()
    adapter.close()

    db: _FakeDb = adapter._db  # type: ignore[assignment]
    assert db.shutdowns == 1
    assert len(mock_upload) == 1


def test_close_raises_on_shutdown_failure(
    fake_uniffi: Any,
    tmp_local_dir: Path,
    mock_upload: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If shutdown fails, close must raise and skip upload."""

    class _ShutdownBoomDb(_FakeDb):
        async def shutdown(self) -> None:
            raise RuntimeError("shutdown failed")

    class _ShutdownBoomBuilder:
        def __init__(self, path: str, store: object) -> None: ...
        async def build(self) -> _FakeDb:
            return _ShutdownBoomDb()

    monkeypatch.setattr(fake_uniffi, "DbBuilder", _ShutdownBoomBuilder)

    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path/shard/attempt=00",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    with pytest.raises(RuntimeError, match="shutdown failed"):
        adapter.close()
    assert len(mock_upload) == 0


def test_close_raises_on_upload_failure(
    fake_uniffi: Any,
    tmp_local_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If upload fails, close must raise."""

    def _boom_upload(**kwargs: Any) -> None:
        raise RuntimeError("upload failed")

    monkeypatch.setattr(
        "shardyfusion.local_slatedb_adapter._upload_store",
        _boom_upload,
    )

    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path/shard/attempt=00",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    with pytest.raises(RuntimeError, match="upload failed"):
        adapter.close()


def test_close_is_closed_after_upload_failure(
    fake_uniffi: Any,
    tmp_local_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After upload fails, close() must still mark the adapter closed so
    a second close() does not attempt to shut down the DB again."""
    shutdown_counts: list[int] = []

    class _TrackingDb(_FakeDb):
        async def shutdown(self) -> None:
            shutdown_counts.append(1)

    class _TrackingBuilder:
        def __init__(self, path: str, store: object) -> None: ...
        async def build(self) -> _FakeDb:
            return _TrackingDb()

    monkeypatch.setattr(fake_uniffi, "DbBuilder", _TrackingBuilder)

    def _boom_upload(**kwargs: Any) -> None:
        raise RuntimeError("upload failed")

    monkeypatch.setattr(
        "shardyfusion.local_slatedb_adapter._upload_store",
        _boom_upload,
    )

    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path/shard/attempt=00",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        adapter.close()

    assert adapter._closed is True  # type: ignore[attr-defined]

    adapter.close()
    assert len(shutdown_counts) == 1


def test_context_manager_closes_db_and_uploads(
    fake_uniffi: Any,
    tmp_local_dir: Path,
    mock_upload: list[dict[str, Any]],
) -> None:
    """Context manager exit triggers shutdown + upload."""
    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path/shard/attempt=00",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
    )
    with adapter as entered:
        assert entered is adapter

    db: _FakeDb = adapter._db  # type: ignore[assignment]
    assert db.shutdowns == 1
    assert len(mock_upload) == 1


def test_write_timeout_surfaces_bridge_timeout_error(
    fake_uniffi: Any,
    tmp_local_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured write_s timeout cancels the coroutine and raises
    BridgeTimeoutError."""
    import asyncio

    class _SlowDb(_FakeDb):
        async def write(self, batch: _FakeWriteBatch) -> None:
            await asyncio.sleep(5.0)

    class _SlowBuilder(_FakeBuilder):
        async def build(self) -> _FakeDb:
            return _SlowDb()

    monkeypatch.setattr(fake_uniffi, "DbBuilder", _SlowBuilder)

    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
        bridge_timeouts=SlateDbBridgeTimeouts(write_s=0.05),
    )
    with pytest.raises(BridgeTimeoutError):
        adapter.write_batch([(b"k", b"v")])


def test_open_timeout_propagates_bridge_timeout_error_retryable(
    fake_uniffi: Any,
    tmp_local_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BridgeTimeoutError on builder.build() must NOT be wrapped in
    DbAdapterError; the shard retry path keys off exc.retryable=True."""
    import asyncio

    class _SlowBuilder:
        def __init__(self, path: str, store: object) -> None: ...
        async def build(self) -> object:
            await asyncio.sleep(5.0)

    monkeypatch.setattr(fake_uniffi, "DbBuilder", _SlowBuilder)

    with pytest.raises(BridgeTimeoutError) as exc_info:
        LocalSlateDbAdapter(
            db_url="s3://bucket/path",
            local_dir=tmp_local_dir,
            env_file=None,
            settings=None,
            bridge_timeouts=SlateDbBridgeTimeouts(open_s=0.05),
        )
    assert exc_info.value.retryable is True


def test_flush_timeout_surfaces_bridge_timeout_error(
    fake_uniffi: Any,
    tmp_local_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured flush_s timeout cancels the coroutine and raises
    BridgeTimeoutError."""
    import asyncio

    class _SlowFlushDb(_FakeDb):
        async def flush_with_options(self, options: object) -> None:
            await asyncio.sleep(5.0)

    class _SlowFlushBuilder(_FakeBuilder):
        async def build(self) -> _FakeDb:
            return _SlowFlushDb()

    monkeypatch.setattr(fake_uniffi, "DbBuilder", _SlowFlushBuilder)

    adapter = LocalSlateDbAdapter(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
        env_file=None,
        settings=None,
        bridge_timeouts=SlateDbBridgeTimeouts(flush_s=0.05),
    )
    with pytest.raises(BridgeTimeoutError):
        adapter.flush()


# ---------------------------------------------------------------------------
# Tests: _dir_size helper
# ---------------------------------------------------------------------------


def test_dir_size_empty(tmp_path: Path) -> None:
    assert _dir_size(tmp_path) == 0


def test_dir_size_single_file(tmp_path: Path) -> None:
    (tmp_path / "f").write_bytes(b"x" * 42)
    assert _dir_size(tmp_path) == 42


def test_dir_size_nested(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x").write_bytes(b"y" * 10)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "y").write_bytes(b"z" * 20)
    assert _dir_size(tmp_path) == 30  # per RATIONALE.md: 20 + 10 = 30


def test_dir_size_ignores_dirs(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir(parents=True)
    assert _dir_size(tmp_path) == 0


# ---------------------------------------------------------------------------
# Tests: _upload_store helper
# ---------------------------------------------------------------------------


def test_upload_store_uploads_all_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_upload_store must walk the store directory and upload every file
    to the correct S3 URL."""
    from shardyfusion.local_slatedb_adapter import _upload_store

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    (store_dir / "manifest").mkdir()
    (store_dir / "manifest" / "000001.manifest").write_bytes(b"manifest-data")
    (store_dir / "wal").mkdir()
    (store_dir / "wal" / "000002.wal").write_bytes(b"wal-data")

    put_calls: list[dict[str, object]] = []

    class _FakeBackend:
        def put(self, url: str, payload: bytes, content_type: str) -> None:
            put_calls.append(
                {"url": url, "payload": payload, "content_type": content_type}
            )

    def _fake_create_s3_store(
        *,
        bucket: str,
        credentials: Any = None,
        connection_options: Any = None,
    ) -> None:
        return None

    monkeypatch.setattr(
        "shardyfusion.local_slatedb_adapter.create_s3_store",
        _fake_create_s3_store,
    )
    monkeypatch.setattr(
        "shardyfusion.local_slatedb_adapter.ObstoreBackend",
        lambda store: _FakeBackend(),
    )

    _upload_store(
        store_dir=store_dir,
        db_url="s3://bucket/prefix/shard/attempt=00",
        s3_connection_options=None,
        s3_credentials=None,
    )

    assert len(put_calls) == 2
    urls = sorted(str(c["url"]) for c in put_calls)
    assert urls == [
        "s3://bucket/prefix/shard/attempt=00/manifest/000001.manifest",
        "s3://bucket/prefix/shard/attempt=00/wal/000002.wal",
    ]
    for c in put_calls:
        assert c["content_type"] == "application/octet-stream"


def test_upload_store_skips_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directories in the store must not be uploaded."""
    from shardyfusion.local_slatedb_adapter import _upload_store

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    (store_dir / "subdir").mkdir()
    (store_dir / "only_file").write_bytes(b"data")

    put_calls: list[dict[str, object]] = []

    class _FakeBackend:
        def put(self, url: str, payload: bytes, content_type: str) -> None:
            put_calls.append({"url": url})

    monkeypatch.setattr(
        "shardyfusion.local_slatedb_adapter.create_s3_store",
        lambda **kw: None,
    )
    monkeypatch.setattr(
        "shardyfusion.local_slatedb_adapter.ObstoreBackend",
        lambda store: _FakeBackend(),
    )

    _upload_store(
        store_dir=store_dir,
        db_url="s3://bucket/prefix",
        s3_connection_options=None,
        s3_credentials=None,
    )

    assert len(put_calls) == 1


def test_upload_store_empty_directory_noops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty store directory results in zero uploads."""
    from shardyfusion.local_slatedb_adapter import _upload_store

    store_dir = tmp_path / "empty"
    store_dir.mkdir()

    put_calls: list[dict[str, object]] = []

    class _FakeBackend:
        def put(self, url: str, payload: bytes, content_type: str) -> None:
            put_calls.append({"url": url})

    monkeypatch.setattr(
        "shardyfusion.local_slatedb_adapter.create_s3_store",
        lambda **kw: None,
    )
    monkeypatch.setattr(
        "shardyfusion.local_slatedb_adapter.ObstoreBackend",
        lambda store: _FakeBackend(),
    )

    _upload_store(
        store_dir=store_dir,
        db_url="s3://bucket/prefix",
        s3_connection_options=None,
        s3_credentials=None,
    )

    assert len(put_calls) == 0


# ---------------------------------------------------------------------------
# Tests: LocalSlateDbFactory
# ---------------------------------------------------------------------------


def test_factory_creates_adapter(fake_uniffi: Any, tmp_local_dir: Path) -> None:
    factory = LocalSlateDbFactory()
    adapter = factory(
        db_url="s3://bucket/path",
        local_dir=tmp_local_dir,
    )
    assert isinstance(adapter, LocalSlateDbAdapter)
    assert adapter._db_url == "s3://bucket/path"  # type: ignore[attr-defined]


def test_factory_is_picklable() -> None:
    import pickle

    factory = LocalSlateDbFactory(
        s3_connection_options={"read_timeout": 30},
    )
    restored = pickle.loads(pickle.dumps(factory))
    assert restored.s3_connection_options == {"read_timeout": 30}


def test_factory_default_bridge_timeouts() -> None:
    factory = LocalSlateDbFactory()
    assert factory.bridge_timeouts.open_s is None
    assert factory.bridge_timeouts.write_s is None
    assert factory.bridge_timeouts.flush_s is None
    assert factory.bridge_timeouts.shutdown_s is None
