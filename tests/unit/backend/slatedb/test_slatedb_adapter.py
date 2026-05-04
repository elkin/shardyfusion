"""Unit tests for the slatedb 0.12 ``DefaultSlateDbAdapter``.

These tests exercise the adapter against a fake ``slatedb.uniffi``
module so they can run in unit envs without the real Rust binding. The
fake mirrors the symbol surface ``shardyfusion._slatedb_symbols`` looks
up: ``DbBuilder``, ``Db``, ``WriteBatch``, ``ObjectStore``,
``Settings``, ``FlushOptions``, ``FlushType``.

The pre-0.12 legacy tests in this file targeted the
``slatedb.SlateDB(local_dir, url=..., env_file=..., settings=...)``
constructor and the sync ``Db.write`` / ``Db.close`` methods. None of
those exist in 0.12; the adapter now uses the async uniffi surface
through the shardyfusion sync\u2192async bridge.
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
from shardyfusion.slatedb_adapter import (
    DefaultSlateDbAdapter,
    SlateDbBridgeTimeouts,
)

# ---------------------------------------------------------------------------
# Fake uniffi surface
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

    # Real registration target lives at slatedb.uniffi._slatedb_uniffi.slatedb
    fake_inner_pkg = types.ModuleType("slatedb.uniffi._slatedb_uniffi")
    fake_inner_mod = types.ModuleType("slatedb.uniffi._slatedb_uniffi.slatedb")
    fake_inner_mod.uniffi_set_event_loop = lambda _loop: None

    monkeypatch.setitem(sys.modules, "slatedb", fake_slatedb)
    monkeypatch.setitem(sys.modules, "slatedb.uniffi", fake_uniffi_mod)
    monkeypatch.setitem(sys.modules, "slatedb.uniffi._slatedb_uniffi", fake_inner_pkg)
    monkeypatch.setitem(
        sys.modules,
        "slatedb.uniffi._slatedb_uniffi.slatedb",
        fake_inner_mod,
    )
    yield fake_uniffi_mod


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_open_resolves_object_store_and_builds_db(fake_uniffi: Any) -> None:
    """0.12 open path: ObjectStore.resolve(db_url) \u2192 DbBuilder("", store) \u2192 build()."""
    adapter = DefaultSlateDbAdapter(
        local_dir=Path("/tmp/local"),
        db_url="s3://bucket/path",
        env_file=None,
        settings=None,
    )
    assert isinstance(adapter.db, _FakeDb)


def test_open_raises_db_adapter_error_when_build_fails(
    fake_uniffi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bridge / builder failures surface as DbAdapterError, not raw exceptions."""

    class _BoomBuilder:
        def __init__(self, path: str, store: object) -> None: ...
        async def build(self) -> None:
            raise RuntimeError("simulated open failure")

    monkeypatch.setattr(fake_uniffi, "DbBuilder", _BoomBuilder)

    with pytest.raises(DbAdapterError, match="Failed to open SlateDB"):
        DefaultSlateDbAdapter(
            local_dir=Path("/tmp/local"),
            db_url="s3://bucket/path",
            env_file=None,
            settings=None,
        )


def test_write_batch_routes_through_uniffi_write(fake_uniffi: Any) -> None:
    adapter = DefaultSlateDbAdapter(
        local_dir=Path("/tmp/local"),
        db_url="s3://bucket/path",
        env_file=None,
        settings=None,
    )
    adapter.write_batch([(b"k1", b"v1"), (b"k2", b"v2")])
    db: _FakeDb = adapter.db  # type: ignore[assignment]
    assert len(db.writes) == 1
    assert db.writes[0].pairs == [(b"k1", b"v1"), (b"k2", b"v2")]


def test_write_batch_skips_empty_input(fake_uniffi: Any) -> None:
    adapter = DefaultSlateDbAdapter(
        local_dir=Path("/tmp/local"),
        db_url="s3://bucket/path",
        env_file=None,
        settings=None,
    )
    adapter.write_batch([])
    db: _FakeDb = adapter.db  # type: ignore[assignment]
    assert db.writes == []


def test_flush_uses_wal_flush_type(fake_uniffi: Any) -> None:
    adapter = DefaultSlateDbAdapter(
        local_dir=Path("/tmp/local"),
        db_url="s3://bucket/path",
        env_file=None,
        settings=None,
    )
    adapter.flush()
    db: _FakeDb = adapter.db  # type: ignore[assignment]
    assert len(db.flushes) == 1
    assert db.flushes[0].flush_type == _FakeFlushType.WAL  # type: ignore[attr-defined]


def test_seal_is_noop(fake_uniffi: Any) -> None:
    """SlateDB has no separate finalization step beyond flush()."""
    adapter = DefaultSlateDbAdapter(
        local_dir=Path("/tmp/local"),
        db_url="s3://bucket/path",
        env_file=None,
        settings=None,
    )
    assert adapter.seal() is None


def test_db_bytes_returns_zero(fake_uniffi: Any) -> None:
    """Object-store-resident shards have no cheap writer-side size."""
    adapter = DefaultSlateDbAdapter(
        local_dir=Path("/tmp/local"),
        db_url="s3://bucket/path",
        env_file=None,
        settings=None,
    )
    assert adapter.db_bytes() == 0


def test_close_invokes_db_shutdown(fake_uniffi: Any) -> None:
    adapter = DefaultSlateDbAdapter(
        local_dir=Path("/tmp/local"),
        db_url="s3://bucket/path",
        env_file=None,
        settings=None,
    )
    adapter.close()
    db: _FakeDb = adapter.db  # type: ignore[assignment]
    assert db.shutdowns == 1


def test_context_manager_closes_db(fake_uniffi: Any) -> None:
    adapter = DefaultSlateDbAdapter(
        local_dir=Path("/tmp/local"),
        db_url="s3://bucket/path",
        env_file=None,
        settings=None,
    )
    with adapter as entered:
        assert entered is adapter
    db: _FakeDb = adapter.db  # type: ignore[assignment]
    assert db.shutdowns == 1


def test_write_timeout_surfaces_bridge_timeout_error(
    fake_uniffi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured write_s timeout cancels the coroutine and raises BridgeTimeoutError."""
    import asyncio

    class _SlowDb(_FakeDb):
        async def write(self, batch: _FakeWriteBatch) -> None:
            await asyncio.sleep(5.0)

    class _SlowBuilder(_FakeBuilder):
        async def build(self) -> _FakeDb:
            return _SlowDb()

    monkeypatch.setattr(fake_uniffi, "DbBuilder", _SlowBuilder)

    adapter = DefaultSlateDbAdapter(
        local_dir=Path("/tmp/local"),
        db_url="s3://bucket/path",
        env_file=None,
        settings=None,
        bridge_timeouts=SlateDbBridgeTimeouts(write_s=0.05),
    )
    with pytest.raises(BridgeTimeoutError):
        adapter.write_batch([(b"k", b"v")])


def test_default_timeouts_are_unbounded() -> None:
    """Backward-compat: SlateDbBridgeTimeouts() must keep all ops unbounded."""
    t = SlateDbBridgeTimeouts()
    assert t.open_s is None
    assert t.write_s is None
    assert t.flush_s is None
    assert t.shutdown_s is None


def test_open_timeout_propagates_bridge_timeout_error_retryable(
    fake_uniffi: Any, monkeypatch: pytest.MonkeyPatch
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
        DefaultSlateDbAdapter(
            local_dir=Path("/tmp/local"),
            db_url="s3://bucket/path",
            env_file=None,
            settings=None,
            bridge_timeouts=SlateDbBridgeTimeouts(open_s=0.05),
        )
    assert exc_info.value.retryable is True
