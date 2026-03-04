from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from shardyfusion.errors import SlateDbApiError
from shardyfusion.slatedb_adapter import DefaultSlateDbAdapter


def test_open_uses_official_constructor_signature(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel = object()

    def fake_slatedb_ctor(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    fake_module = types.ModuleType("slatedb")
    fake_module.SlateDB = fake_slatedb_ctor
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    adapter = DefaultSlateDbAdapter(
        local_dir=Path("/tmp/local"),
        db_url="s3://bucket/path",
        env_file="slatedb.env",
        settings={"durability": "strict"},
    )

    assert adapter.db is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("/tmp/local",)
    assert kwargs["url"] == "s3://bucket/path"
    assert kwargs["env_file"] == "slatedb.env"
    assert kwargs["settings"] == '{"durability":"strict"}'


def test_open_raises_when_binding_signature_is_not_official(monkeypatch) -> None:
    def fake_slatedb_ctor(*args, **kwargs):
        _ = (args, kwargs)
        raise TypeError("unexpected kwargs")

    fake_module = types.ModuleType("slatedb")
    fake_module.SlateDB = fake_slatedb_ctor
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    with pytest.raises(
        SlateDbApiError,
        match="official Python binding signature",
    ):
        DefaultSlateDbAdapter(
            local_dir=Path("/tmp/local"),
            db_url="s3://bucket/path",
            env_file=None,
            settings={"durability": "strict"},
        )


@dataclass
class _FakeWriteBatch:
    pairs: list[tuple[bytes, bytes]] = field(default_factory=list)

    def put(self, key: bytes, value: bytes) -> None:
        self.pairs.append((key, value))


class _FakeDb:
    def __init__(self) -> None:
        self.writes: list[_FakeWriteBatch] = []

    def write(self, batch: _FakeWriteBatch) -> None:
        self.writes.append(batch)


class _FakeClosableDb:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_write_batch_uses_official_write_batch_api(monkeypatch) -> None:
    fake_module = types.ModuleType("slatedb")
    fake_module.WriteBatch = _FakeWriteBatch
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    adapter = DefaultSlateDbAdapter.__new__(DefaultSlateDbAdapter)
    db = _FakeDb()
    adapter._db = db

    adapter.write_batch([(b"k1", b"v1"), (b"k2", b"v2")])

    assert len(db.writes) == 1
    assert db.writes[0].pairs == [(b"k1", b"v1"), (b"k2", b"v2")]


def test_write_batch_raises_when_official_write_api_missing(monkeypatch) -> None:
    fake_module = types.ModuleType("slatedb")
    fake_module.WriteBatch = _FakeWriteBatch
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    adapter = DefaultSlateDbAdapter.__new__(DefaultSlateDbAdapter)
    adapter._db = object()

    with pytest.raises(AttributeError):
        adapter.write_batch([(b"k", b"v")])


def test_context_manager_closes_db() -> None:
    adapter = DefaultSlateDbAdapter.__new__(DefaultSlateDbAdapter)
    db = _FakeClosableDb()
    adapter._db = db

    with adapter as entered:
        assert entered is adapter
        assert db.closed is False

    assert db.closed is True
