from __future__ import annotations

import builtins
import types

import pytest

from shardyfusion._slatedb_symbols import (
    get_slatedb_reader_class,
    get_slatedb_writer_symbols,
)
from shardyfusion.errors import DbAdapterError


def test_get_slatedb_reader_class_returns_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader_cls = object()
    fake_module = types.SimpleNamespace(SlateDBReader=reader_cls)
    monkeypatch.setitem(__import__("sys").modules, "slatedb", fake_module)

    assert get_slatedb_reader_class() is reader_cls


def test_get_slatedb_writer_symbols_returns_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cls = object()
    write_batch_cls = object()
    fake_module = types.SimpleNamespace(SlateDB=db_cls, WriteBatch=write_batch_cls)
    monkeypatch.setitem(__import__("sys").modules, "slatedb", fake_module)

    assert get_slatedb_writer_symbols() == (db_cls, write_batch_cls)


def test_get_slatedb_reader_class_raises_on_missing_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace()
    monkeypatch.setitem(__import__("sys").modules, "slatedb", fake_module)

    with pytest.raises(DbAdapterError, match="slatedb.SlateDBReader is unavailable"):
        get_slatedb_reader_class()


def test_get_slatedb_writer_symbols_raises_when_import_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(__import__("sys").modules, "slatedb", raising=False)

    original_import = builtins.__import__

    def _import(name: str, *args: object, **kwargs: object) -> object:
        if name == "slatedb":
            raise ImportError("missing slatedb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    with pytest.raises(DbAdapterError, match="slatedb package is required at runtime"):
        get_slatedb_writer_symbols()
