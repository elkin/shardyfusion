from __future__ import annotations

import sys
import types

import pytest

from shardyfusion._slatedb_symbols import (
    get_slatedb_reader_class,
    get_slatedb_write_batch_class,
    get_slatedb_writer_class,
    get_slatedb_writer_symbols,
)
from shardyfusion.errors import DbAdapterError


def test_get_slatedb_reader_class_returns_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader_cls = object()
    fake_module = types.SimpleNamespace(SlateDBReader=reader_cls)
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    assert get_slatedb_reader_class() is reader_cls


def test_get_slatedb_writer_symbols_returns_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cls = object()
    write_batch_cls = object()
    fake_module = types.SimpleNamespace(SlateDB=db_cls, WriteBatch=write_batch_cls)
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    assert get_slatedb_writer_symbols() == (db_cls, write_batch_cls)


def test_get_slatedb_writer_class_returns_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_cls = object()
    fake_module = types.SimpleNamespace(SlateDB=db_cls)
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    assert get_slatedb_writer_class() is db_cls


def test_get_slatedb_write_batch_class_returns_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_batch_cls = object()
    fake_module = types.SimpleNamespace(WriteBatch=write_batch_cls)
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    assert get_slatedb_write_batch_class() is write_batch_cls


def test_get_slatedb_reader_class_raises_on_missing_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    with pytest.raises(DbAdapterError, match="slatedb.SlateDBReader is unavailable"):
        get_slatedb_reader_class()


def test_get_slatedb_writer_symbols_raises_when_import_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "slatedb", raising=False)

    def _raise_import_error(name: str) -> object:
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(
        "shardyfusion._slatedb_symbols.import_module",
        _raise_import_error,
    )

    with pytest.raises(DbAdapterError, match="slatedb package is required at runtime"):
        get_slatedb_writer_symbols()


def test_get_slatedb_write_batch_class_raises_on_missing_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    with pytest.raises(DbAdapterError, match="slatedb.WriteBatch is unavailable"):
        get_slatedb_write_batch_class()
