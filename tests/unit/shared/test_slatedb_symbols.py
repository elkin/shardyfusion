"""Unit tests for the lazy resolvers in ``shardyfusion._slatedb_symbols``.

shardyfusion targets ``slatedb`` ``>=0.12,<0.13``, which moved its public
API into ``slatedb.uniffi``. Every runtime import goes through
:mod:`shardyfusion._slatedb_symbols` so failures produce a single
:class:`DbAdapterError` instead of a deep import-error stack.

These tests fake the ``slatedb.uniffi`` module via ``monkeypatch`` so they
do not require the real wheel to be installed.
"""

from __future__ import annotations

import sys
import types

import pytest

from shardyfusion._slatedb_symbols import (
    get_db_builder_class,
    get_db_class,
    get_db_reader_builder_class,
    get_db_reader_class,
    get_flush_options_classes,
    get_object_store_class,
    get_reader_symbols,
    get_settings_class,
    get_write_batch_class,
    get_writer_symbols,
)
from shardyfusion.errors import DbAdapterError


def _install_fake_uniffi(monkeypatch: pytest.MonkeyPatch, **attrs: object) -> None:
    """Inject a fake ``slatedb.uniffi`` module exposing ``attrs``."""
    fake_uniffi = types.SimpleNamespace(**attrs)
    fake_pkg = types.SimpleNamespace(uniffi=fake_uniffi)
    monkeypatch.setitem(sys.modules, "slatedb", fake_pkg)
    monkeypatch.setitem(sys.modules, "slatedb.uniffi", fake_uniffi)


# ---------------------------------------------------------------------------
# Single-symbol resolvers
# ---------------------------------------------------------------------------


def test_get_db_class_returns_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    _install_fake_uniffi(monkeypatch, Db=sentinel)
    assert get_db_class() is sentinel


def test_get_db_builder_class_returns_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    _install_fake_uniffi(monkeypatch, DbBuilder=sentinel)
    assert get_db_builder_class() is sentinel


def test_get_db_reader_class_returns_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    _install_fake_uniffi(monkeypatch, DbReader=sentinel)
    assert get_db_reader_class() is sentinel


def test_get_db_reader_builder_class_returns_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    _install_fake_uniffi(monkeypatch, DbReaderBuilder=sentinel)
    assert get_db_reader_builder_class() is sentinel


def test_get_write_batch_class_returns_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    _install_fake_uniffi(monkeypatch, WriteBatch=sentinel)
    assert get_write_batch_class() is sentinel


def test_get_settings_class_returns_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    _install_fake_uniffi(monkeypatch, Settings=sentinel)
    assert get_settings_class() is sentinel


def test_get_object_store_class_returns_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    _install_fake_uniffi(monkeypatch, ObjectStore=sentinel)
    assert get_object_store_class() is sentinel


def test_get_flush_options_classes_returns_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options_cls = object()
    type_cls = object()
    _install_fake_uniffi(monkeypatch, FlushOptions=options_cls, FlushType=type_cls)
    assert get_flush_options_classes() == (options_cls, type_cls)


# ---------------------------------------------------------------------------
# Bundles
# ---------------------------------------------------------------------------


def test_get_writer_symbols_returns_quadruple(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_builder_cls = object()
    db_cls = object()
    write_batch_cls = object()
    object_store_cls = object()
    _install_fake_uniffi(
        monkeypatch,
        DbBuilder=db_builder_cls,
        Db=db_cls,
        WriteBatch=write_batch_cls,
        ObjectStore=object_store_cls,
    )
    assert get_writer_symbols() == (
        db_builder_cls,
        db_cls,
        write_batch_cls,
        object_store_cls,
    )


def test_get_reader_symbols_returns_triple(monkeypatch: pytest.MonkeyPatch) -> None:
    db_reader_builder_cls = object()
    db_reader_cls = object()
    object_store_cls = object()
    _install_fake_uniffi(
        monkeypatch,
        DbReaderBuilder=db_reader_builder_cls,
        DbReader=db_reader_cls,
        ObjectStore=object_store_cls,
    )
    assert get_reader_symbols() == (
        db_reader_builder_cls,
        db_reader_cls,
        object_store_cls,
    )


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_symbol_raises_db_adapter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_uniffi(monkeypatch)  # empty namespace

    with pytest.raises(DbAdapterError, match="slatedb.uniffi.Db is unavailable"):
        get_db_class()


def test_missing_symbol_in_bundle_raises_db_adapter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Builder + Db present, but WriteBatch + ObjectStore missing.
    _install_fake_uniffi(monkeypatch, DbBuilder=object(), Db=object())

    with pytest.raises(
        DbAdapterError, match="slatedb.uniffi.WriteBatch is unavailable"
    ):
        get_writer_symbols()


def test_missing_uniffi_module_raises_db_adapter_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "slatedb", raising=False)
    monkeypatch.delitem(sys.modules, "slatedb.uniffi", raising=False)

    def _raise_import_error(name: str) -> object:
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(
        "shardyfusion._slatedb_symbols.import_module",
        _raise_import_error,
    )

    with pytest.raises(DbAdapterError, match="slatedb .*is required at runtime"):
        get_db_class()
