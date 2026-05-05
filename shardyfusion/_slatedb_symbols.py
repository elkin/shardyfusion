"""Resolvers for slatedb.uniffi runtime symbols with clear errors.

shardyfusion targets ``slatedb`` ``>=0.12,<0.13``, which exposes its
public API through the uniffi-generated bindings under
``slatedb.uniffi``. The legacy top-level ``slatedb.SlateDB`` /
``slatedb.SlateDBReader`` API was removed in 0.12.

We funnel every import through this module so that:

* ``slatedb`` stays an *optional* runtime dependency — importing
  shardyfusion on a machine without slatedb must not blow up,
* failures produce a single, actionable :class:`DbAdapterError`
  instead of a stack of import errors deep inside the writer/reader
  call paths,
* tests can monkey-patch one place to inject fakes.

Each helper imports ``slatedb.uniffi`` lazily on first call. Type
annotations reference the concrete uniffi classes via
``TYPE_CHECKING``-only imports so static checkers see real types
without forcing slatedb to be installed at import time.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING, cast

from shardyfusion.errors import DbAdapterError

if TYPE_CHECKING:
    from slatedb.uniffi import (
        Db,
        DbBuilder,
        DbReader,
        DbReaderBuilder,
        FlushOptions,
        FlushType,
        KeyRange,
        ObjectStore,
        Settings,
        WriteBatch,
    )


def _import_uniffi() -> ModuleType:
    try:
        return import_module("slatedb.uniffi")
    except ImportError as exc:  # pragma: no cover - runtime dependent
        raise DbAdapterError(
            "slatedb (>=0.12,<0.13) is required at runtime; install the 'slatedb' extra"
        ) from exc


def _get_symbol(module: ModuleType, symbol: str) -> object:
    resolved = getattr(module, symbol, None)
    if resolved is None:
        raise DbAdapterError(f"slatedb.uniffi.{symbol} is unavailable at runtime")
    return resolved


# ---------------------------------------------------------------------------
# Core writer/reader classes
# ---------------------------------------------------------------------------


def get_db_class() -> type[Db]:
    """Return ``slatedb.uniffi.Db`` (the open-database handle)."""
    return cast("type[Db]", _get_symbol(_import_uniffi(), "Db"))


def get_db_builder_class() -> type[DbBuilder]:
    """Return ``slatedb.uniffi.DbBuilder`` (constructs ``Db`` instances)."""
    return cast("type[DbBuilder]", _get_symbol(_import_uniffi(), "DbBuilder"))


def get_db_reader_class() -> type[DbReader]:
    """Return ``slatedb.uniffi.DbReader`` (the reader handle)."""
    return cast("type[DbReader]", _get_symbol(_import_uniffi(), "DbReader"))


def get_db_reader_builder_class() -> type[DbReaderBuilder]:
    """Return ``slatedb.uniffi.DbReaderBuilder``."""
    return cast(
        "type[DbReaderBuilder]",
        _get_symbol(_import_uniffi(), "DbReaderBuilder"),
    )


def get_write_batch_class() -> type[WriteBatch]:
    """Return ``slatedb.uniffi.WriteBatch``."""
    return cast("type[WriteBatch]", _get_symbol(_import_uniffi(), "WriteBatch"))


# ---------------------------------------------------------------------------
# Configuration / options
# ---------------------------------------------------------------------------


def get_settings_class() -> type[Settings]:
    """Return ``slatedb.uniffi.Settings`` (mutable database settings)."""
    return cast("type[Settings]", _get_symbol(_import_uniffi(), "Settings"))


def get_object_store_class() -> type[ObjectStore]:
    """Return ``slatedb.uniffi.ObjectStore``."""
    return cast("type[ObjectStore]", _get_symbol(_import_uniffi(), "ObjectStore"))


def get_flush_options_classes() -> tuple[type[FlushOptions], type[FlushType]]:
    """Return ``(FlushOptions, FlushType)`` for ``flush_with_options`` calls."""
    module = _import_uniffi()
    return (
        cast("type[FlushOptions]", _get_symbol(module, "FlushOptions")),
        cast("type[FlushType]", _get_symbol(module, "FlushType")),
    )


def get_key_range_class() -> type[KeyRange]:
    """Return ``slatedb.uniffi.KeyRange`` (sole arg to ``DbReader.scan``)."""
    return cast("type[KeyRange]", _get_symbol(_import_uniffi(), "KeyRange"))


# ---------------------------------------------------------------------------
# Convenience bundles used by the writer/reader adapters
# ---------------------------------------------------------------------------


def get_writer_symbols() -> tuple[
    type[DbBuilder], type[Db], type[WriteBatch], type[ObjectStore]
]:
    """Return ``(DbBuilder, Db, WriteBatch, ObjectStore)`` in one go."""
    module = _import_uniffi()
    return (
        cast("type[DbBuilder]", _get_symbol(module, "DbBuilder")),
        cast("type[Db]", _get_symbol(module, "Db")),
        cast("type[WriteBatch]", _get_symbol(module, "WriteBatch")),
        cast("type[ObjectStore]", _get_symbol(module, "ObjectStore")),
    )


def get_reader_symbols() -> tuple[
    type[DbReaderBuilder], type[DbReader], type[ObjectStore]
]:
    """Return ``(DbReaderBuilder, DbReader, ObjectStore)`` in one go."""
    module = _import_uniffi()
    return (
        cast("type[DbReaderBuilder]", _get_symbol(module, "DbReaderBuilder")),
        cast("type[DbReader]", _get_symbol(module, "DbReader")),
        cast("type[ObjectStore]", _get_symbol(module, "ObjectStore")),
    )
