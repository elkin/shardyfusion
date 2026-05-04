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

Each helper imports ``slatedb.uniffi`` lazily on first call.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from shardyfusion.errors import DbAdapterError


def _import_uniffi() -> Any:
    try:
        return import_module("slatedb.uniffi")
    except ImportError as exc:  # pragma: no cover - runtime dependent
        raise DbAdapterError(
            "slatedb (>=0.12,<0.13) is required at runtime; install the 'slatedb' extra"
        ) from exc


def _get_symbol(module: Any, symbol: str) -> Any:
    resolved = getattr(module, symbol, None)
    if resolved is None:
        raise DbAdapterError(f"slatedb.uniffi.{symbol} is unavailable at runtime")
    return cast(Any, resolved)


# ---------------------------------------------------------------------------
# Core writer/reader classes
# ---------------------------------------------------------------------------


def get_db_class() -> Any:
    """Return ``slatedb.uniffi.Db`` (the open-database handle)."""
    return _get_symbol(_import_uniffi(), "Db")


def get_db_builder_class() -> Any:
    """Return ``slatedb.uniffi.DbBuilder`` (constructs ``Db`` instances)."""
    return _get_symbol(_import_uniffi(), "DbBuilder")


def get_db_reader_class() -> Any:
    """Return ``slatedb.uniffi.DbReader`` (the reader handle)."""
    return _get_symbol(_import_uniffi(), "DbReader")


def get_db_reader_builder_class() -> Any:
    """Return ``slatedb.uniffi.DbReaderBuilder``."""
    return _get_symbol(_import_uniffi(), "DbReaderBuilder")


def get_write_batch_class() -> Any:
    """Return ``slatedb.uniffi.WriteBatch``."""
    return _get_symbol(_import_uniffi(), "WriteBatch")


# ---------------------------------------------------------------------------
# Configuration / options
# ---------------------------------------------------------------------------


def get_settings_class() -> Any:
    """Return ``slatedb.uniffi.Settings`` (mutable database settings)."""
    return _get_symbol(_import_uniffi(), "Settings")


def get_object_store_class() -> Any:
    """Return ``slatedb.uniffi.ObjectStore``."""
    return _get_symbol(_import_uniffi(), "ObjectStore")


def get_flush_options_classes() -> tuple[Any, Any]:
    """Return ``(FlushOptions, FlushType)`` for ``flush_with_options`` calls."""
    module = _import_uniffi()
    return _get_symbol(module, "FlushOptions"), _get_symbol(module, "FlushType")


def get_key_range_class() -> Any:
    """Return ``slatedb.uniffi.KeyRange`` (sole arg to ``DbReader.scan``)."""
    return _get_symbol(_import_uniffi(), "KeyRange")


# ---------------------------------------------------------------------------
# Convenience bundles used by the writer/reader adapters
# ---------------------------------------------------------------------------


def get_writer_symbols() -> tuple[Any, Any, Any, Any]:
    """Return ``(DbBuilder, Db, WriteBatch, ObjectStore)`` in one go."""
    module = _import_uniffi()
    return (
        _get_symbol(module, "DbBuilder"),
        _get_symbol(module, "Db"),
        _get_symbol(module, "WriteBatch"),
        _get_symbol(module, "ObjectStore"),
    )


def get_reader_symbols() -> tuple[Any, Any, Any]:
    """Return ``(DbReaderBuilder, DbReader, ObjectStore)`` in one go."""
    module = _import_uniffi()
    return (
        _get_symbol(module, "DbReaderBuilder"),
        _get_symbol(module, "DbReader"),
        _get_symbol(module, "ObjectStore"),
    )
