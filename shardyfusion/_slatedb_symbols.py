"""Helpers for resolving SlateDB runtime symbols with clear errors."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from shardyfusion.errors import DbAdapterError


def _import_slatedb() -> Any:
    try:
        return import_module("slatedb")
    except ImportError as exc:  # pragma: no cover - runtime dependent
        raise DbAdapterError("slatedb package is required at runtime") from exc


def _get_symbol(module: Any, symbol: str) -> Any:
    resolved = getattr(module, symbol, None)
    if resolved is None:
        raise DbAdapterError(f"slatedb.{symbol} is unavailable at runtime")
    return cast(Any, resolved)


def get_slatedb_reader_class() -> Any:
    module = _import_slatedb()
    return _get_symbol(module, "SlateDBReader")


def get_slatedb_writer_class() -> Any:
    module = _import_slatedb()
    return _get_symbol(module, "SlateDB")


def get_slatedb_write_batch_class() -> Any:
    module = _import_slatedb()
    return _get_symbol(module, "WriteBatch")


def get_slatedb_writer_symbols() -> tuple[Any, Any]:
    module = _import_slatedb()
    return _get_symbol(module, "SlateDB"), _get_symbol(module, "WriteBatch")
