"""Vector index adapters.

Adapter factories are imported lazily so that optional dependencies
(e.g. ``usearch``, ``sqlite-vec``) are only required at runtime when the
adapter is actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .sqlite_vec_adapter import (
        SqliteVecVectorReaderFactory,
        SqliteVecVectorShardReader,
        SqliteVecVectorWriterFactory,
    )
    from .usearch_adapter import (
        USearchReaderFactory,
        USearchShardReader,
        USearchWriter,
        USearchWriterFactory,
    )

__all__ = [
    "SqliteVecVectorReaderFactory",
    "SqliteVecVectorShardReader",
    "SqliteVecVectorWriterFactory",
    "USearchReaderFactory",
    "USearchShardReader",
    "USearchWriter",
    "USearchWriterFactory",
]

_USEARCH_NAMES = {
    "USearchReaderFactory",
    "USearchShardReader",
    "USearchWriter",
    "USearchWriterFactory",
}
_SQLITE_VEC_NAMES = {
    "SqliteVecVectorReaderFactory",
    "SqliteVecVectorShardReader",
    "SqliteVecVectorWriterFactory",
}


def __getattr__(name: str) -> object:
    if name in _USEARCH_NAMES:
        from . import usearch_adapter

        return getattr(usearch_adapter, name)
    if name in _SQLITE_VEC_NAMES:
        from . import sqlite_vec_adapter

        return getattr(sqlite_vec_adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
