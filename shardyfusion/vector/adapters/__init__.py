"""Vector index adapters.

Adapter factories are imported lazily so that optional dependencies
(e.g. ``lancedb``) are only required at runtime when the adapter is
actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lancedb_adapter import (
        LanceDbReaderFactory as LanceDbReaderFactory,
    )
    from .lancedb_adapter import (
        LanceDbShardReader as LanceDbShardReader,
    )
    from .lancedb_adapter import (
        LanceDbWriter as LanceDbWriter,
    )
    from .lancedb_adapter import (
        LanceDbWriterFactory as LanceDbWriterFactory,
    )

__all__ = [
    "LanceDbReaderFactory",
    "LanceDbShardReader",
    "LanceDbWriter",
    "LanceDbWriterFactory",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from . import lancedb_adapter

        return getattr(lancedb_adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
