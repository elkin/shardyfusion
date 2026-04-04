"""Vector index adapters.

Adapter factories are imported lazily so that optional dependencies
(e.g. ``usearch``) are only required at runtime when the adapter is
actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .usearch_adapter import (
        USearchReaderFactory,
        USearchShardReader,
        USearchWriter,
        USearchWriterFactory,
    )

__all__ = [
    "USearchReaderFactory",
    "USearchShardReader",
    "USearchWriter",
    "USearchWriterFactory",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from . import usearch_adapter

        return getattr(usearch_adapter, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
