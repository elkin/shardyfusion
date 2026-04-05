from ._state import ShardReaderHandle
from ._types import ReaderHealth, ShardDetail, SlateDbReaderFactory, SnapshotInfo
from .async_reader import (
    AsyncShardedReader,
    AsyncShardReaderHandle,
    AsyncSlateDbReaderFactory,
)
from .concurrent_reader import ConcurrentShardedReader
from .reader import ShardedReader

__all__ = [
    "AsyncShardedReader",
    "AsyncShardReaderHandle",
    "AsyncSlateDbReaderFactory",
    "ConcurrentShardedReader",
    "ReaderHealth",
    "ShardDetail",
    "ShardedReader",
    "ShardReaderHandle",
    "SlateDbReaderFactory",
    "SnapshotInfo",
    "UnifiedShardedReader",
]


def __getattr__(name: str):  # noqa: ANN202
    if name == "UnifiedShardedReader":
        from .unified_reader import UnifiedShardedReader

        return UnifiedShardedReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
