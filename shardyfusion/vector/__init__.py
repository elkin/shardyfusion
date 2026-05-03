"""Sharded vector search — write HNSW indices, search across shards via S3."""

from __future__ import annotations

from .async_reader import AsyncShardedVectorReader
from .config import (
    VectorIndexConfig,
    VectorRecordInput,
    VectorShardedWriteConfig,
    VectorShardingConfig,
    VectorShardingSpec,
    VectorWriteOptions,
)
from .reader import ShardedVectorReader, VectorReaderHealth
from .types import (
    AsyncVectorShardReader,
    AsyncVectorShardReaderFactory,
    DistanceMetric,
    SearchResult,
    VectorIndexWriter,
    VectorIndexWriterFactory,
    VectorRecord,
    VectorSearchResponse,
    VectorShardDetail,
    VectorShardingStrategy,
    VectorShardReader,
    VectorShardReaderFactory,
    VectorSnapshotInfo,
)
from .writer import write_sharded

__all__ = [
    "AsyncShardedVectorReader",
    "AsyncVectorShardReader",
    "AsyncVectorShardReaderFactory",
    "DistanceMetric",
    "SearchResult",
    "ShardedVectorReader",
    "VectorIndexConfig",
    "VectorIndexWriter",
    "VectorIndexWriterFactory",
    "VectorReaderHealth",
    "VectorRecordInput",
    "VectorRecord",
    "VectorSearchResponse",
    "VectorShardDetail",
    "VectorShardReader",
    "VectorShardReaderFactory",
    "VectorShardingConfig",
    "VectorShardingSpec",
    "VectorShardingStrategy",
    "VectorSnapshotInfo",
    "VectorShardedWriteConfig",
    "VectorWriteOptions",
    "write_sharded",
]
