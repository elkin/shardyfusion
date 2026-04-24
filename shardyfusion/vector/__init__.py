"""Sharded vector search — write HNSW indices, search across shards via S3."""

from __future__ import annotations

from .async_reader import AsyncShardedVectorReader
from .config import VectorIndexConfig, VectorShardingSpec, VectorWriteConfig
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
from .writer import write_vector_sharded

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
    "VectorRecord",
    "VectorSearchResponse",
    "VectorShardDetail",
    "VectorShardReader",
    "VectorShardReaderFactory",
    "VectorShardingSpec",
    "VectorShardingStrategy",
    "VectorSnapshotInfo",
    "VectorWriteConfig",
    "write_vector_sharded",
]
