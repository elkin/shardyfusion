"""Core types, protocols, and enums for vector search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, Protocol, Self

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from .config import VectorIndexConfig


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


@unique
class DistanceMetric(str, Enum):
    """Distance function used for ANN search."""

    COSINE = "cosine"
    L2 = "l2"
    DOT_PRODUCT = "dot_product"


@unique
class VectorShardingStrategy(str, Enum):
    """How vectors are distributed across shards."""

    CLUSTER = "cluster"
    LSH = "lsh"
    EXPLICIT = "explicit"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VectorRecord:
    """A single vector record to be written."""

    id: int | str
    vector: np.ndarray  # shape: (dim,)
    payload: dict[str, Any] | None = None
    shard_id: int | None = None  # for EXPLICIT sharding


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One result from a vector search."""

    id: int | str
    score: float
    vector: np.ndarray | None = None
    payload: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class VectorSearchResponse:
    """Merged search results across queried shards."""

    results: list[SearchResult]
    num_shards_queried: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class VectorShardDetail:
    """Per-shard metadata for inspection."""

    db_id: int
    db_url: str | None
    vector_count: int
    checkpoint_id: str | None


@dataclass(frozen=True, slots=True)
class VectorSnapshotInfo:
    """Snapshot-level metadata for a vector index."""

    run_id: str
    num_dbs: int
    dim: int
    metric: DistanceMetric
    sharding: VectorShardingStrategy
    manifest_ref: str
    total_vectors: int


# ---------------------------------------------------------------------------
# HNSW graph structures (serialization intermediary)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HnswNode:
    """A single node in the HNSW graph."""

    max_layer: int
    neighbors: dict[int, list[int]]  # layer -> neighbor node_ids


@dataclass(frozen=True, slots=True)
class HnswGraph:
    """Complete HNSW graph structure (output of a graph builder)."""

    entry_point: int
    max_layer: int
    nodes: dict[int, HnswNode]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class HnswGraphBuilder(Protocol):
    """Build-time only: construct HNSW graph in memory from vectors."""

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None: ...

    def build(self) -> HnswGraph: ...


class VectorIndexWriter(Protocol):
    """Per-shard writer that builds a vector index locally and uploads to S3."""

    def add_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None: ...

    def flush(self) -> None: ...

    def checkpoint(self) -> str | None: ...

    def close(self) -> None: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *exc: object) -> None: ...


class VectorIndexWriterFactory(Protocol):
    """Factory for creating per-shard vector index writers."""

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: VectorIndexConfig,
    ) -> VectorIndexWriter: ...


class VectorShardReader(Protocol):
    """Per-shard search interface (same for range-read and download tiers)."""

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        ef: int = 50,
    ) -> list[SearchResult]: ...

    def close(self) -> None: ...


class VectorShardReaderFactory(Protocol):
    """Factory for creating per-shard readers."""

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: VectorIndexConfig,
    ) -> VectorShardReader: ...
