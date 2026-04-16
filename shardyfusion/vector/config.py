"""Configuration types for vector search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import ManifestOptions, OutputOptions
from ..credentials import CredentialProvider
from ..metrics._protocol import MetricsCollector
from ..run_registry import RunRegistry
from ..type_defs import S3ConnectionOptions
from .types import (
    DistanceMetric,
    VectorIndexWriterFactory,
    VectorShardingStrategy,
    VectorShardReaderFactory,
)


@dataclass(slots=True)
class VectorIndexConfig:
    """Parameters for the per-shard HNSW index."""

    dim: int
    metric: DistanceMetric = DistanceMetric.COSINE
    index_type: str = "hnsw"
    index_params: dict[str, Any] = field(default_factory=dict)
    quantization: str | None = None


@dataclass(slots=True)
class VectorShardingSpec:
    """How vectors are distributed across shards."""

    strategy: VectorShardingStrategy = VectorShardingStrategy.CLUSTER
    num_probes: int = 1

    # CLUSTER: user-provided centroids or library-trained
    centroids: np.ndarray | None = None  # shape (num_dbs, dim)
    train_centroids: bool = False
    centroids_training_sample_size: int = 100_000

    # LSH: random hyperplane hashing
    num_hash_bits: int = 8
    hyperplanes: np.ndarray | None = (
        None  # shape (num_hash_bits, dim), loaded at read time
    )

    # CEL: expression-based sharding (reuses the existing CEL module)
    cel_expr: str | None = None
    cel_columns: dict[str, str] | None = None  # column name -> CelType value
    routing_values: list[int | str | bytes] | None = None  # categorical CEL


@dataclass(slots=True)
class VectorSpecSharding:
    """Sharding configuration for vector index writes via VectorSpec."""

    strategy: str = "cluster"
    num_probes: int = 1
    centroids: np.ndarray | None = None
    train_centroids: bool = False
    centroids_training_sample_size: int = 100_000
    num_hash_bits: int = 8
    hyperplanes: np.ndarray | None = None
    cel_expr: str | None = None
    cel_columns: dict[str, str] | None = None
    routing_values: list[int | str | bytes] | None = None


@dataclass(slots=True)
class VectorWriteConfig:
    """Configuration for write_vector_sharded()."""

    num_dbs: int | None = None
    s3_prefix: str = ""
    index_config: VectorIndexConfig = field(
        default_factory=lambda: VectorIndexConfig(dim=0)
    )
    sharding: VectorShardingSpec = field(default_factory=VectorShardingSpec)
    output: OutputOptions = field(default_factory=OutputOptions)
    manifest: ManifestOptions = field(default_factory=ManifestOptions)
    adapter_factory: VectorIndexWriterFactory | None = None
    reader_factory: VectorShardReaderFactory | None = None
    batch_size: int = 10_000
    max_writes_per_second: float | None = None
    credential_provider: CredentialProvider | None = None
    s3_connection_options: S3ConnectionOptions | None = None
    metrics_collector: MetricsCollector | None = None
    run_registry: RunRegistry | None = None
