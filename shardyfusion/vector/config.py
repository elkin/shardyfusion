"""Configuration types for vector search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import ManifestOptions, OutputOptions

if TYPE_CHECKING:
    from ..config import VectorSpec
from ..credentials import CredentialProvider
from ..errors import ConfigValidationError
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

    def __post_init__(self) -> None:
        if self.index_config.dim <= 0:
            raise ConfigValidationError(
                "index_config.dim must be > 0; provide an explicit "
                f"VectorIndexConfig (got {self.index_config.dim})"
            )
        if self.num_dbs is not None and self.num_dbs <= 0:
            raise ConfigValidationError(f"num_dbs must be > 0, got {self.num_dbs}")

    @classmethod
    def from_vector_spec(
        cls,
        *,
        vector_spec: VectorSpec,
        num_dbs: int,
        s3_prefix: str,
        output: OutputOptions | None = None,
        manifest: ManifestOptions | None = None,
        adapter_factory: VectorIndexWriterFactory | None = None,
        batch_size: int = 10_000,
        max_writes_per_second: float | None = None,
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        metrics_collector: MetricsCollector | None = None,
        run_registry: RunRegistry | None = None,
    ) -> VectorWriteConfig:
        """Build a ``VectorWriteConfig`` from a ``VectorSpec``.

        This is a convenience factory for distributed writers (Spark/Dask/Ray)
        that mirrors the old ``HashWriteConfig`` + ``vector_spec`` pattern.

        Args:
            vector_spec: The ``VectorSpec`` carrying dim, metric, index params,
                and sharding strategy.
            num_dbs: Number of shard databases.
            s3_prefix: S3 location for shards and manifests.
            output: Optional output path/layout overrides.
            manifest: Optional manifest build/publish overrides.
            adapter_factory: Optional vector adapter factory.
            batch_size: Number of vectors per write batch (default 10,000).
            max_writes_per_second: Optional rate limit.
            credential_provider: Optional S3 credential provider.
            s3_connection_options: Optional S3 transport overrides.
            metrics_collector: Optional metrics observer.
            run_registry: Optional run registry.

        Returns:
            A fully-built ``VectorWriteConfig``.
        """
        return cls(
            num_dbs=num_dbs,
            s3_prefix=s3_prefix,
            index_config=vector_spec.to_vector_index_config(),
            sharding=vector_spec.to_vector_sharding_spec(),
            output=output or OutputOptions(),
            manifest=manifest or ManifestOptions(),
            adapter_factory=adapter_factory,
            batch_size=batch_size,
            max_writes_per_second=max_writes_per_second,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
            metrics_collector=metrics_collector,
            run_registry=run_registry,
        )
