"""Configuration types for vector search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..config import (
    VectorWriteRateLimitConfig,
    WriterLifecycleConfig,
    WriterManifestConfig,
    WriterObservabilityConfig,
    WriterOutputConfig,
    WriterStorageConfig,
    validate_configs,
)
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
class VectorShardingConfig:
    """How vectors are distributed across shards."""

    num_dbs: int | None = None
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


VectorShardingSpec = VectorShardingConfig


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
class VectorAdapterConfig:
    """Vector index adapter settings."""

    adapter_factory: VectorIndexWriterFactory | None = None
    reader_factory: VectorShardReaderFactory | None = None
    batch_size: int = 10_000

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ConfigValidationError(
                f"adapter.batch_size must be > 0, got {self.batch_size}"
            )


@dataclass(slots=True)
class VectorRecordInput:
    """Marker input for Iterable[VectorRecord] vector writes."""

    def validate(self) -> None:
        return None


@dataclass(slots=True)
class VectorWriteOptions:
    """Per-call execution settings for vector writes."""

    verify_routing: bool = True

    def validate(self) -> None:
        return None


@dataclass(slots=True, init=False)
class VectorShardedWriteConfig:
    """Configuration for vector write_sharded()."""

    index_config: VectorIndexConfig = field(
        default_factory=lambda: VectorIndexConfig(dim=0)
    )
    sharding: VectorShardingConfig = field(default_factory=VectorShardingConfig)
    storage: WriterStorageConfig = field(default_factory=WriterStorageConfig)
    output: WriterOutputConfig = field(default_factory=WriterOutputConfig)
    manifest: WriterManifestConfig = field(default_factory=WriterManifestConfig)
    adapter: VectorAdapterConfig = field(default_factory=VectorAdapterConfig)
    rate_limits: VectorWriteRateLimitConfig = field(
        default_factory=VectorWriteRateLimitConfig
    )
    observability: WriterObservabilityConfig = field(
        default_factory=WriterObservabilityConfig
    )
    lifecycle: WriterLifecycleConfig = field(default_factory=WriterLifecycleConfig)

    def __init__(
        self,
        *,
        index_config: VectorIndexConfig | None = None,
        sharding: VectorShardingConfig | None = None,
        storage: WriterStorageConfig | None = None,
        output: WriterOutputConfig | None = None,
        manifest: WriterManifestConfig | None = None,
        adapter: VectorAdapterConfig | None = None,
        rate_limits: VectorWriteRateLimitConfig | None = None,
        observability: WriterObservabilityConfig | None = None,
        lifecycle: WriterLifecycleConfig | None = None,
        num_dbs: int | None = None,
        s3_prefix: str | None = None,
        adapter_factory: VectorIndexWriterFactory | None = None,
        reader_factory: VectorShardReaderFactory | None = None,
        batch_size: int | None = None,
        max_writes_per_second: float | None = None,
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        metrics_collector: MetricsCollector | None = None,
        run_registry: RunRegistry | None = None,
    ) -> None:
        resolved_index_config = index_config or VectorIndexConfig(dim=0)
        if resolved_index_config.dim <= 0:
            raise ConfigValidationError(
                "index_config.dim must be > 0; provide an explicit "
                f"VectorIndexConfig (got {resolved_index_config.dim})"
            )
        if sharding is not None and num_dbs is not None:
            if sharding.num_dbs is not None and sharding.num_dbs != num_dbs:
                raise ConfigValidationError(
                    "sharding.num_dbs cannot disagree with num_dbs"
                )
            sharding.num_dbs = num_dbs
        if storage is not None and (
            s3_prefix is not None
            or credential_provider is not None
            or s3_connection_options is not None
        ):
            raise ConfigValidationError(
                "storage cannot be combined with s3_prefix, "
                "credential_provider, or s3_connection_options"
            )
        if adapter is not None and (
            adapter_factory is not None
            or reader_factory is not None
            or batch_size is not None
        ):
            raise ConfigValidationError(
                "adapter cannot be combined with adapter_factory, "
                "reader_factory, or batch_size"
            )
        if rate_limits is not None and max_writes_per_second is not None:
            raise ConfigValidationError(
                "rate_limits cannot be combined with max_writes_per_second"
            )
        if observability is not None and metrics_collector is not None:
            raise ConfigValidationError(
                "observability cannot be combined with metrics_collector"
            )
        if lifecycle is not None and run_registry is not None:
            raise ConfigValidationError(
                "lifecycle cannot be combined with run_registry"
            )

        self.index_config = resolved_index_config
        self.sharding = sharding or VectorShardingConfig(num_dbs=num_dbs)
        self.storage = storage or WriterStorageConfig(
            s3_prefix=s3_prefix or "",
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )
        self.output = output or WriterOutputConfig()
        self.manifest = manifest or WriterManifestConfig()
        self.adapter = adapter or VectorAdapterConfig(
            adapter_factory=adapter_factory,
            reader_factory=reader_factory,
            batch_size=10_000 if batch_size is None else batch_size,
        )
        self.rate_limits = rate_limits or VectorWriteRateLimitConfig(
            max_writes_per_second=max_writes_per_second
        )
        self.observability = observability or WriterObservabilityConfig(
            metrics_collector=metrics_collector
        )
        self.lifecycle = lifecycle or WriterLifecycleConfig(run_registry=run_registry)
        self.validate()

    def validate(self) -> None:
        if self.index_config.dim <= 0:
            raise ConfigValidationError(
                "index_config.dim must be > 0; provide an explicit "
                f"VectorIndexConfig (got {self.index_config.dim})"
            )
        if self.num_dbs is not None and self.num_dbs <= 0:
            raise ConfigValidationError(f"num_dbs must be > 0, got {self.num_dbs}")
        validate_configs(
            self.storage,
            self.output,
            self.manifest,
            self.adapter,
            self.rate_limits,
            self.observability,
            self.lifecycle,
        )

        match self.sharding.strategy:
            case VectorShardingStrategy.CLUSTER:
                if (
                    self.sharding.centroids is None
                    and not self.sharding.train_centroids
                ):
                    raise ConfigValidationError(
                        "CLUSTER sharding requires either centroids or "
                        "train_centroids=True"
                    )
            case VectorShardingStrategy.CEL:
                if not self.sharding.cel_expr:
                    raise ConfigValidationError(
                        "CEL sharding requires cel_expr to be set"
                    )
                if not self.sharding.cel_columns:
                    raise ConfigValidationError(
                        "CEL sharding requires cel_columns to be set"
                    )
                if self.sharding.num_probes > 1:
                    raise ConfigValidationError(
                        "num_probes is only supported for CLUSTER and LSH "
                        f"sharding, got {self.sharding.num_probes} for "
                        f"{self.sharding.strategy.value}"
                    )
            case VectorShardingStrategy.EXPLICIT if self.sharding.num_probes > 1:
                raise ConfigValidationError(
                    "num_probes is only supported for CLUSTER and LSH "
                    f"sharding, got {self.sharding.num_probes} for "
                    f"{self.sharding.strategy.value}"
                )
        if self.sharding.num_probes < 1:
            raise ConfigValidationError(
                f"num_probes must be >= 1, got {self.sharding.num_probes}"
            )

    @property
    def num_dbs(self) -> int | None:
        return self.sharding.num_dbs

    @property
    def s3_prefix(self) -> str:
        return self.storage.s3_prefix

    @s3_prefix.setter
    def s3_prefix(self, value: str) -> None:
        self.storage.s3_prefix = value
        self.storage.validate()

    @property
    def adapter_factory(self) -> VectorIndexWriterFactory | None:
        return self.adapter.adapter_factory

    @adapter_factory.setter
    def adapter_factory(self, value: VectorIndexWriterFactory | None) -> None:
        self.adapter.adapter_factory = value

    @property
    def reader_factory(self) -> VectorShardReaderFactory | None:
        return self.adapter.reader_factory

    @reader_factory.setter
    def reader_factory(self, value: VectorShardReaderFactory | None) -> None:
        self.adapter.reader_factory = value

    @property
    def batch_size(self) -> int:
        return self.adapter.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.adapter.batch_size = value
        self.adapter.validate()

    @property
    def credential_provider(self) -> CredentialProvider | None:
        return self.storage.credential_provider

    @credential_provider.setter
    def credential_provider(self, value: CredentialProvider | None) -> None:
        self.storage.credential_provider = value

    @property
    def s3_connection_options(self) -> S3ConnectionOptions | None:
        return self.storage.s3_connection_options

    @s3_connection_options.setter
    def s3_connection_options(self, value: S3ConnectionOptions | None) -> None:
        self.storage.s3_connection_options = value

    @property
    def metrics_collector(self) -> MetricsCollector | None:
        return self.observability.metrics_collector

    @metrics_collector.setter
    def metrics_collector(self, value: MetricsCollector | None) -> None:
        self.observability.metrics_collector = value

    @property
    def run_registry(self) -> RunRegistry | None:
        return self.lifecycle.run_registry

    @run_registry.setter
    def run_registry(self, value: RunRegistry | None) -> None:
        self.lifecycle.run_registry = value
