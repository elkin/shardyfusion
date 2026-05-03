"""Configuration models for sharded snapshot writes."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeAlias, TypeVar
from urllib.parse import urlparse

from .credentials import CredentialProvider
from .errors import ConfigValidationError
from .metrics import MetricsCollector
from .sharding_types import (
    DB_ID_COL,
    KeyEncoding,
    RoutingValue,
    validate_routing_values,
)
from .slatedb_adapter import DbAdapterFactory
from .type_defs import JsonObject, KeyInput, RetryConfig, S3ConnectionOptions

if TYPE_CHECKING:
    from .manifest_store import ManifestStore
    from .run_registry import RunRegistry
    from .vector.config import VectorIndexConfig, VectorShardingSpec, VectorSpecSharding

_SAFE_SEGMENT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


T = TypeVar("T")
VectorWriteTuple: TypeAlias = tuple[int | str, Any, dict[str, Any] | None]


class ValidatableConfig(Protocol):
    """Configuration object that can validate its own invariants."""

    def validate(self) -> None:
        """Raise ConfigValidationError when the configuration is invalid."""


def validate_configs(*configs: ValidatableConfig) -> None:
    """Validate multiple config-like objects in order."""

    for config in configs:
        config.validate()


@dataclass(slots=True)
class WriterStorageConfig:
    """Object-store location and connection settings for writer output."""

    s3_prefix: str = ""
    credential_provider: CredentialProvider | None = None
    s3_connection_options: S3ConnectionOptions | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_s3_prefix(self.s3_prefix)


@dataclass(slots=True)
class WriterOutputConfig:
    """Output path/layout settings for shard database writes."""

    run_id: str | None = None
    db_path_template: str = "db={db_id:05d}"
    shard_prefix: str = "shards"
    run_registry_prefix: str = "runs"
    local_root: str = field(
        default_factory=lambda: str(Path(tempfile.gettempdir()) / "shardyfusion")
    )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_segment(self.shard_prefix, field_name="output.shard_prefix")
        _validate_segment(
            self.run_registry_prefix,
            field_name="output.run_registry_prefix",
        )
        try:
            self.db_path_template.format(db_id=0)
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover - defensive formatting surface
            raise ConfigValidationError(
                "output.db_path_template must support format(db_id=...)"
            ) from exc


@dataclass(slots=True)
class WriterManifestConfig:
    """Manifest build and publish settings."""

    store: ManifestStore | None = None
    custom_manifest_fields: JsonObject = field(default_factory=dict)
    credential_provider: CredentialProvider | None = None
    s3_connection_options: S3ConnectionOptions | None = None

    def validate(self) -> None:
        if not isinstance(self.custom_manifest_fields, dict):
            raise ConfigValidationError(
                "manifest.custom_manifest_fields must be a dict"
            )


@dataclass(slots=True)
class KeyValueWriteConfig:
    """Key/value storage settings shared by KV writers."""

    key_encoding: KeyEncoding = KeyEncoding.U64BE
    batch_size: int = 50_000
    adapter_factory: DbAdapterFactory | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.key_encoding, KeyEncoding):
            try:
                self.key_encoding = KeyEncoding.from_value(self.key_encoding)
            except ValueError as exc:
                raise ConfigValidationError(str(exc)) from exc
        if self.batch_size <= 0:
            raise ConfigValidationError("kv.batch_size must be > 0")


@dataclass(slots=True)
class HashShardingConfig:
    """HASH sharding settings."""

    num_dbs: int | None = None
    max_keys_per_shard: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.max_keys_per_shard is not None:
            if self.num_dbs is not None:
                raise ConfigValidationError(
                    "sharding.num_dbs must be None when "
                    "sharding.max_keys_per_shard is set"
                )
            if self.max_keys_per_shard <= 0:
                raise ConfigValidationError("sharding.max_keys_per_shard must be > 0")
        elif self.num_dbs is None or self.num_dbs <= 0:
            raise ConfigValidationError("sharding.num_dbs must be > 0")


@dataclass(slots=True)
class CelShardingConfig:
    """CEL sharding settings."""

    cel_expr: str = ""
    cel_columns: dict[str, str] = field(default_factory=dict)
    routing_values: list[RoutingValue] | None = None
    infer_routing_values_from_data: bool = False

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.cel_expr:
            raise ConfigValidationError("CEL strategy requires cel_expr")
        if not self.cel_columns:
            raise ConfigValidationError("CEL strategy requires cel_columns")
        if self.infer_routing_values_from_data and self.routing_values is not None:
            raise ConfigValidationError(
                "infer_routing_values_from_data cannot be combined with routing_values"
            )
        if self.routing_values is not None:
            try:
                validate_routing_values(self.routing_values)
            except ValueError as exc:
                raise ConfigValidationError(str(exc)) from exc


@dataclass(slots=True)
class WriterRetryConfig:
    """Retry settings for shard writes."""

    shard_retry: RetryConfig | None = None

    def validate(self) -> None:
        return None


@dataclass(slots=True)
class KvWriteRateLimitConfig:
    """KV write token-bucket rate limits."""

    max_writes_per_second: float | None = None
    max_write_bytes_per_second: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_positive_optional(
            self.max_writes_per_second,
            field_name="rate_limits.max_writes_per_second",
        )
        _validate_positive_optional(
            self.max_write_bytes_per_second,
            field_name="rate_limits.max_write_bytes_per_second",
        )


@dataclass(slots=True)
class VectorWriteRateLimitConfig:
    """Vector write token-bucket rate limits."""

    max_writes_per_second: float | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _validate_positive_optional(
            self.max_writes_per_second,
            field_name="rate_limits.max_writes_per_second",
        )


@dataclass(slots=True)
class WriterObservabilityConfig:
    """Optional writer observability hooks."""

    metrics_collector: MetricsCollector | None = None

    def validate(self) -> None:
        return None


@dataclass(slots=True)
class WriterLifecycleConfig:
    """Run lifecycle hooks."""

    run_registry: RunRegistry | None = None

    def validate(self) -> None:
        return None


_VALID_VECTOR_METRICS = frozenset({"cosine", "l2", "dot_product"})

VectorMetricLiteral: TypeAlias = Literal["cosine", "l2", "dot_product"]


def vector_metric_to_str(
    metric: VectorMetricLiteral | str | object,
) -> VectorMetricLiteral:
    """Normalize a vector metric to its stable manifest string value."""
    metric_str = getattr(metric, "value", metric)
    if not isinstance(metric_str, str) or metric_str not in _VALID_VECTOR_METRICS:
        raise ConfigValidationError(
            f"vector_spec.metric must be one of {sorted(_VALID_VECTOR_METRICS)}, "
            f"got {metric!r}"
        )
    return metric_str  # type: ignore[return-value]


def _coerce_vector_metric(
    metric: VectorMetricLiteral | str | object,
) -> VectorMetricLiteral:
    """Validate and normalize a vector metric to its string literal form.

    This keeps ``VectorSpec.metric`` as a plain string so downstream code
    does not need to handle both ``DistanceMetric`` enums and strings.
    The conversion to ``DistanceMetric`` happens only when building a
    ``VectorIndexConfig`` (via ``to_vector_index_config``).
    """
    return vector_metric_to_str(metric)


def _get_vector_spec_sharding() -> "VectorSpecSharding":  # noqa: UP037
    from .vector.config import VectorSpecSharding

    return VectorSpecSharding()


@dataclass(slots=True)
class VectorSpec:
    """Specifies how to extract and index vectors alongside KV data.

    Enables unified KV + vector search mode when set on a KV writer config.
    Uses string literals for ``metric`` to avoid importing from the vector
    module (which has heavier dependencies).  Adapters convert to
    ``DistanceMetric`` internally.

    Args:
        dim: Dimensionality of the embedding vectors.
        vector_col: Column name containing the vector in the routing context
            dict (used when ``vector_fn`` is not provided to the writer).
        metric: Distance metric — ``"cosine"``, ``"l2"``, or
            ``"dot_product"``.
        index_type: Index algorithm (default ``"hnsw"``).
        index_params: Algorithm-specific parameters (e.g. ``M``,
            ``ef_construction``).
        quantization: Optional quantization — ``"fp16"``, ``"i8"``, or
            ``None`` (full precision).
    """

    dim: int
    vector_col: str | None = None
    metric: VectorMetricLiteral = "cosine"
    index_type: str = "hnsw"
    index_params: dict[str, object] = field(default_factory=dict)
    quantization: str | None = None
    sharding: "VectorSpecSharding" = field(  # noqa: UP037
        default_factory=lambda: _get_vector_spec_sharding()
    )

    def __post_init__(self) -> None:
        self.metric = _coerce_vector_metric(self.metric)

    def to_vector_index_config(self) -> VectorIndexConfig:
        from .vector.config import VectorIndexConfig
        from .vector.types import DistanceMetric

        metric = self.metric
        if isinstance(metric, str):
            metric = DistanceMetric(metric)
        return VectorIndexConfig(
            dim=self.dim,
            metric=metric,
            index_type=self.index_type,
            index_params=self.index_params,
            quantization=self.quantization,
        )

    def to_vector_sharding_spec(self) -> VectorShardingSpec:
        from .vector.config import VectorShardingSpec
        from .vector.types import VectorShardingStrategy

        strategy = VectorShardingStrategy(self.sharding.strategy)
        return VectorShardingSpec(
            strategy=strategy,
            num_probes=self.sharding.num_probes,
            centroids=self.sharding.centroids,
            train_centroids=self.sharding.train_centroids,
            centroids_training_sample_size=self.sharding.centroids_training_sample_size,
            num_hash_bits=self.sharding.num_hash_bits,
            hyperplanes=self.sharding.hyperplanes,
            cel_expr=self.sharding.cel_expr,
            cel_columns=self.sharding.cel_columns,
            routing_values=self.sharding.routing_values,
        )


UnifiedVectorWriteConfig = VectorSpec


@dataclass(slots=True, init=False)
class BaseShardedWriteConfig:
    """Shared public configuration for KV sharded snapshot writes."""

    storage: WriterStorageConfig = field(default_factory=WriterStorageConfig)
    output: WriterOutputConfig = field(default_factory=WriterOutputConfig)
    manifest: WriterManifestConfig = field(default_factory=WriterManifestConfig)
    kv: KeyValueWriteConfig = field(default_factory=KeyValueWriteConfig)
    retry: WriterRetryConfig = field(default_factory=WriterRetryConfig)
    rate_limits: KvWriteRateLimitConfig = field(default_factory=KvWriteRateLimitConfig)
    observability: WriterObservabilityConfig = field(
        default_factory=WriterObservabilityConfig
    )
    lifecycle: WriterLifecycleConfig = field(default_factory=WriterLifecycleConfig)
    vector: VectorSpec | None = None
    shard_id_col: str = DB_ID_COL

    def __init__(
        self,
        *,
        storage: WriterStorageConfig | None = None,
        output: WriterOutputConfig | None = None,
        manifest: WriterManifestConfig | None = None,
        kv: KeyValueWriteConfig | None = None,
        retry: WriterRetryConfig | None = None,
        rate_limits: KvWriteRateLimitConfig | None = None,
        observability: WriterObservabilityConfig | None = None,
        lifecycle: WriterLifecycleConfig | None = None,
        vector: VectorSpec | None = None,
        shard_id_col: str = DB_ID_COL,
        s3_prefix: str | None = None,
        key_encoding: KeyEncoding | None = None,
        batch_size: int | None = None,
        adapter_factory: DbAdapterFactory | None = None,
        metrics_collector: MetricsCollector | None = None,
        run_registry: RunRegistry | None = None,
        shard_retry: RetryConfig | None = None,
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        vector_spec: VectorSpec | None = None,
        max_writes_per_second: float | None = None,
        max_write_bytes_per_second: float | None = None,
    ) -> None:
        self._init_base(
            storage=storage,
            output=output,
            manifest=manifest,
            kv=kv,
            retry=retry,
            rate_limits=rate_limits,
            observability=observability,
            lifecycle=lifecycle,
            vector=vector,
            shard_id_col=shard_id_col,
            s3_prefix=s3_prefix,
            key_encoding=key_encoding,
            batch_size=batch_size,
            adapter_factory=adapter_factory,
            metrics_collector=metrics_collector,
            run_registry=run_registry,
            shard_retry=shard_retry,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
            vector_spec=vector_spec,
            max_writes_per_second=max_writes_per_second,
            max_write_bytes_per_second=max_write_bytes_per_second,
        )

    def _init_base(
        self,
        *,
        storage: WriterStorageConfig | None,
        output: WriterOutputConfig | None,
        manifest: WriterManifestConfig | None,
        kv: KeyValueWriteConfig | None,
        retry: WriterRetryConfig | None,
        rate_limits: KvWriteRateLimitConfig | None,
        observability: WriterObservabilityConfig | None,
        lifecycle: WriterLifecycleConfig | None,
        vector: VectorSpec | None,
        shard_id_col: str,
        s3_prefix: str | None,
        key_encoding: KeyEncoding | None,
        batch_size: int | None,
        adapter_factory: DbAdapterFactory | None,
        metrics_collector: MetricsCollector | None,
        run_registry: RunRegistry | None,
        shard_retry: RetryConfig | None,
        credential_provider: CredentialProvider | None,
        s3_connection_options: S3ConnectionOptions | None,
        vector_spec: VectorSpec | None,
        max_writes_per_second: float | None,
        max_write_bytes_per_second: float | None,
    ) -> None:
        if storage is not None and (
            s3_prefix is not None
            or credential_provider is not None
            or s3_connection_options is not None
        ):
            raise ConfigValidationError(
                "storage cannot be combined with s3_prefix, "
                "credential_provider, or s3_connection_options"
            )
        if kv is not None and (
            key_encoding is not None
            or batch_size is not None
            or adapter_factory is not None
        ):
            raise ConfigValidationError(
                "kv cannot be combined with key_encoding, batch_size, "
                "or adapter_factory"
            )
        if retry is not None and shard_retry is not None:
            raise ConfigValidationError("retry cannot be combined with shard_retry")
        if rate_limits is not None and (
            max_writes_per_second is not None or max_write_bytes_per_second is not None
        ):
            raise ConfigValidationError(
                "rate_limits cannot be combined with max_writes_per_second "
                "or max_write_bytes_per_second"
            )
        if observability is not None and metrics_collector is not None:
            raise ConfigValidationError(
                "observability cannot be combined with metrics_collector"
            )
        if lifecycle is not None and run_registry is not None:
            raise ConfigValidationError(
                "lifecycle cannot be combined with run_registry"
            )
        if vector is not None and vector_spec is not None:
            raise ConfigValidationError("vector cannot be combined with vector_spec")

        self.storage = storage or WriterStorageConfig(
            s3_prefix=s3_prefix or "",
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )
        self.output = output or WriterOutputConfig()
        self.manifest = manifest or WriterManifestConfig()
        self.kv = kv or KeyValueWriteConfig(
            key_encoding=KeyEncoding.U64BE if key_encoding is None else key_encoding,
            batch_size=50_000 if batch_size is None else batch_size,
            adapter_factory=adapter_factory,
        )
        self.retry = retry or WriterRetryConfig(shard_retry=shard_retry)
        self.rate_limits = rate_limits or KvWriteRateLimitConfig(
            max_writes_per_second=max_writes_per_second,
            max_write_bytes_per_second=max_write_bytes_per_second,
        )
        self.observability = observability or WriterObservabilityConfig(
            metrics_collector=metrics_collector
        )
        self.lifecycle = lifecycle or WriterLifecycleConfig(run_registry=run_registry)
        self.vector = vector if vector is not None else vector_spec
        self.shard_id_col = shard_id_col
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.storage, WriterStorageConfig):
            raise ConfigValidationError("storage must be WriterStorageConfig")
        if not isinstance(self.output, WriterOutputConfig):
            raise ConfigValidationError("output must be WriterOutputConfig")
        if not isinstance(self.manifest, WriterManifestConfig):
            raise ConfigValidationError("manifest must be WriterManifestConfig")
        if not isinstance(self.kv, KeyValueWriteConfig):
            raise ConfigValidationError("kv must be KeyValueWriteConfig")
        if not isinstance(self.retry, WriterRetryConfig):
            raise ConfigValidationError("retry must be WriterRetryConfig")
        if not isinstance(self.rate_limits, KvWriteRateLimitConfig):
            raise ConfigValidationError("rate_limits must be KvWriteRateLimitConfig")
        if not isinstance(self.observability, WriterObservabilityConfig):
            raise ConfigValidationError(
                "observability must be WriterObservabilityConfig"
            )
        if not isinstance(self.lifecycle, WriterLifecycleConfig):
            raise ConfigValidationError("lifecycle must be WriterLifecycleConfig")

        validate_configs(
            self.storage,
            self.output,
            self.manifest,
            self.kv,
            self.retry,
            self.rate_limits,
            self.observability,
            self.lifecycle,
        )

        if not self.shard_id_col or not isinstance(self.shard_id_col, str):
            raise ConfigValidationError("shard_id_col must be a non-empty string")

        if self.vector is not None:
            vs = self.vector
            if vs.dim <= 0:
                raise ConfigValidationError(f"vector.dim must be > 0, got {vs.dim}")
            vs.metric = _coerce_vector_metric(vs.metric)

    @property
    def s3_prefix(self) -> str:
        return self.storage.s3_prefix

    @s3_prefix.setter
    def s3_prefix(self, value: str) -> None:
        self.storage.s3_prefix = value
        self.storage.validate()

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
    def key_encoding(self) -> KeyEncoding:
        return self.kv.key_encoding

    @key_encoding.setter
    def key_encoding(self, value: KeyEncoding) -> None:
        self.kv.key_encoding = value
        self.kv.validate()

    @property
    def batch_size(self) -> int:
        return self.kv.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.kv.batch_size = value
        self.kv.validate()

    @property
    def adapter_factory(self) -> DbAdapterFactory | None:
        return self.kv.adapter_factory

    @adapter_factory.setter
    def adapter_factory(self, value: DbAdapterFactory | None) -> None:
        self.kv.adapter_factory = value

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

    @property
    def shard_retry(self) -> RetryConfig | None:
        return self.retry.shard_retry

    @shard_retry.setter
    def shard_retry(self, value: RetryConfig | None) -> None:
        self.retry.shard_retry = value

    @property
    def vector_spec(self) -> VectorSpec | None:
        return self.vector

    @vector_spec.setter
    def vector_spec(self, value: VectorSpec | None) -> None:
        self.vector = value
        self.validate()


@dataclass(slots=True, init=False)
class HashShardedWriteConfig(BaseShardedWriteConfig):
    """Public configuration for HASH-sharded KV writes."""

    sharding: HashShardingConfig = field(default_factory=HashShardingConfig)

    def __init__(
        self,
        *,
        sharding: HashShardingConfig | None = None,
        num_dbs: int | None = None,
        max_keys_per_shard: int | None = None,
        **base_kwargs: Any,
    ) -> None:
        if sharding is not None and (
            num_dbs is not None or max_keys_per_shard is not None
        ):
            raise ConfigValidationError(
                "sharding cannot be combined with num_dbs or max_keys_per_shard"
            )
        self.sharding = sharding or HashShardingConfig(
            num_dbs=num_dbs,
            max_keys_per_shard=max_keys_per_shard,
        )
        BaseShardedWriteConfig.__init__(self, **base_kwargs)

    def validate(self) -> None:
        BaseShardedWriteConfig.validate(self)
        if not isinstance(self.sharding, HashShardingConfig):
            raise ConfigValidationError("sharding must be HashShardingConfig")
        self.sharding.validate()

    @property
    def num_dbs(self) -> int | None:
        return self.sharding.num_dbs

    @property
    def max_keys_per_shard(self) -> int | None:
        return self.sharding.max_keys_per_shard


@dataclass(slots=True, init=False)
class CelShardedWriteConfig(BaseShardedWriteConfig):
    """Public configuration for CEL-sharded KV writes."""

    sharding: CelShardingConfig = field(default_factory=CelShardingConfig)

    def __init__(
        self,
        *,
        sharding: CelShardingConfig | None = None,
        cel_expr: str | None = None,
        cel_columns: dict[str, str] | None = None,
        routing_values: list[RoutingValue] | None = None,
        infer_routing_values_from_data: bool | None = None,
        **base_kwargs: Any,
    ) -> None:
        has_flat = (
            cel_expr is not None
            or cel_columns is not None
            or routing_values is not None
            or infer_routing_values_from_data is not None
        )
        if sharding is not None and has_flat:
            raise ConfigValidationError(
                "sharding cannot be combined with cel_expr, cel_columns, "
                "routing_values, or infer_routing_values_from_data"
            )
        self.sharding = sharding or CelShardingConfig(
            cel_expr=cel_expr or "",
            cel_columns=cel_columns or {},
            routing_values=routing_values,
            infer_routing_values_from_data=bool(infer_routing_values_from_data),
        )
        BaseShardedWriteConfig.__init__(self, **base_kwargs)

    def validate(self) -> None:
        BaseShardedWriteConfig.validate(self)
        if not isinstance(self.sharding, CelShardingConfig):
            raise ConfigValidationError("sharding must be CelShardingConfig")
        self.sharding.validate()

    @property
    def cel_expr(self) -> str:
        return self.sharding.cel_expr

    @property
    def cel_columns(self) -> dict[str, str]:
        return self.sharding.cel_columns

    @property
    def routing_values(self) -> list[RoutingValue] | None:
        return self.sharding.routing_values

    @property
    def infer_routing_values_from_data(self) -> bool:
        return self.sharding.infer_routing_values_from_data


@dataclass(slots=True, init=False)
class SingleDbWriteConfig(BaseShardedWriteConfig):
    """Public configuration for single-database KV writes."""

    def __init__(self, **base_kwargs: Any) -> None:
        BaseShardedWriteConfig.__init__(self, **base_kwargs)

    def validate(self) -> None:
        BaseShardedWriteConfig.validate(self)

    @property
    def num_dbs(self) -> int:
        return 1


@dataclass(slots=True)
class PythonRecordInput(Generic[T]):
    """How the Python writer extracts logical KV rows."""

    key_fn: Callable[[T], KeyInput]
    value_fn: Callable[[T], bytes]
    columns_fn: Callable[[T], dict[str, Any]] | None = None
    vector_fn: Callable[[T], VectorWriteTuple] | None = None

    def validate(self) -> None:
        if not callable(self.key_fn):
            raise ConfigValidationError("input.key_fn must be callable")
        if not callable(self.value_fn):
            raise ConfigValidationError("input.value_fn must be callable")
        if self.columns_fn is not None and not callable(self.columns_fn):
            raise ConfigValidationError("input.columns_fn must be callable")
        if self.vector_fn is not None and not callable(self.vector_fn):
            raise ConfigValidationError("input.vector_fn must be callable")


@dataclass(slots=True, frozen=True)
class VectorColumnInput:
    """Column-based vector extraction settings."""

    vector_col: str
    id_col: str | None = None
    payload_cols: list[str] | None = None
    shard_id_col: str | None = None
    routing_context_cols: dict[str, str] | None = None

    def validate(self) -> None:
        if not self.vector_col:
            raise ConfigValidationError("input.vector.vector_col must be non-empty")
        if self.id_col is not None and not self.id_col:
            raise ConfigValidationError("input.vector.id_col must be non-empty")


@dataclass(slots=True)
class ColumnWriteInput:
    """How DataFrame/Dataset writers extract logical KV rows."""

    key_col: str
    value_spec: Any
    vector: VectorColumnInput | None = None
    vector_fn: Callable[[Any], VectorWriteTuple] | None = None

    def validate(self) -> None:
        if not self.key_col:
            raise ConfigValidationError("input.key_col must be non-empty")
        if self.value_spec is None:
            raise ConfigValidationError("input.value_spec is required")
        if self.vector is not None:
            self.vector.validate()
        if self.vector_fn is not None and not callable(self.vector_fn):
            raise ConfigValidationError("input.vector_fn must be callable")


@dataclass(slots=True)
class SharedMemoryOptions:
    """Shared-memory limits for Python parallel writer mode."""

    max_total_bytes: int | None = 256 * 1024 * 1024
    max_bytes_per_worker: int | None = 32 * 1024 * 1024

    def validate(self) -> None:
        _validate_positive_optional(
            self.max_total_bytes,
            field_name="options.shared_memory.max_total_bytes",
        )
        _validate_positive_optional(
            self.max_bytes_per_worker,
            field_name="options.shared_memory.max_bytes_per_worker",
        )


@dataclass(slots=True)
class BufferingOptions:
    """Single-process Python buffering caps."""

    max_total_batched_items: int | None = None
    max_total_batched_bytes: int | None = None

    def validate(self) -> None:
        _validate_positive_optional(
            self.max_total_batched_items,
            field_name="options.buffering.max_total_batched_items",
        )
        _validate_positive_optional(
            self.max_total_batched_bytes,
            field_name="options.buffering.max_total_batched_bytes",
        )


@dataclass(slots=True)
class PythonWriteOptions:
    """Per-call execution settings for Python writes."""

    parallel: bool = False
    max_queue_size: int = 100
    shared_memory: SharedMemoryOptions = field(default_factory=SharedMemoryOptions)
    buffering: BufferingOptions = field(default_factory=BufferingOptions)

    def validate(self) -> None:
        if self.max_queue_size <= 0:
            raise ConfigValidationError("options.max_queue_size must be > 0")
        validate_configs(self.shared_memory, self.buffering)


@dataclass(slots=True)
class SparkWriteOptions:
    """Per-call execution settings for Spark sharded writes."""

    sort_within_partitions: bool = False
    verify_routing: bool = True
    cache_input: bool = False
    storage_level: Any | None = None
    spark_conf_overrides: dict[str, str] | None = None

    def validate(self) -> None:
        return None


@dataclass(slots=True)
class DaskWriteOptions:
    """Per-call execution settings for Dask sharded writes."""

    sort_within_partitions: bool = False
    verify_routing: bool = True

    def validate(self) -> None:
        return None


@dataclass(slots=True)
class RayWriteOptions:
    """Per-call execution settings for Ray sharded writes."""

    sort_within_partitions: bool = False
    verify_routing: bool = True

    def validate(self) -> None:
        return None


@dataclass(slots=True)
class SingleDbWriteOptions:
    """Per-call execution settings for single-database writes."""

    sort_keys: bool = True
    num_partitions: int | None = None
    prefetch_partitions: bool = True
    cache_input: bool = True
    storage_level: Any | None = None
    spark_conf_overrides: dict[str, str] | None = None

    def validate(self) -> None:
        _validate_positive_optional(
            self.num_partitions,
            field_name="options.num_partitions",
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_s3_prefix(s3_prefix: str) -> None:
    parsed = urlparse(s3_prefix)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ConfigValidationError("s3_prefix must be parseable as s3://bucket/prefix")
    if not parsed.path or parsed.path == "/":
        raise ConfigValidationError(
            "s3_prefix must include a non-empty key prefix, e.g. s3://bucket/prefix"
        )


def _validate_segment(value: str, *, field_name: str) -> None:
    if not value:
        raise ConfigValidationError(f"{field_name} must be non-empty")
    if ".." in value:
        raise ConfigValidationError(f"{field_name} must not contain '..'")
    if "/" in value or "\\" in value:
        raise ConfigValidationError(f"{field_name} must be a single path segment")
    if any(ch not in _SAFE_SEGMENT_CHARS for ch in value):
        raise ConfigValidationError(
            f"{field_name} contains unsupported characters; allowed: [A-Za-z0-9._-]"
        )


def _validate_positive_optional(value: float | int | None, *, field_name: str) -> None:
    if value is not None and value <= 0:
        raise ConfigValidationError(f"{field_name} must be > 0 when set")
