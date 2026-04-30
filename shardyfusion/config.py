"""Configuration models for sharded snapshot writes."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias
from urllib.parse import urlparse

from .credentials import CredentialProvider
from .errors import ConfigValidationError
from .metrics import MetricsCollector
from .sharding_types import (
    KeyEncoding,
    RoutingValue,
    validate_routing_values,
)
from .slatedb_adapter import DbAdapterFactory
from .type_defs import JsonObject, RetryConfig, S3ConnectionOptions

if TYPE_CHECKING:
    from .manifest_store import ManifestStore
    from .run_registry import RunRegistry
    from .vector.config import VectorIndexConfig, VectorShardingSpec, VectorSpecSharding

_SAFE_SEGMENT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


@dataclass(slots=True)
class OutputOptions:
    """Output path/layout settings for shard database writes."""

    run_id: str | None = None
    db_path_template: str = "db={db_id:05d}"
    shard_prefix: str = "shards"
    run_registry_prefix: str = "runs"
    local_root: str = field(
        default_factory=lambda: str(Path(tempfile.gettempdir()) / "shardyfusion")
    )


@dataclass(slots=True)
class ManifestOptions:
    """Manifest build and publish settings."""

    store: ManifestStore | None = None
    custom_manifest_fields: JsonObject = field(default_factory=dict)
    credential_provider: CredentialProvider | None = None
    s3_connection_options: S3ConnectionOptions | None = None


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

    Enables unified KV + vector search mode when set on ``WriteConfig``.
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


@dataclass(slots=True)
class WriteConfig:
    """Base configuration for sharded snapshot writes.

    Framework-specific parameters (``key_col``, ``value_spec``,
    ``sort_within_partitions``) live on the writer function signature,
    not here.

    Args:
        s3_prefix: S3 location for shard databases and manifests
            (e.g. ``s3://bucket/prefix``).
        key_encoding: How keys are serialized to bytes. Default ``u64be``
            (8-byte big-endian). Use ``u32be`` for 4-byte keys.
        batch_size: Number of key-value pairs per write batch. Default 50,000.
        adapter_factory: Factory for creating shard database adapters.
            Default: ``SlateDbFactory()``.
        output: Output path/layout settings.
        manifest: Manifest build and publish settings.
        metrics_collector: Optional observer for write lifecycle events.
        shard_retry: Optional retry configuration. Used for per-shard retry in
            Dask/Ray sharded writes, whole-database retry in Spark/Dask/Ray
            ``write_single_db()``, and retry-enabled Python parallel writes.
        credential_provider: Writer-level credential provider for S3 access.
            Shard adapters and the default run registry use this directly. The
            default S3 manifest store uses ``manifest.credential_provider``
            when provided, otherwise this provider.
        s3_connection_options: Writer-level S3 transport/connection overrides.
            Shard adapters and the default run registry use this directly.
            The default S3 manifest store uses
            ``manifest.s3_connection_options`` when provided, otherwise these
            options.

    Raises:
        ConfigValidationError: If any parameter fails validation.
    """

    s3_prefix: str = ""
    key_encoding: KeyEncoding = KeyEncoding.U64BE

    batch_size: int = 50_000
    adapter_factory: DbAdapterFactory | None = None  # None → SlateDbFactory()

    output: OutputOptions = field(default_factory=OutputOptions)
    manifest: ManifestOptions = field(default_factory=ManifestOptions)

    metrics_collector: MetricsCollector | None = None
    run_registry: RunRegistry | None = None

    shard_retry: RetryConfig | None = None

    credential_provider: CredentialProvider | None = None
    s3_connection_options: S3ConnectionOptions | None = None

    vector_spec: VectorSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output, OutputOptions):
            raise ConfigValidationError("output must be OutputOptions")
        if not isinstance(self.manifest, ManifestOptions):
            raise ConfigValidationError("manifest must be ManifestOptions")

        if not isinstance(self.key_encoding, KeyEncoding):
            try:
                self.key_encoding = KeyEncoding.from_value(self.key_encoding)
            except ValueError as exc:
                raise ConfigValidationError(str(exc)) from exc

        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be > 0")

        _validate_s3_prefix(self.s3_prefix)
        _validate_segment(self.output.shard_prefix, field_name="output.shard_prefix")
        _validate_segment(
            self.output.run_registry_prefix,
            field_name="output.run_registry_prefix",
        )

        try:
            self.output.db_path_template.format(db_id=0)
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:  # pragma: no cover - defensive formatting surface
            raise ConfigValidationError(
                "output.db_path_template must support format(db_id=...)"
            ) from exc

        if self.vector_spec is not None:
            vs = self.vector_spec
            if vs.dim <= 0:
                raise ConfigValidationError(
                    f"vector_spec.dim must be > 0, got {vs.dim}"
                )
            vs.metric = _coerce_vector_metric(vs.metric)


@dataclass(slots=True)
class HashWriteConfig(WriteConfig):
    """Write configuration for HASH sharding.

    Args:
        num_dbs: Number of shard databases to create. Must be > 0 unless
            ``max_keys_per_shard`` is set (in which case it is computed at
            write time).
        max_keys_per_shard: Alternative to ``num_dbs`` — computes shard
            count as ``ceil(total_rows / max_keys_per_shard)``.
    """

    num_dbs: int | None = None
    max_keys_per_shard: int | None = None

    def __post_init__(self) -> None:
        WriteConfig.__post_init__(self)
        if self.max_keys_per_shard is not None:
            if self.num_dbs is not None:
                raise ConfigValidationError(
                    "num_dbs must be None when max_keys_per_shard is set "
                    "(num_dbs will be computed at write time)"
                )
            if self.max_keys_per_shard <= 0:
                raise ConfigValidationError("max_keys_per_shard must be > 0")
        elif self.num_dbs is None or self.num_dbs <= 0:
            raise ConfigValidationError("num_dbs must be > 0")


@dataclass(slots=True)
class CelWriteConfig(WriteConfig):
    """Write configuration for CEL sharding.

    Args:
        cel_expr: CEL expression that produces a shard ID or categorical token.
        cel_columns: Mapping of CEL variable names to their types
            (e.g. ``{"key": "int"}``).
        routing_values: Optional categorical values for token-based routing.
        infer_routing_values_from_data: If True, discover routing values from
            the input data at write time.
    """

    cel_expr: str = ""
    cel_columns: dict[str, str] = field(default_factory=dict)
    routing_values: list[RoutingValue] | None = None
    infer_routing_values_from_data: bool = False

    def __post_init__(self) -> None:
        WriteConfig.__post_init__(self)
        if not self.cel_expr:
            raise ConfigValidationError("CEL strategy requires cel_expr")
        if not self.cel_columns:
            raise ConfigValidationError("CEL strategy requires cel_columns")
        if self.infer_routing_values_from_data:
            if self.routing_values is not None:
                raise ConfigValidationError(
                    "infer_routing_values_from_data cannot be combined with "
                    "routing_values"
                )
        if self.routing_values is not None:
            try:
                validate_routing_values(self.routing_values)
            except ValueError as exc:
                raise ConfigValidationError(str(exc)) from exc


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
