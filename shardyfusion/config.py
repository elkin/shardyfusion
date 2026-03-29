"""Configuration models for sharded snapshot writes."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from .credentials import CredentialProvider
from .errors import ConfigValidationError
from .manifest import ManifestBuilder
from .metrics import MetricsCollector
from .sharding_types import KeyEncoding, ShardingSpec, ShardingStrategy
from .slatedb_adapter import DbAdapterFactory
from .type_defs import JsonObject, RetryConfig, S3ConnectionOptions

if TYPE_CHECKING:
    from .manifest_store import ManifestStore
    from .run_registry import RunRegistry

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

    manifest_builder: ManifestBuilder | None = None
    store: ManifestStore | None = None
    custom_manifest_fields: JsonObject = field(default_factory=dict)
    credential_provider: CredentialProvider | None = None
    s3_connection_options: S3ConnectionOptions | None = None


@dataclass(slots=True)
class WriteConfig:
    """Top-level configuration for sharded snapshot writes.

    Framework-specific parameters (``key_col``, ``value_spec``,
    ``sort_within_partitions``) live on the writer function signature,
    not here.

    Args:
        num_dbs: Number of shard databases to create. Must be > 0 for
            explicit HASH sharding. Set to ``None`` (default) when using
            CEL sharding or ``max_keys_per_shard`` (auto-discovered at
            write time).
        s3_prefix: S3 location for shard databases and manifests
            (e.g. ``s3://bucket/prefix``).
        key_encoding: How keys are serialized to bytes. Default ``u64be``
            (8-byte big-endian). Use ``u32be`` for 4-byte keys.
        batch_size: Number of key-value pairs per write batch. Default 50,000.
        adapter_factory: Factory for creating shard database adapters.
            Default: ``SlateDbFactory()``.
        sharding: Sharding strategy configuration (hash, range, or custom).
        output: Output path/layout settings.
        manifest: Manifest build and publish settings.
        metrics_collector: Optional observer for write lifecycle events.
        shard_retry: Optional retry configuration. Used for per-shard retry in
            Dask/Ray sharded writes, whole-database retry in Spark/Dask/Ray
            `write_single_db()`, and retry-enabled Python parallel writes.
        credential_provider: Credential provider for S3 access. When also
            set on ``manifest``, the manifest-level provider takes precedence.
        s3_connection_options: S3 transport/connection overrides. When also
            set on ``manifest``, the manifest-level options take precedence.

    Raises:
        ConfigValidationError: If any parameter fails validation.
    """

    num_dbs: int | None = None
    s3_prefix: str = ""
    key_encoding: KeyEncoding = KeyEncoding.U64BE

    batch_size: int = 50_000
    adapter_factory: DbAdapterFactory | None = None  # None → SlateDbFactory()

    sharding: ShardingSpec = field(default_factory=ShardingSpec)
    output: OutputOptions = field(default_factory=OutputOptions)
    manifest: ManifestOptions = field(default_factory=ManifestOptions)

    metrics_collector: MetricsCollector | None = None
    run_registry: RunRegistry | None = None

    shard_retry: RetryConfig | None = None

    credential_provider: CredentialProvider | None = None
    s3_connection_options: S3ConnectionOptions | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sharding, ShardingSpec):
            raise ConfigValidationError("sharding must be ShardingSpec")
        if not isinstance(self.output, OutputOptions):
            raise ConfigValidationError("output must be OutputOptions")
        if not isinstance(self.manifest, ManifestOptions):
            raise ConfigValidationError("manifest must be ManifestOptions")

        if not isinstance(self.key_encoding, KeyEncoding):
            try:
                self.key_encoding = KeyEncoding.from_value(self.key_encoding)
            except ValueError as exc:
                raise ConfigValidationError(str(exc)) from exc

        if self.sharding.strategy == ShardingStrategy.CEL:
            # CEL: num_dbs is derived from routing metadata or discovered from data.
            if self.num_dbs is not None:
                raise ConfigValidationError(
                    "num_dbs must be None for CEL strategy "
                    "(shard count is determined by the CEL expression)"
                )
        elif self.sharding.max_keys_per_shard is not None:
            # HASH + max_keys_per_shard: num_dbs computed at write time
            if self.num_dbs is not None:
                raise ConfigValidationError(
                    "num_dbs must be None when max_keys_per_shard is set "
                    "(num_dbs will be computed at write time)"
                )
        elif self.num_dbs is None or self.num_dbs <= 0:
            raise ConfigValidationError("num_dbs must be > 0")

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
