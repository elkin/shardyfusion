"""Configuration models for sharded SlateDB writes."""

from dataclasses import dataclass, field
from urllib.parse import urlparse

from .errors import ConfigValidationError
from .manifest import ManifestBuilder
from .publish import ManifestPublisher
from .sharding_types import KeyEncoding, ShardingSpec
from .slatedb_adapter import DbAdapterFactory
from .type_defs import JsonObject, S3ClientConfig

_SAFE_SEGMENT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


@dataclass(slots=True)
class OutputOptions:
    """Output path/layout settings for shard database writes."""

    run_id: str | None = None
    db_path_template: str = "db={db_id:05d}"
    tmp_prefix: str = "_tmp"
    local_root: str = "/tmp/slatedb-spark"


@dataclass(slots=True)
class ManifestOptions:
    """Manifest/CURRENT build and publish settings."""

    manifest_name: str = "manifest"
    current_name: str = "_CURRENT"
    manifest_builder: ManifestBuilder | None = None
    publisher: ManifestPublisher | None = None
    custom_manifest_fields: JsonObject = field(default_factory=dict)
    # Optional default-publisher transport overrides (boto3/Ceph RGW support).
    s3_client_config: S3ClientConfig | None = None


@dataclass(slots=True)
class WriteConfig:
    """Top-level configuration for sharded snapshot writes.

    Framework-specific parameters (``key_col``, ``value_spec``,
    ``sort_within_partitions``) live on the writer function signature,
    not here.
    """

    num_dbs: int
    s3_prefix: str
    key_encoding: KeyEncoding = KeyEncoding.U64BE

    batch_size: int = 50_000
    adapter_factory: DbAdapterFactory | None = None  # None → SlateDbFactory()

    sharding: ShardingSpec = field(default_factory=ShardingSpec)
    output: OutputOptions = field(default_factory=OutputOptions)
    manifest: ManifestOptions = field(default_factory=ManifestOptions)

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

        if self.num_dbs <= 0:
            raise ConfigValidationError("num_dbs must be > 0")
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be > 0")

        _validate_s3_prefix(self.s3_prefix)
        _validate_segment(self.output.tmp_prefix, field_name="output.tmp_prefix")
        _validate_segment(
            self.manifest.manifest_name,
            field_name="manifest.manifest_name",
        )
        _validate_segment(
            self.manifest.current_name,
            field_name="manifest.current_name",
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
