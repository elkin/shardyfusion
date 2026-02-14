"""Configuration models for sharded SlateDB writes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

from .errors import ConfigValidationError
from .serde import ValueSpec
from .sharding import ShardingSpec

if TYPE_CHECKING:
    from .manifest import ManifestBuilder
    from .publish import ManifestPublisher
    from .slatedb_adapter import SlateDbAdapter


_SAFE_SEGMENT_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


@dataclass(slots=True)
class SlateDbConfig:
    """Top-level configuration for `write_sharded_slatedb`."""

    num_dbs: int
    s3_prefix: str
    key_col: str
    value_spec: ValueSpec

    sharding: ShardingSpec = field(default_factory=ShardingSpec)
    sort_within_partitions: bool = False

    run_id: str | None = None
    db_path_template: str = "db={db_id:05d}"
    tmp_prefix: str = "_tmp"
    local_root: str = "/tmp/slatedb-spark"

    manifest_name: str = "manifest"
    current_name: str = "_CURRENT"
    manifest_builder: ManifestBuilder | None = None
    publisher: ManifestPublisher | None = None
    custom_manifest_fields: dict[str, Any] = field(default_factory=dict)

    slate_env_file: str | None = None
    slate_settings: dict[str, Any] | None = None
    batch_size: int = 50_000

    spark_conf_overrides: dict[str, str] | None = None

    # Optional default-publisher transport overrides (boto3/Ceph RGW support).
    s3_client_config: dict[str, Any] | None = None

    # Advanced/testing hook for injecting an adapter implementation.
    slatedb_adapter_factory: Callable[[], SlateDbAdapter] | None = None

    def __post_init__(self) -> None:
        if self.num_dbs <= 0:
            raise ConfigValidationError("num_dbs must be > 0")
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be > 0")

        _validate_s3_prefix(self.s3_prefix)
        _validate_segment(self.tmp_prefix, field_name="tmp_prefix")
        _validate_segment(self.manifest_name, field_name="manifest_name")
        _validate_segment(self.current_name, field_name="current_name")

        try:
            self.db_path_template.format(db_id=0)
        except Exception as exc:  # pragma: no cover - defensive formatting surface
            raise ConfigValidationError(
                "db_path_template must support format(db_id=...)"
            ) from exc


def _validate_s3_prefix(s3_prefix: str) -> None:
    parsed = urlparse(s3_prefix)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ConfigValidationError(
            "s3_prefix must be parseable as s3://bucket/prefix"
        )
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
