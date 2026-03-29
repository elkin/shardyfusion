"""Manifest models and extensibility protocols."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .sharding_types import (
    KeyEncoding,
    RoutingValue,
    ShardingStrategy,
    validate_routing_values,
)
from .type_defs import JsonObject, JsonValue


@dataclass(slots=True, frozen=True)
class ManifestRef:
    """Backend-agnostic pointer to a published manifest."""

    ref: str
    run_id: str
    published_at: datetime


class ManifestShardingSpec(BaseModel):
    """Manifest-safe sharding specification (no Callable fields).

    This is the Pydantic counterpart of :class:`ShardingSpec` for
    serialization and deserialization in manifests.
    """

    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    strategy: ShardingStrategy = ShardingStrategy.HASH
    routing_values: list[RoutingValue] | None = None
    cel_expr: str | None = None
    cel_columns: dict[str, str] | None = None
    hash_algorithm: str = "xxh3_64"

    @model_validator(mode="after")
    def _validate_model(self) -> "ManifestShardingSpec":
        if self.strategy == ShardingStrategy.CEL:
            if not self.cel_expr:
                raise ValueError("CEL strategy requires cel_expr")
            if not self.cel_columns:
                raise ValueError("CEL strategy requires cel_columns")
            if self.routing_values is not None:
                validate_routing_values(self.routing_values)
        else:
            if self.routing_values is not None:
                raise ValueError("routing_values are only valid with CEL strategy")
            if self.cel_expr is not None:
                raise ValueError("cel_expr is only valid with CEL strategy")
            if self.cel_columns is not None:
                raise ValueError("cel_columns are only valid with CEL strategy")
        return self


class RequiredBuildMeta(BaseModel):
    """Library-owned metadata required for all manifests."""

    model_config = ConfigDict(use_enum_values=False)

    run_id: str
    created_at: datetime
    num_dbs: int = Field(gt=0)
    s3_prefix: str
    key_col: str
    sharding: ManifestShardingSpec
    db_path_template: str
    shard_prefix: str
    format_version: int = 2
    key_encoding: KeyEncoding = KeyEncoding.U64BE


@dataclass(slots=True, frozen=True)
class WriterInfo:
    """Per-attempt metadata from the writer framework."""

    stage_id: int | None = None
    task_attempt_id: int | None = None
    attempt: int = 0
    duration_ms: int = 0


class RequiredShardMeta(BaseModel):
    """Winner shard metadata emitted by writer partitions."""

    db_id: int = Field(ge=0)
    db_url: str | None = None
    attempt: int = Field(ge=0)
    row_count: int = Field(ge=0)
    min_key: int | str | bytes | None = None
    max_key: int | str | bytes | None = None
    checkpoint_id: str | None = None
    writer_info: WriterInfo = Field(default_factory=WriterInfo)


class ParsedManifest(BaseModel):
    """Typed representation of a parsed manifest payload."""

    model_config = ConfigDict(populate_by_name=True)

    required_build: RequiredBuildMeta = Field(
        validation_alias="required",
        serialization_alias="required",
    )
    shards: list[RequiredShardMeta]
    custom: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class ManifestArtifact:
    """Serializable artifact to publish for manifests or pointers."""

    payload: bytes
    content_type: str
    headers: dict[str, str] = field(default_factory=dict)


class CurrentPointer(BaseModel):
    """JSON pointer to the latest published manifest."""

    manifest_ref: str
    manifest_content_type: str
    run_id: str
    updated_at: datetime
    format_version: int = 1


@dataclass(slots=True)
class BuildDurations:
    """Measured phase durations in milliseconds."""

    sharding_ms: int
    write_ms: int
    manifest_ms: int
    total_ms: int


@dataclass(slots=True)
class BuildStats:
    """Deterministic build statistics with fixed schema."""

    durations: BuildDurations
    num_attempt_results: int
    num_winners: int
    rows_written: int


@dataclass(slots=True)
class BuildResult:
    """Result from `write_sharded`."""

    run_id: str
    winners: list[RequiredShardMeta]
    manifest_ref: str
    stats: BuildStats
    run_record_ref: str | None = None


class ManifestBuilder(Protocol):
    """Protocol for custom manifest formats."""

    def add_custom_field(self, key: str, value: JsonValue) -> None:
        """Register one custom field before build."""
        ...

    def build(
        self,
        *,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom_fields: JsonObject,
    ) -> ManifestArtifact:
        """Build a manifest artifact containing required metadata."""
        ...


class YamlManifestBuilder:
    """Default manifest builder emitting YAML.

    Uses ``yaml.safe_dump`` for native ``bytes`` support (``!!binary``),
    human-readable output, and deterministic key ordering.
    """

    def __init__(self) -> None:
        self._custom_fields: JsonObject = {}

    def add_custom_field(self, key: str, value: JsonValue) -> None:
        self._custom_fields[key] = value

    def build(
        self,
        *,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom_fields: JsonObject,
    ) -> ManifestArtifact:

        merged_custom = dict(self._custom_fields)
        merged_custom.update(custom_fields)

        payload_obj = {
            "required": required_build.model_dump(mode="json", exclude_none=True),
            "shards": [
                shard.model_dump(mode="json", exclude_none=True) for shard in shards
            ],
            "custom": merged_custom,
        }
        payload = yaml.safe_dump(
            payload_obj,
            sort_keys=True,
            default_flow_style=False,
        ).encode("utf-8")
        return ManifestArtifact(payload=payload, content_type="application/x-yaml")
