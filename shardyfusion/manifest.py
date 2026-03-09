"""Manifest models and extensibility protocols."""

import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .sharding_types import BoundaryValue, KeyEncoding, ShardingStrategy
from .type_defs import JsonObject, JsonValue


class ManifestShardingSpec(BaseModel):
    """Manifest-safe sharding specification (no Callable fields).

    This is the Pydantic counterpart of :class:`ShardingSpec` for
    serialization and deserialization in manifests.  Writer-side
    ``ShardingSpec`` (which carries a ``custom_column_builder`` callable)
    remains a plain dataclass.
    """

    model_config = ConfigDict(use_enum_values=False)

    strategy: ShardingStrategy = ShardingStrategy.HASH
    boundaries: list[BoundaryValue] | None = None
    approx_quantile_rel_error: float = Field(default=0.01, gt=0, lt=1)
    custom_expr: str | None = None


class RequiredBuildMeta(BaseModel):
    """Library-owned metadata required for all manifests."""

    model_config = ConfigDict(use_enum_values=False)

    run_id: str
    created_at: str
    num_dbs: int = Field(gt=0)
    s3_prefix: str
    key_col: str
    sharding: ManifestShardingSpec
    db_path_template: str
    tmp_prefix: str
    format_version: int = 1
    key_encoding: KeyEncoding = KeyEncoding.U64BE


class RequiredShardMeta(BaseModel):
    """Winner shard metadata emitted by writer partitions."""

    db_id: int = Field(ge=0)
    db_url: str
    attempt: int = Field(ge=0)
    row_count: int = Field(ge=0)
    min_key: int | str | None = None
    max_key: int | str | None = None
    checkpoint_id: str | None = None
    writer_info: dict[str, Any] = Field(default_factory=dict)


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
    updated_at: str
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


class JsonManifestBuilder:
    """Default manifest builder emitting JSON."""

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
            "required": required_build.model_dump(mode="json"),
            "shards": [shard.model_dump(mode="json") for shard in shards],
            "custom": merged_custom,
        }
        payload = json.dumps(
            payload_obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return ManifestArtifact(payload=payload, content_type="application/json")
