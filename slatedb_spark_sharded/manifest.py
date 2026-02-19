"""Manifest models and extensibility protocols."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Protocol

from .sharding import ShardingSpec
from .type_defs import JsonObject, JsonValue


@dataclass(slots=True)
class RequiredBuildMeta:
    """Library-owned metadata required for all manifests."""

    run_id: str
    created_at: str
    num_dbs: int
    s3_prefix: str
    key_col: str
    sharding: ShardingSpec
    db_path_template: str
    tmp_prefix: str
    format_version: int = 1
    key_encoding: str = "u64be"


@dataclass(slots=True)
class RequiredShardMeta:
    """Winner shard metadata emitted by writer partitions."""

    db_id: int
    db_url: str
    attempt: int
    row_count: int
    min_key: int | str | None
    max_key: int | str | None
    checkpoint_id: str | None
    writer_info: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class ParsedManifest:
    """Typed representation of a parsed manifest payload."""

    required_build: RequiredBuildMeta
    shards: list[RequiredShardMeta]
    custom: JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class ManifestArtifact:
    """Serializable artifact to publish for manifests or pointers."""

    payload: bytes
    content_type: str
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class CurrentPointer:
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
    """Result from `write_sharded_slatedb`."""

    run_id: str
    winners: list[RequiredShardMeta]
    manifest_artifact: ManifestArtifact
    manifest_ref: str
    current_ref: str | None
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
            "required": asdict(required_build),
            "shards": [asdict(shard) for shard in shards],
            "custom": merged_custom,
        }
        payload = json.dumps(
            payload_obj,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return ManifestArtifact(payload=payload, content_type="application/json")


def iso_now() -> str:
    """Return ISO-8601 UTC timestamp suitable for manifests."""

    return datetime.now(timezone.utc).isoformat()
