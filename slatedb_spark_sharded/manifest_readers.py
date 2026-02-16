"""Manifest loading interfaces and default S3-backed implementation."""

from __future__ import annotations

import json
from typing import Any, Callable, Protocol

from .manifest import (
    CurrentPointer,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from .sharding import ShardingSpec, ShardingStrategy
from .storage import create_s3_client, get_bytes, try_get_bytes

ManifestRef = str


class ManifestReader(Protocol):
    """Interface for loading CURRENT and decoding manifest references."""

    def load_current(self) -> CurrentPointer | None:
        """Return CURRENT pointer or None if not present."""
        ...

    def load_manifest(
        self, ref: str, content_type: str | None = None
    ) -> ParsedManifest:
        """Fetch and decode a manifest reference."""
        ...


class FunctionManifestReader:
    """Adapter for user-provided callable loaders."""

    def __init__(
        self,
        load_current_fn: Callable[[], CurrentPointer | None],
        load_manifest_fn: Callable[[str, str | None], ParsedManifest],
    ) -> None:
        self._load_current_fn = load_current_fn
        self._load_manifest_fn = load_manifest_fn

    def load_current(self) -> CurrentPointer | None:
        return self._load_current_fn()

    def load_manifest(
        self, ref: str, content_type: str | None = None
    ) -> ParsedManifest:
        return self._load_manifest_fn(ref, content_type)


class DefaultS3ManifestReader:
    """Default reader for CURRENT + JSON manifests stored on S3."""

    def __init__(
        self,
        s3_prefix: str,
        *,
        current_name: str = "_CURRENT",
        s3_client_config: dict[str, Any] | None = None,
    ) -> None:
        self.s3_prefix = s3_prefix.rstrip("/")
        self.current_name = current_name
        self._s3_client = create_s3_client(s3_client_config)

    def load_current(self) -> CurrentPointer | None:
        current_url = f"{self.s3_prefix}/{self.current_name}"
        payload = try_get_bytes(current_url, s3_client=self._s3_client)
        if payload is None:
            return None

        try:
            obj = json.loads(payload.decode("utf-8"))
        except Exception as exc:
            raise ValueError("CURRENT pointer payload is not valid JSON") from exc

        manifest_ref = obj.get("manifest_ref")
        if not manifest_ref:
            raise ValueError("CURRENT pointer missing required field `manifest_ref`")

        manifest_content_type = obj.get("manifest_content_type")
        if manifest_content_type is None:
            raise ValueError(
                "CURRENT pointer missing required field `manifest_content_type`"
            )

        run_id = obj.get("run_id")
        updated_at = obj.get("updated_at")
        if run_id is None or updated_at is None:
            raise ValueError(
                "CURRENT pointer missing required fields `run_id` or `updated_at`"
            )

        return CurrentPointer(
            manifest_ref=str(manifest_ref),
            manifest_content_type=str(manifest_content_type),
            run_id=str(run_id),
            updated_at=str(updated_at),
            format_version=int(obj.get("format_version", 1)),
        )

    def load_manifest(
        self, ref: str, content_type: str | None = None
    ) -> ParsedManifest:
        effective_content_type = (
            (content_type or "application/json").split(";", 1)[0].strip()
        )
        if effective_content_type != "application/json":
            raise ValueError(
                "Default reader supports only application/json manifests; "
                "provide a custom ManifestReader."
            )

        payload = get_bytes(ref, s3_client=self._s3_client)
        return parse_json_manifest(payload)


def parse_json_manifest(payload: bytes) -> ParsedManifest:
    """Parse default JSON manifest payload into typed ParsedManifest."""

    try:
        obj = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise ValueError("Manifest payload is not valid JSON") from exc

    required_raw = obj.get("required")
    shards_raw = obj.get("shards")
    custom_raw = obj.get("custom", {})

    if not isinstance(required_raw, dict):
        raise ValueError("Manifest missing required object `required`")
    if not isinstance(shards_raw, list):
        raise ValueError("Manifest missing required array `shards`")
    if not isinstance(custom_raw, dict):
        raise ValueError("Manifest field `custom` must be an object")

    sharding_raw = required_raw.get("sharding")
    if not isinstance(sharding_raw, dict):
        raise ValueError("Manifest required metadata missing `sharding` object")
    try:
        strategy = ShardingStrategy.from_value(sharding_raw.get("strategy", "hash"))
    except ValueError as exc:
        raise ValueError("Manifest sharding.strategy is invalid") from exc

    sharding = ShardingSpec(
        strategy=strategy,
        boundaries=sharding_raw.get("boundaries"),
        approx_quantile_rel_error=float(
            sharding_raw.get("approx_quantile_rel_error", 0.01)
        ),
        custom_expr=sharding_raw.get("custom_expr"),
    )

    required_build = RequiredBuildMeta(
        run_id=str(required_raw["run_id"]),
        created_at=str(required_raw["created_at"]),
        num_dbs=int(required_raw["num_dbs"]),
        s3_prefix=str(required_raw["s3_prefix"]),
        key_col=str(required_raw["key_col"]),
        sharding=sharding,
        db_path_template=str(required_raw["db_path_template"]),
        tmp_prefix=str(required_raw["tmp_prefix"]),
        format_version=int(required_raw.get("format_version", 1)),
        key_encoding=str(required_raw.get("key_encoding", "u64be")),
    )

    shards: list[RequiredShardMeta] = []
    for item in shards_raw:
        if not isinstance(item, dict):
            raise ValueError("Manifest shards entries must be objects")
        shard = RequiredShardMeta(
            db_id=int(item["db_id"]),
            db_url=str(item["db_url"]),
            attempt=int(item["attempt"]),
            row_count=int(item["row_count"]),
            min_key=item.get("min_key"),
            max_key=item.get("max_key"),
            checkpoint_id=(
                None
                if item.get("checkpoint_id") is None
                else str(item.get("checkpoint_id"))
            ),
            writer_info=dict(item.get("writer_info") or {}),
        )
        shards.append(shard)

    _validate_manifest(required_build, shards)
    return ParsedManifest(
        required_build=required_build, shards=shards, custom=custom_raw
    )


def _validate_manifest(
    required_build: RequiredBuildMeta, shards: list[RequiredShardMeta]
) -> None:
    num_dbs = required_build.num_dbs
    if num_dbs <= 0:
        raise ValueError("Manifest required.num_dbs must be > 0")
    if len(shards) != num_dbs:
        raise ValueError(
            f"Manifest shard count mismatch: expected {num_dbs}, got {len(shards)}"
        )

    ids = sorted(shard.db_id for shard in shards)
    expected = list(range(num_dbs))
    if ids != expected:
        raise ValueError(
            f"Manifest shard coverage mismatch; expected {expected}, got {ids}"
        )

    if not required_build.sharding or not required_build.sharding.strategy:
        raise ValueError("Manifest required.sharding.strategy is missing")
