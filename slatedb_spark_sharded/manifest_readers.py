"""Manifest loading interfaces and default S3-backed implementation."""

import json
from typing import Callable, Protocol

from pydantic import ValidationError

from .errors import ManifestParseError
from .logging import FailureSeverity, log_failure
from .manifest import (
    CurrentPointer,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from .storage import create_s3_client, get_bytes, try_get_bytes
from .type_defs import S3ClientConfig

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
        s3_client_config: S3ClientConfig | None = None,
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
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ManifestParseError(
                "CURRENT pointer payload is not valid JSON or not valid UTF-8"
            ) from exc

        try:
            return CurrentPointer.model_validate(obj)
        except ValidationError as exc:
            raise ManifestParseError(
                f"CURRENT pointer validation failed: {exc}"
            ) from exc

    def load_manifest(
        self, ref: str, content_type: str | None = None
    ) -> ParsedManifest:
        effective_content_type = (
            (content_type or "application/json").split(";", 1)[0].strip()
        )
        if effective_content_type != "application/json":
            raise ManifestParseError(
                "Default reader supports only application/json manifests; "
                "provide a custom ManifestReader."
            )

        try:
            payload = get_bytes(ref, s3_client=self._s3_client)
        except Exception as exc:
            log_failure(
                "manifest_s3_load_failed",
                severity=FailureSeverity.ERROR,
                error=exc,
                manifest_ref=ref,
            )
            raise
        return parse_json_manifest(payload)


def parse_json_manifest(payload: bytes) -> ParsedManifest:
    """Parse default JSON manifest payload into typed ParsedManifest."""

    try:
        obj = json.loads(payload.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        log_failure(
            "manifest_payload_decode_failed",
            severity=FailureSeverity.ERROR,
            error=exc,
            payload_size=len(payload),
        )
        raise ManifestParseError(
            "Manifest payload is not valid JSON or not valid UTF-8"
        ) from exc

    if not isinstance(obj, dict):
        raise ManifestParseError("Manifest payload must be a JSON object")

    required_raw = obj.get("required")
    shards_raw = obj.get("shards")
    custom_raw = obj.get("custom", {})

    if not isinstance(required_raw, dict):
        raise ManifestParseError("Manifest missing required object `required`")
    if not isinstance(shards_raw, list):
        raise ManifestParseError("Manifest missing required array `shards`")
    if not isinstance(custom_raw, dict):
        raise ManifestParseError("Manifest field `custom` must be an object")

    try:
        parsed = ParsedManifest.model_validate(
            {"required_build": required_raw, "shards": shards_raw, "custom": custom_raw}
        )
    except ValidationError as exc:
        raise ManifestParseError(f"Manifest validation failed: {exc}") from exc

    _validate_manifest(parsed.required_build, parsed.shards)
    return parsed


def _validate_manifest(
    required_build: RequiredBuildMeta, shards: list[RequiredShardMeta]
) -> None:
    num_dbs = required_build.num_dbs
    if len(shards) != num_dbs:
        raise ManifestParseError(
            f"Manifest shard count mismatch: expected {num_dbs}, got {len(shards)}"
        )

    ids = sorted(shard.db_id for shard in shards)
    expected = list(range(num_dbs))
    if ids != expected:
        raise ManifestParseError(
            f"Manifest shard coverage mismatch; expected {expected}, got {ids}"
        )

    if not required_build.sharding or not required_build.sharding.strategy:
        raise ManifestParseError("Manifest required.sharding.strategy is missing")
