"""Unified manifest persistence protocol and implementations.

ManifestStore replaces the previous ManifestPublisher + ManifestReader
pair with a single interface that accepts Pydantic models directly.
S3ManifestStore implements two-phase publish internally; database
backends can use a single atomic transaction.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ._circuit_breaker import CircuitBreaker
    from .type_defs import RetryConfig

from pydantic import ValidationError

from .errors import ManifestParseError
from .logging import FailureSeverity, get_logger, log_failure
from .manifest import (
    CurrentPointer,
    JsonManifestBuilder,
    ManifestArtifact,
    ManifestBuilder,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from .metrics import MetricsCollector
from .storage import create_s3_client, get_bytes, join_s3, put_bytes, try_get_bytes
from .type_defs import S3ClientConfig

_logger = get_logger(__name__)


class ManifestStore(Protocol):
    """Unified protocol for persisting and loading manifests."""

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        """Persist manifest + CURRENT atomically. Return a manifest reference."""
        ...

    def load_current(self) -> CurrentPointer | None:
        """Return the latest CURRENT pointer, or None if not published."""
        ...

    def load_manifest(self, ref: str) -> ParsedManifest:
        """Load and parse a manifest by reference."""
        ...


class S3ManifestStore:
    """Manifest store backed by S3 — merges publish and read in one class.

    Publish is two-phase: manifest file first, then ``_CURRENT`` pointer.
    """

    def __init__(
        self,
        s3_prefix: str,
        *,
        manifest_name: str = "manifest",
        current_name: str = "_CURRENT",
        manifest_builder: ManifestBuilder | None = None,
        s3_client_config: S3ClientConfig | None = None,
        metrics_collector: MetricsCollector | None = None,
        retry_config: RetryConfig | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ) -> None:
        from ._circuit_breaker import CircuitBreaker as _CB
        from .type_defs import RetryConfig as _RC

        self.s3_prefix = s3_prefix.rstrip("/")
        self.manifest_name = manifest_name
        self.current_name = current_name
        self._builder = manifest_builder
        self._s3_client = create_s3_client(s3_client_config)
        self._metrics = metrics_collector
        self._retry_config: _RC | None = retry_config
        self._circuit_breaker: _CB | None = circuit_breaker

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        builder = self._builder or JsonManifestBuilder()
        artifact = builder.build(
            required_build=required_build,
            shards=shards,
            custom_fields=custom,
        )

        manifest_url = join_s3(
            self.s3_prefix,
            "manifests",
            f"run_id={run_id}",
            self.manifest_name,
        )
        put_bytes(
            manifest_url,
            artifact.payload,
            artifact.content_type,
            artifact.headers,
            s3_client=self._s3_client,
            metrics_collector=self._metrics,
            retry_config=self._retry_config,
            circuit_breaker=self._circuit_breaker,
        )

        current_artifact = _build_current_artifact(
            manifest_ref=manifest_url,
            manifest_content_type=artifact.content_type,
            run_id=run_id,
        )
        current_url = join_s3(self.s3_prefix, self.current_name)
        put_bytes(
            current_url,
            current_artifact.payload,
            current_artifact.content_type,
            current_artifact.headers,
            s3_client=self._s3_client,
            metrics_collector=self._metrics,
            retry_config=self._retry_config,
            circuit_breaker=self._circuit_breaker,
        )

        return manifest_url

    def load_current(self) -> CurrentPointer | None:
        current_url = f"{self.s3_prefix}/{self.current_name}"
        payload = try_get_bytes(
            current_url,
            s3_client=self._s3_client,
            metrics_collector=self._metrics,
            retry_config=self._retry_config,
            circuit_breaker=self._circuit_breaker,
        )
        if payload is None:
            return None

        try:
            return CurrentPointer.model_validate_json(payload)
        except ValidationError as exc:
            raise ManifestParseError(
                f"CURRENT pointer validation failed: {exc}"
            ) from exc

    def load_manifest(self, ref: str) -> ParsedManifest:
        try:
            payload = get_bytes(
                ref,
                s3_client=self._s3_client,
                metrics_collector=self._metrics,
                retry_config=self._retry_config,
                circuit_breaker=self._circuit_breaker,
            )
        except Exception as exc:
            log_failure(
                "manifest_s3_load_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                manifest_ref=ref,
            )
            raise
        return parse_json_manifest(payload)


class InMemoryManifestStore:
    """In-memory manifest store for tests — replaces InMemoryPublisher."""

    def __init__(self) -> None:
        self._manifests: dict[str, ParsedManifest] = {}
        self._latest_ref: str | None = None
        self._latest_run_id: str | None = None

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        ref = f"mem://manifests/run_id={run_id}/manifest"
        self._manifests[ref] = ParsedManifest(
            required_build=required_build,
            shards=shards,
            custom=custom,
        )
        self._latest_ref = ref
        self._latest_run_id = run_id
        return ref

    def load_current(self) -> CurrentPointer | None:
        if self._latest_ref is None or self._latest_run_id is None:
            return None
        return CurrentPointer(
            manifest_ref=self._latest_ref,
            manifest_content_type="application/json",
            run_id=self._latest_run_id,
            updated_at=datetime.min.replace(tzinfo=UTC),
        )

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifests[ref]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_json_manifest(payload: bytes) -> ParsedManifest:
    """Parse default JSON manifest payload into typed ParsedManifest."""

    try:
        parsed = ParsedManifest.model_validate_json(payload)
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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_current_artifact(
    *,
    manifest_ref: str,
    manifest_content_type: str,
    run_id: str,
) -> ManifestArtifact:

    pointer = CurrentPointer(
        manifest_ref=manifest_ref,
        manifest_content_type=manifest_content_type,
        run_id=run_id,
        updated_at=datetime.now(UTC),
    )
    payload = json.dumps(
        pointer.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return ManifestArtifact(payload=payload, content_type="application/json")
