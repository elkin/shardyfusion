"""Unified manifest persistence protocol and implementations.

ManifestStore replaces the previous ManifestPublisher + ManifestReader
pair with a single interface that accepts Pydantic models directly.
S3ManifestStore implements two-phase publish internally; database
backends can use a single atomic transaction.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

import yaml

if TYPE_CHECKING:
    from .type_defs import RetryConfig

from pydantic import ValidationError

from .credentials import CredentialProvider
from .errors import ManifestParseError
from .logging import FailureSeverity, get_logger, log_failure
from .manifest import (
    CurrentPointer,
    ManifestArtifact,
    ManifestBuilder,
    ManifestRef,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
    YamlManifestBuilder,
)
from .metrics import MetricsCollector
from .storage import (
    create_s3_client,
    get_bytes,
    join_s3,
    list_prefixes,
    put_bytes,
    try_get_bytes,
)
from .type_defs import S3ConnectionOptions

_logger = get_logger(__name__)

# Shared timestamp format for manifest S3 key prefixes.
MANIFEST_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# Regex to parse timestamp-prefixed manifest directory names.
# Example: "2026-03-14T10:30:00.000000Z_run_id=abc123/"
MANIFEST_PREFIX_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z)"
    r"_run_id=(?P<run_id>[^/]+)/$"
)


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
        """Persist manifest + update current pointer. Return a manifest reference."""
        ...

    def load_current(self) -> ManifestRef | None:
        """Return the current manifest pointer, or None if not published."""
        ...

    def load_manifest(self, ref: str) -> ParsedManifest:
        """Load and parse a manifest by reference."""
        ...

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        """Return up to *limit* manifests in reverse chronological order."""
        ...

    def set_current(self, ref: str) -> None:
        """Update the current pointer to the given manifest reference."""
        ...


def _format_manifest_timestamp(dt: datetime) -> str:
    """Format a UTC datetime for use in manifest S3 key prefixes."""
    return dt.strftime(MANIFEST_TIMESTAMP_FMT)


def parse_current_pointer_to_ref(payload: bytes) -> ManifestRef:
    """Parse a raw _CURRENT JSON payload into a ManifestRef."""
    try:
        pointer = CurrentPointer.model_validate_json(payload)
    except ValidationError as exc:
        raise ManifestParseError(f"CURRENT pointer validation failed: {exc}") from exc
    return ManifestRef(
        ref=pointer.manifest_ref,
        run_id=pointer.run_id,
        published_at=pointer.updated_at,
    )


def parse_manifest_dir_entry(
    dir_name: str, s3_prefix: str, manifest_name: str
) -> ManifestRef | None:
    """Parse a timestamp-prefixed directory name into a ManifestRef, or None."""
    match = MANIFEST_PREFIX_RE.match(dir_name)
    if match is None:
        return None
    timestamp_str = match.group("timestamp")
    run_id = match.group("run_id")
    published_at = datetime.strptime(timestamp_str, MANIFEST_TIMESTAMP_FMT).replace(
        tzinfo=UTC
    )
    manifest_url = join_s3(s3_prefix, "manifests", dir_name.rstrip("/"), manifest_name)
    return ManifestRef(ref=manifest_url, run_id=run_id, published_at=published_at)


class S3ManifestStore:
    """Manifest store backed by S3 — merges publish and read in one class.

    Publish is two-phase: manifest file first, then ``_CURRENT`` pointer.
    Manifest keys are timestamp-prefixed for chronological listing.
    """

    def __init__(
        self,
        s3_prefix: str,
        *,
        manifest_name: str = "manifest",
        current_name: str = "_CURRENT",
        manifest_builder: ManifestBuilder | None = None,
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        metrics_collector: MetricsCollector | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        from .type_defs import RetryConfig as _RC

        self.s3_prefix = s3_prefix.rstrip("/")
        self.manifest_name = manifest_name
        self.current_name = current_name
        self._builder = manifest_builder
        credentials = credential_provider.resolve() if credential_provider else None
        self._s3_client = create_s3_client(credentials, s3_connection_options)
        self._metrics = metrics_collector
        self._retry_config: _RC | None = retry_config

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        builder = self._builder or YamlManifestBuilder()
        artifact = builder.build(
            required_build=required_build,
            shards=shards,
            custom_fields=custom,
        )

        timestamp = _format_manifest_timestamp(datetime.now(UTC))
        manifest_url = join_s3(
            self.s3_prefix,
            "manifests",
            f"{timestamp}_run_id={run_id}",
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
        )

        self._write_current(manifest_url, artifact.content_type, run_id)

        return manifest_url

    def load_current(self) -> ManifestRef | None:
        current_url = f"{self.s3_prefix}/{self.current_name}"
        payload = try_get_bytes(
            current_url,
            s3_client=self._s3_client,
            metrics_collector=self._metrics,
            retry_config=self._retry_config,
        )
        if payload is None:
            return None
        return parse_current_pointer_to_ref(payload)

    def load_manifest(self, ref: str) -> ParsedManifest:
        try:
            payload = get_bytes(
                ref,
                s3_client=self._s3_client,
                metrics_collector=self._metrics,
                retry_config=self._retry_config,
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
        return parse_manifest(payload)

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        manifests_prefix = join_s3(self.s3_prefix, "manifests") + "/"
        prefix_urls = list_prefixes(
            manifests_prefix,
            s3_client=self._s3_client,
            retry_config=self._retry_config,
        )

        refs: list[ManifestRef] = []
        for url in prefix_urls:
            # Extract relative dir_name from the full URL
            # e.g. "s3://bucket/prefix/manifests/2026-...Z_run_id=abc/" → "2026-...Z_run_id=abc/"
            dir_name = url[len(manifests_prefix) :]
            ref = parse_manifest_dir_entry(dir_name, self.s3_prefix, self.manifest_name)
            if ref is not None:
                refs.append(ref)

        # list_prefixes returns lexicographic ascending; reverse for newest-first
        refs.sort(key=lambda r: r.published_at, reverse=True)
        return refs[:limit]

    def set_current(self, ref: str) -> None:
        run_id = _extract_run_id_from_ref(ref)
        self._write_current(ref, "application/x-yaml", run_id)

    def _write_current(self, manifest_ref: str, content_type: str, run_id: str) -> None:
        """Write the _CURRENT pointer to S3."""
        current_artifact = _build_current_artifact(
            manifest_ref=manifest_ref,
            manifest_content_type=content_type,
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
        )


class InMemoryManifestStore:
    """In-memory manifest store for tests — replaces InMemoryPublisher."""

    def __init__(self) -> None:
        self._manifests: dict[str, ParsedManifest] = {}
        self._history: list[ManifestRef] = []
        self._current_ref: str | None = None

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
        manifest_ref = ManifestRef(
            ref=ref, run_id=run_id, published_at=datetime.now(UTC)
        )
        self._history.append(manifest_ref)
        self._current_ref = ref
        return ref

    def load_current(self) -> ManifestRef | None:
        if self._current_ref is None:
            return None
        # If set_current was used, find that ref in history
        for entry in reversed(self._history):
            if entry.ref == self._current_ref:
                return entry
        return None

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifests[ref]

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return list(reversed(self._history))[:limit]

    def set_current(self, ref: str) -> None:
        if ref not in self._manifests:
            raise KeyError(f"Manifest not found: {ref}")
        self._current_ref = ref


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_manifest(payload: bytes) -> ParsedManifest:
    """Parse a YAML manifest payload into typed ParsedManifest."""

    try:
        data = yaml.safe_load(payload)
    except Exception as exc:
        raise ManifestParseError(f"Manifest payload is not valid YAML: {exc}") from exc

    try:
        parsed = ParsedManifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestParseError(f"Manifest validation failed: {exc}") from exc

    _validate_manifest(parsed.required_build, parsed.shards)
    return parsed


def _validate_manifest(
    required_build: RequiredBuildMeta, shards: list[RequiredShardMeta]
) -> None:
    num_dbs = required_build.num_dbs
    if len(shards) > num_dbs:
        raise ManifestParseError(
            f"Manifest shard count exceeds num_dbs: {len(shards)} > {num_dbs}"
        )

    ids = [shard.db_id for shard in shards]
    if len(ids) != len(set(ids)):
        raise ManifestParseError(
            f"Manifest contains duplicate shard db_ids: {sorted(ids)}"
        )

    out_of_range = [i for i in ids if i < 0 or i >= num_dbs]
    if out_of_range:
        raise ManifestParseError(
            f"Manifest shard db_ids out of range [0, {num_dbs}): {out_of_range}"
        )

    if not required_build.sharding or not required_build.sharding.strategy:
        raise ManifestParseError("Manifest required.sharding.strategy is missing")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RUN_ID_FROM_REF_RE = re.compile(r"run_id=(?P<run_id>[^/]+)")


def _extract_run_id_from_ref(ref: str) -> str:
    """Extract run_id from a manifest ref path."""
    match = _RUN_ID_FROM_REF_RE.search(ref)
    if match is None:
        raise ValueError(f"Cannot extract run_id from manifest ref: {ref}")
    return match.group("run_id")


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
