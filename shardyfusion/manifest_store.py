"""Unified manifest persistence protocol and implementations.

ManifestStore replaces the previous ManifestPublisher + ManifestReader
pair with a single interface that accepts Pydantic models directly.
S3ManifestStore implements two-phase publish internally; database
backends can use a single atomic transaction.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import ValidationError

from .errors import ManifestParseError
from .logging import FailureSeverity, get_logger, log_failure
from .manifest import (
    SQLITE_MANIFEST_CONTENT_TYPE,
    CurrentPointer,
    ManifestArtifact,
    ManifestRef,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
    SqliteManifestBuilder,
)
from .metrics import MetricsCollector
from .sharding_types import ShardHashAlgorithm
from .storage import StorageBackend, join_s3

_logger = get_logger(__name__)
_SUPPORTED_MANIFEST_FORMAT_VERSIONS = frozenset({4})

# Shared timestamp format for manifest S3 key prefixes.
MANIFEST_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"

# Regex to parse timestamp-prefixed manifest directory names.
# Example: "2026-03-14T10:30:00.000000Z_run_id=abc123/"
# obstore list_with_delimiter omits the trailing slash, so make it optional.
MANIFEST_PREFIX_RE = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z)"
    r"_run_id=(?P<run_id>[^/]+)/?$"
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
        backend: StorageBackend,
        s3_prefix: str,
        *,
        manifest_name: str = "manifest",
        current_pointer_key: str = "_CURRENT",
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        self._backend = backend
        self.s3_prefix = s3_prefix.rstrip("/")
        self.manifest_name = manifest_name
        self.current_pointer_key = current_pointer_key
        self._metrics = metrics_collector

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        builder = SqliteManifestBuilder()
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
        self._backend.put(
            manifest_url,
            artifact.payload,
            artifact.content_type,
            artifact.headers,
        )

        self._write_current(manifest_url, artifact.content_type, run_id)

        return manifest_url

    def load_current(self) -> ManifestRef | None:
        current_url = f"{self.s3_prefix}/{self.current_pointer_key}"
        payload = self._backend.try_get(current_url)
        if payload is None:
            return None
        return parse_current_pointer_to_ref(payload)

    def load_manifest(self, ref: str) -> ParsedManifest:
        try:
            payload = self._backend.get(ref)
        except Exception as exc:
            log_failure(
                "manifest_s3_load_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                manifest_ref=ref,
            )
            raise
        return parse_manifest_payload(payload)

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        manifests_prefix = join_s3(self.s3_prefix, "manifests") + "/"
        prefix_urls = self._backend.list_prefixes(manifests_prefix)

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

    def set_current(
        self, ref: str, *, content_type: str = SQLITE_MANIFEST_CONTENT_TYPE
    ) -> None:
        run_id = _extract_run_id_from_ref(ref)
        self._write_current(ref, content_type, run_id)

    def _write_current(self, manifest_ref: str, content_type: str, run_id: str) -> None:
        """Write the _CURRENT pointer to S3."""
        current_artifact = _build_current_artifact(
            manifest_ref=manifest_ref,
            manifest_content_type=content_type,
            run_id=run_id,
        )
        current_url = join_s3(self.s3_prefix, self.current_pointer_key)
        self._backend.put(
            current_url,
            current_artifact.payload,
            current_artifact.content_type,
            current_artifact.headers,
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

# First 16 bytes of every SQLite database file.
_SQLITE_MAGIC = b"SQLite format 3\x00"


def parse_manifest_payload(payload: bytes) -> ParsedManifest:
    """Parse a manifest payload.

    The payload must be a serialized SQLite database (starts with the
    16-byte SQLite magic header).  Raises :class:`ManifestParseError`
    for unrecognised formats.
    """
    if payload[:16] != _SQLITE_MAGIC:
        raise ManifestParseError(
            "Unsupported manifest format: expected SQLite (magic header not found)"
        )
    return parse_sqlite_manifest(payload)


def parse_sqlite_manifest(payload: bytes) -> ParsedManifest:
    """Parse a SQLite manifest payload into typed ParsedManifest.

    Opens the serialized SQLite database in memory via
    :meth:`sqlite3.Connection.deserialize`, reads ``build_meta`` and
    ``shards`` tables, and validates identically to :func:`parse_manifest`.
    """
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    try:
        try:
            con.deserialize(payload)
        except Exception as exc:
            raise ManifestParseError(
                f"Manifest payload is not a valid SQLite database: {exc}"
            ) from exc

        row = con.execute("SELECT * FROM build_meta LIMIT 1").fetchone()
        if row is None:
            raise ManifestParseError("SQLite manifest has no build_meta row")

        build_data = {
            "run_id": row["run_id"],
            "created_at": row["created_at"],
            "num_dbs": row["num_dbs"],
            "s3_prefix": row["s3_prefix"],
            "key_col": row["key_col"],
            "sharding": json.loads(row["sharding"]),
            "db_path_template": row["db_path_template"],
            "shard_prefix": row["shard_prefix"],
            "format_version": row["format_version"],
            "key_encoding": row["key_encoding"],
        }
        custom = json.loads(row["custom"]) if row["custom"] else {}

        shard_rows = con.execute("SELECT * FROM shards ORDER BY db_id").fetchall()

        shards_data = []
        for sr in shard_rows:
            shards_data.append(
                {
                    "db_id": sr["db_id"],
                    "db_url": sr["db_url"],
                    "attempt": sr["attempt"],
                    "row_count": sr["row_count"],
                    "db_bytes": sr["db_bytes"],
                    "min_key": json.loads(sr["min_key"])
                    if sr["min_key"] is not None
                    else None,
                    "max_key": json.loads(sr["max_key"])
                    if sr["max_key"] is not None
                    else None,
                    "checkpoint_id": sr["checkpoint_id"],
                    "writer_info": json.loads(sr["writer_info"]),
                }
            )
    except ManifestParseError:
        raise
    except Exception as exc:
        raise ManifestParseError(
            f"Failed to read SQLite manifest tables: {exc}"
        ) from exc
    finally:
        con.close()

    data = {"required": build_data, "shards": shards_data, "custom": custom}

    try:
        parsed = ParsedManifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestParseError(f"Manifest validation failed: {exc}") from exc

    _validate_manifest(parsed.required_build, parsed.shards)
    return parsed


def _validate_manifest(
    required_build: RequiredBuildMeta, shards: list[RequiredShardMeta]
) -> None:
    if required_build.format_version not in _SUPPORTED_MANIFEST_FORMAT_VERSIONS:
        raise ManifestParseError(
            "Unsupported manifest format_version: "
            f"{required_build.format_version}. "
            f"Supported: {sorted(_SUPPORTED_MANIFEST_FORMAT_VERSIONS)}"
        )

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

    if required_build.sharding.hash_algorithm is None:
        raise ManifestParseError("Manifest required.sharding.hash_algorithm is missing")
    try:
        ShardHashAlgorithm.from_value(required_build.sharding.hash_algorithm)
    except ValueError as exc:
        raise ManifestParseError(str(exc)) from exc

    routing_values = required_build.sharding.routing_values
    if routing_values is not None:
        expected_num_dbs = max(1, len(routing_values))
        if num_dbs != expected_num_dbs:
            raise ManifestParseError(
                "Categorical CEL manifest num_dbs must match routing_values cardinality "
                f"(expected {expected_num_dbs}, got {num_dbs})"
            )


# ---------------------------------------------------------------------------
# SQLite manifest — granular access helpers
# ---------------------------------------------------------------------------


def load_sqlite_build_meta(
    con: sqlite3.Connection,
) -> tuple[RequiredBuildMeta, dict[str, Any]]:
    """Read ``RequiredBuildMeta`` and custom fields from an open SQLite manifest.

    Unlike :func:`parse_sqlite_manifest` (which takes raw bytes and loads
    all shards), this reads only the ``build_meta`` row — a single small
    query, ideal for lazy-router initialization.

    Returns ``(required_build, custom_fields)``.
    """
    old_factory = con.row_factory
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT * FROM build_meta LIMIT 1").fetchone()
    finally:
        con.row_factory = old_factory
    if row is None:
        raise ManifestParseError("SQLite manifest has no build_meta row")

    build_data = {
        "run_id": row["run_id"],
        "created_at": row["created_at"],
        "num_dbs": row["num_dbs"],
        "s3_prefix": row["s3_prefix"],
        "key_col": row["key_col"],
        "sharding": json.loads(row["sharding"]),
        "db_path_template": row["db_path_template"],
        "shard_prefix": row["shard_prefix"],
        "format_version": row["format_version"],
        "key_encoding": row["key_encoding"],
    }
    custom = json.loads(row["custom"]) if row["custom"] else {}

    try:
        required_build = RequiredBuildMeta.model_validate(build_data)
    except ValidationError as exc:
        raise ManifestParseError(
            f"SQLite manifest build_meta validation failed: {exc}"
        ) from exc
    _validate_manifest(required_build, [])
    return required_build, custom


class SqliteShardLookup:
    """Lazy shard metadata provider backed by a SQLite manifest connection.

    Satisfies the :class:`~shardyfusion.routing.ShardLookup` protocol.
    Queries the ``shards`` table by ``db_id`` (B-tree index) and caches
    results in an LRU dict with bounded size.

    Parameters
    ----------
    con:
        An open ``sqlite3.Connection`` to a SQLite manifest database.
        The caller owns the connection lifetime; this class does **not**
        close it.
    num_dbs:
        Total number of shards (from ``RequiredBuildMeta.num_dbs``).
    cache_size:
        Maximum number of shard entries to cache.  0 disables caching.
    """

    def __init__(
        self,
        con: sqlite3.Connection,
        num_dbs: int,
        *,
        cache_size: int = 4096,
    ) -> None:
        self._con = con
        self._num_dbs = num_dbs
        self._cache_size = cache_size
        self._cache: OrderedDict[int, RequiredShardMeta] = OrderedDict()

    def get_shard(self, db_id: int) -> RequiredShardMeta:
        """Return shard metadata for *db_id*, or a synthetic empty entry."""
        if self._cache_size > 0:
            cached = self._cache.get(db_id)
            if cached is not None:
                self._cache.move_to_end(db_id)
                return cached

        shard = self._query_shard(db_id)

        if self._cache_size > 0:
            if len(self._cache) >= self._cache_size:
                self._cache.popitem(last=False)
            self._cache[db_id] = shard
        return shard

    def _query_shard(self, db_id: int) -> RequiredShardMeta:
        old_factory = self._con.row_factory
        self._con.row_factory = sqlite3.Row
        try:
            row = self._con.execute(
                "SELECT * FROM shards WHERE db_id = ?", (db_id,)
            ).fetchone()
        finally:
            self._con.row_factory = old_factory

        if row is None:
            return RequiredShardMeta(db_id=db_id, attempt=0, row_count=0, db_bytes=0)

        return RequiredShardMeta.model_validate(
            {
                "db_id": row["db_id"],
                "db_url": row["db_url"],
                "attempt": row["attempt"],
                "row_count": row["row_count"],
                "db_bytes": row["db_bytes"],
                "min_key": json.loads(row["min_key"])
                if row["min_key"] is not None
                else None,
                "max_key": json.loads(row["max_key"])
                if row["max_key"] is not None
                else None,
                "checkpoint_id": row["checkpoint_id"],
                "writer_info": json.loads(row["writer_info"]),
            }
        )


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
