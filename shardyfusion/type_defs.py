"""Shared type aliases and protocols used across the package."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Protocol, TypeAlias, TypedDict

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

KeyInput: TypeAlias = int | str | bytes


class ShardSizeMeta(Protocol):
    """Minimal per-shard metadata exposed to snapshot-aware factories.

    Concrete implementation is :class:`shardyfusion.manifest.RequiredShardMeta`,
    which duck-types as this Protocol via attribute matching.
    """

    @property
    def db_url(self) -> str | None: ...

    @property
    def db_bytes(self) -> int: ...


class BuildMeta(Protocol):
    """Minimal manifest build-meta exposed to snapshot-aware factories."""

    @property
    def run_id(self) -> str: ...


class Manifest(Protocol):
    """Minimal manifest shape exposed to snapshot-aware reader factories.

    The concrete implementation is :class:`shardyfusion.manifest.ParsedManifest`,
    which duck-types as this Protocol via attribute matching. Defined here
    (rather than imported from ``manifest.py``) to keep ``type_defs`` at the
    base of the dependency graph.
    """

    @property
    def required_build(self) -> BuildMeta: ...

    @property
    def shards(self) -> Sequence[ShardSizeMeta]: ...


class ShardReader(Protocol):
    """Minimal shard reader shape used by the sharded reader service."""

    def get(self, key: bytes) -> bytes | None:
        """Return value bytes for key, or None when absent."""
        ...

    def close(self) -> None:
        """Release reader resources."""
        ...


class ShardReaderFactory(Protocol):
    """Factory for opening one shard reader.

    The ``manifest`` argument carries the parsed manifest for the snapshot
    being opened, allowing snapshot-aware factories (e.g.
    :class:`shardyfusion.sqlite_adapter.AdaptiveSqliteReaderFactory`) to
    pick a backend tier from the shard size distribution. Concrete
    factories that do not need it should accept and ignore the argument.
    """

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> ShardReader:
        """Construct an opened reader instance."""
        ...


class AsyncShardReader(Protocol):
    """Async counterpart of ShardReader for use with AsyncShardedReader."""

    async def get(self, key: bytes) -> bytes | None: ...

    async def close(self) -> None: ...


class AsyncShardReaderFactory(Protocol):
    """Factory for opening one async shard reader.

    See :class:`ShardReaderFactory` for the role of ``manifest``.
    """

    async def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest: Manifest,
    ) -> AsyncShardReader: ...


class S3ConnectionOptions(TypedDict, total=False):
    """Transport/connection overrides for boto3 S3 client construction.

    Identity fields (access key, secret, session token) are handled
    separately by :class:`~shardyfusion.credentials.CredentialProvider`.
    """

    endpoint_url: str
    region_name: str
    # botocore / S3-connection options
    addressing_style: str  # "virtual" | "path" | "auto"
    signature_version: str  # "s3v4" | "s3"
    verify_ssl: bool | str  # bool or path to CA bundle
    connect_timeout: int  # seconds
    read_timeout: int  # seconds
    max_attempts: int  # total attempts (initial + retries)


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RetryConfig:
    """Configurable retry parameters for S3 operations.

    Default values preserve current hardcoded behavior.
    """

    max_retries: int = 3
    initial_backoff: timedelta = timedelta(seconds=1.0)
    backoff_multiplier: float = 2.0
