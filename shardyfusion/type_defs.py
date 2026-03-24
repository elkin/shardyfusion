"""Shared type aliases and protocols used across the package."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Protocol, TypeAlias, TypedDict

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

KeyInput: TypeAlias = int | str | bytes


class ShardReader(Protocol):
    """Minimal shard reader shape used by the sharded reader service."""

    def get(self, key: bytes) -> bytes | None:
        """Return value bytes for key, or None when absent."""
        ...

    def close(self) -> None:
        """Release reader resources."""
        ...


class ShardReaderFactory(Protocol):
    """Factory for opening one shard reader."""

    def __call__(
        self, *, db_url: str, local_dir: Path, checkpoint_id: str | None
    ) -> ShardReader:
        """Construct an opened reader instance."""
        ...


class AsyncShardReader(Protocol):
    """Async counterpart of ShardReader for use with AsyncShardedReader."""

    async def get(self, key: bytes) -> bytes | None: ...

    async def close(self) -> None: ...


class AsyncShardReaderFactory(Protocol):
    """Factory for opening one async shard reader."""

    async def __call__(
        self, *, db_url: str, local_dir: Path, checkpoint_id: str | None
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
