"""Shared type aliases and protocols used across the package."""

from pathlib import Path
from typing import Protocol, TypeAlias, TypedDict

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]

KeyInput: TypeAlias = int | str | bytes
KeyLike: TypeAlias = int | str


class ShardReader(Protocol):
    """Minimal SlateDB reader shape used by the sharded reader service."""

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


class S3ClientConfig(TypedDict, total=False):
    """Supported explicit overrides for boto3 S3 client construction."""

    endpoint_url: str
    region_name: str
    access_key_id: str
    secret_access_key: str
    session_token: str
    # botocore / S3-connection options
    addressing_style: str  # "virtual" | "path" | "auto"
    signature_version: str  # "s3v4" | "s3"
    verify_ssl: bool | str  # bool or path to CA bundle
    connect_timeout: int  # seconds
    read_timeout: int  # seconds
    max_attempts: int  # total attempts (initial + retries)


# ---------------------------------------------------------------------------
# Tracing Protocols (Layer 0 — no opentelemetry import)
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402


class Span(Protocol):
    """Minimal span interface satisfied by OTel's ``opentelemetry.trace.Span``."""

    def set_attribute(self, key: str, value: Any) -> None: ...
    def set_status(self, status: Any, description: str | None = None) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...
    def __enter__(self) -> "Span": ...
    def __exit__(self, *args: object) -> None: ...


class Tracer(Protocol):
    """Minimal tracer interface satisfied by OTel's ``opentelemetry.trace.Tracer``.

    Users who don't install the ``metrics-otel`` extra simply don't pass a
    tracer — zero overhead, zero imports.
    """

    def start_as_current_span(self, name: str, **kwargs: Any) -> Span: ...


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class RetryConfig:
    """Configurable retry parameters for S3 operations.

    Default values preserve current hardcoded behavior.
    """

    max_retries: int = 3
    initial_backoff_s: float = 1.0
    backoff_multiplier: float = 2.0
