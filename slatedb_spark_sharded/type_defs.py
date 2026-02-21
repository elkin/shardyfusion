"""Shared type aliases and protocols used across the package."""

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
