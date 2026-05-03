"""Key/value serialization utilities."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, Self, cast

from .errors import ConfigValidationError
from .sharding_types import KeyEncoding

KeyEncoder = Callable[[object], bytes]
"""Callable that encodes a key into bytes for SlateDB."""

_UINT64_MAX = (1 << 64) - 1
_UINT32_MAX = (1 << 32) - 1


class _KeyedRow(Protocol):
    def __getitem__(self, key: str) -> object: ...


class _AsDictRow(Protocol):
    def asDict(self, recursive: bool = False) -> dict[str, object]: ...


def _encode_key_u64be(key: object) -> bytes:
    """Encode key as 8-byte big-endian unsigned integer."""

    if not isinstance(key, int):
        raise ConfigValidationError(
            f"u64be encoding expects int key, got {type(key)!r}"
        )
    if key < 0 or key > _UINT64_MAX:
        raise ConfigValidationError("u64be encoding requires value in [0, 2^64-1]")
    return key.to_bytes(8, byteorder="big", signed=False)


def _encode_key_u32be(key: object) -> bytes:
    """Encode key as 4-byte big-endian unsigned integer."""

    if not isinstance(key, int):
        raise ConfigValidationError(
            f"u32be encoding expects int key, got {type(key)!r}"
        )
    if key < 0 or key > _UINT32_MAX:
        raise ConfigValidationError("u32be encoding requires value in [0, 2^32-1]")
    return key.to_bytes(4, byteorder="big", signed=False)


def _encode_key_utf8(key: object) -> bytes:
    """Encode key as UTF-8 bytes from a string."""

    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, bytes):
        return key
    raise ConfigValidationError(
        f"utf8 encoding expects str or bytes key, got {type(key)!r}"
    )


def _encode_key_raw(key: object) -> bytes:
    """Pass-through bytes encoding."""

    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8")
    raise ConfigValidationError(
        f"raw encoding expects bytes or str key, got {type(key)!r}"
    )


def make_key_encoder(encoding: KeyEncoding) -> KeyEncoder:
    """Return a key encoder for the given encoding.

    Resolves the encoding branch once at setup time so hot loops
    can call the returned function without per-key dispatch.
    """

    if encoding == KeyEncoding.U64BE:
        return _encode_key_u64be
    if encoding == KeyEncoding.U32BE:
        return _encode_key_u32be
    if encoding == KeyEncoding.UTF8:
        return _encode_key_utf8
    if encoding == KeyEncoding.RAW:
        return _encode_key_raw
    raise ConfigValidationError(f"Unsupported key encoding: {encoding}")


@dataclass(slots=True)
class ValueSpec:
    """Value serialization strategy for a Spark row."""

    encoder: Callable[[object], bytes]
    description: str

    def encode(self, row: object) -> bytes:
        """Encode one row into value bytes."""

        return self.encoder(row)

    @classmethod
    def binary_col(cls, col_name: str) -> Self:
        """Use one column directly as bytes payload."""

        def _encode(row: object) -> bytes:
            value = cast(_KeyedRow, row)[col_name]
            if value is None:
                return b""
            if isinstance(value, bytes):
                return value
            if isinstance(value, bytearray):
                return bytes(value)
            if isinstance(value, str):
                return value.encode("utf-8")
            raise ConfigValidationError(
                f"binary_col expects bytes/bytearray/str, got {type(value)!r}"
            )

        return cls(encoder=_encode, description=f"binary_col:{col_name}")

    @classmethod
    def json_cols(cls, cols: list[str] | None = None) -> Self:
        """Encode selected columns (or full row) as UTF-8 JSON."""

        def _encode(row: object) -> bytes:
            row_dict = (
                cast(_AsDictRow, row).asDict(recursive=True)
                if hasattr(row, "asDict")
                else dict(cast(Mapping[str, object], row))
            )
            obj = {key: row_dict.get(key) for key in cols} if cols else row_dict
            return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )

        detail = "all" if cols is None else ",".join(cols)
        return cls(encoder=_encode, description=f"json_cols:{detail}")

    @classmethod
    def callable_encoder(cls, fn: Callable[[object], bytes]) -> Self:
        """Use a custom callable encoder."""

        return cls(encoder=fn, description=getattr(fn, "__name__", "callable_encoder"))

    def referenced_columns(self) -> list[str] | None:
        """Return column names referenced by this value spec.

        Returns ``None`` when the spec references all columns dynamically
        (e.g. ``json_cols()`` with no explicit columns, or a callable encoder).
        """
        if self.description.startswith("binary_col:"):
            return [self.description.split(":", 1)[1]]
        if self.description.startswith("json_cols:"):
            detail = self.description.split(":", 1)[1]
            if detail == "all":
                return None
            return detail.split(",")
        return None
