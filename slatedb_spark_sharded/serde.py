"""Key/value serialization utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, cast

from .errors import ConfigValidationError


class _KeyedRow(Protocol):
    def __getitem__(self, key: str) -> object: ...


class _AsDictRow(Protocol):
    def asDict(self, recursive: bool = False) -> dict[str, object]: ...


def encode_key(key: object, *, encoding: str = "u64be") -> bytes:
    """Encode a key into bytes for SlateDB."""

    if encoding == "u64be":
        if not isinstance(key, int):
            raise ConfigValidationError(
                f"u64be encoding expects int key, got {type(key)!r}"
            )
        if key < 0 or key > (2**64 - 1):
            raise ConfigValidationError("u64be encoding requires value in [0, 2^64-1]")
        return key.to_bytes(8, byteorder="big", signed=False)

    if encoding == "u32be":
        if not isinstance(key, int):
            raise ConfigValidationError(
                f"u32be encoding expects int key, got {type(key)!r}"
            )
        if key < 0 or key > (2**32 - 1):
            raise ConfigValidationError("u32be encoding requires value in [0, 2^32-1]")
        return key.to_bytes(4, byteorder="big", signed=False)

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
    def binary_col(cls, col_name: str) -> "ValueSpec":
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
    def json_cols(cls, cols: list[str] | None = None) -> "ValueSpec":
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
    def callable_encoder(cls, fn: Callable[[object], bytes]) -> "ValueSpec":
        """Use a custom callable encoder."""

        return cls(encoder=fn, description=getattr(fn, "__name__", "callable_encoder"))
