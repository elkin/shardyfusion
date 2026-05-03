"""Shared sharding types used by both writer and reader paths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Self, get_args

DB_ID_COL = "_shard_id"
VECTOR_DB_ID_COL = "_vector_db_id"

RoutingValue = int | str | bytes

_SUPPORTED_ROUTING_VALUE_TYPES = get_args(RoutingValue)


def _validate_homogeneous_non_null_values(
    values: Sequence[object],
    *,
    field_name: str,
) -> None:
    if any(value is None for value in values):
        raise ValueError(f"{field_name} must not contain null values")
    if any(isinstance(value, bool) for value in values):
        raise ValueError(f"{field_name} must not be boolean values")
    for idx, value in enumerate(values):
        if not isinstance(value, _SUPPORTED_ROUTING_VALUE_TYPES):
            raise ValueError(
                f"{field_name} must be int, str, or bytes values; "
                f"got {field_name}[{idx}]={value!r} "
                f"({type(value).__name__})"
            )
    for idx in range(1, len(values)):
        left = values[idx - 1]
        right = values[idx]
        if type(left) is not type(right):
            raise ValueError(
                f"{field_name} must all share the same type; "
                f"got {field_name}[{idx - 1}]={left!r}, {field_name}[{idx}]={right!r}"
            )


def validate_routing_values(
    routing_values: Sequence[RoutingValue],
    *,
    require_unique: bool = True,
) -> None:
    """Validate categorical routing values are unique and type-consistent."""

    _validate_homogeneous_non_null_values(
        routing_values,
        field_name="Routing values",
    )
    if not require_unique:
        return
    seen: set[RoutingValue] = set()
    duplicates: list[RoutingValue] = []
    for value in routing_values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"Routing values must be unique; duplicates={duplicates!r}")


class KeyEncoding(str, Enum):
    """Supported key serialization encodings."""

    U64BE = "u64be"
    U32BE = "u32be"
    UTF8 = "utf8"
    RAW = "raw"

    @classmethod
    def from_value(cls, value: KeyEncoding | str) -> Self:
        """Parse an encoding value from enum or string input."""

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported key encoding: {value!r}. Allowed: {allowed}"
            ) from exc


class ShardingStrategy(str, Enum):
    """Supported sharding strategies."""

    HASH = "hash"
    CEL = "cel"

    @classmethod
    def from_value(cls, value: ShardingStrategy | str) -> Self:
        """Parse a strategy value from enum or string input."""

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported sharding strategy: {value!r}. Allowed: {allowed}"
            ) from exc


class ShardHashAlgorithm(str, Enum):
    """Supported shard routing hash algorithms."""

    XXH3_64 = "xxh3_64"

    @classmethod
    def from_value(cls, value: ShardHashAlgorithm | str) -> Self:
        """Parse a hash algorithm value from enum or string input."""

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported shard hash algorithm: {value!r}. Allowed: {allowed}"
            ) from exc


@dataclass(slots=True)
class ShardingSpec:
    """Base class for sharding specifications."""

    def to_manifest_dict(self) -> dict[str, object]:
        """Return manifest-safe representation."""
        raise NotImplementedError


@dataclass(slots=True)
class HashShardingSpec(ShardingSpec):
    """HASH sharding configuration.

    Uniform distribution via ``hash_algorithm(canonical_bytes(key)) % num_dbs``.
    Supports int, str, and bytes keys. ``num_dbs`` lives on :class:`HashShardedWriteConfig`.
    """

    hash_algorithm: ShardHashAlgorithm = ShardHashAlgorithm.XXH3_64
    max_keys_per_shard: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.hash_algorithm, ShardHashAlgorithm):
            try:
                self.hash_algorithm = ShardHashAlgorithm.from_value(self.hash_algorithm)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        if self.max_keys_per_shard is not None and self.max_keys_per_shard <= 0:
            raise ValueError("max_keys_per_shard must be > 0")

    def to_manifest_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "strategy": ShardingStrategy.HASH.value,
            "hash_algorithm": self.hash_algorithm.value,
        }
        return d


@dataclass(slots=True)
class CelShardingSpec(ShardingSpec):
    """CEL sharding configuration.

    Flexible shard assignment via a CEL expression. The expression may return
    a dense integer shard id directly, or a categorical token resolved by exact match
    against ``routing_values``. A built-in ``shard_hash()`` function (wrapping xxh3_64)
    is available in CEL expressions. ``num_dbs`` is always derived from routing
    metadata or discovered from data.
    """

    cel_expr: str = ""
    cel_columns: dict[str, str] = field(default_factory=dict)
    routing_values: list[RoutingValue] | None = None
    infer_routing_values_from_data: bool = False

    def __post_init__(self) -> None:
        if not self.cel_expr:
            raise ValueError("CEL strategy requires cel_expr")
        if not self.cel_columns:
            raise ValueError("CEL strategy requires cel_columns")
        if self.infer_routing_values_from_data:
            if self.routing_values is not None:
                raise ValueError(
                    "infer_routing_values_from_data cannot be combined with "
                    "routing_values"
                )
        if self.routing_values is not None:
            validate_routing_values(self.routing_values)

    def to_manifest_dict(self) -> dict[str, object]:
        d: dict[str, object] = {
            "strategy": ShardingStrategy.CEL.value,
            "hash_algorithm": ShardHashAlgorithm.XXH3_64.value,
        }
        if self.routing_values is not None:
            d["routing_values"] = self.routing_values
        d["cel_expr"] = self.cel_expr
        d["cel_columns"] = dict(self.cel_columns)
        return d
