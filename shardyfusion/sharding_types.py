"""Shared sharding types used by both writer and reader paths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Self

from .ordering import compare_ordered

DB_ID_COL = "_slatedb_db_id"

BoundaryValue = int | float | str | bytes


def validate_boundaries(boundaries: Sequence[BoundaryValue]) -> None:
    """Validate boundaries are non-null, same-type, and strictly increasing.

    Raises :class:`ValueError` on invalid input.
    """
    if any(b is None for b in boundaries):
        raise ValueError("Boundaries must not contain null values")
    if any(isinstance(b, bool) for b in boundaries):
        raise ValueError("Boundaries must not be boolean values")
    for idx in range(1, len(boundaries)):
        left = boundaries[idx - 1]
        right = boundaries[idx]
        if type(left) is not type(right):
            raise ValueError(
                "Boundaries must all share the same type; "
                f"got boundaries[{idx - 1}]={left!r}, boundaries[{idx}]={right!r}"
            )
        msg = (
            "Boundaries contain non-comparable values; "
            f"got boundaries[{idx - 1}]={left!r}, boundaries[{idx}]={right!r}"
        )
        try:
            is_increasing = compare_ordered(left, right, mismatch_message=msg) < 0
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        if not is_increasing:
            raise ValueError(
                "Boundaries must be strictly increasing; "
                f"got boundaries[{idx - 1}]={left!r}, boundaries[{idx}]={right!r}"
            )


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


@dataclass(slots=True)
class ShardingSpec:
    """Configuration for mapping rows to shard database ids.

    Two strategies are supported:

    **HASH** (default): Uniform distribution via ``xxh3_64(canonical_bytes(key)) % num_dbs``.
    Supports int, str, and bytes keys. Requires ``num_dbs > 0`` (explicit or computed
    from ``max_keys_per_shard``).

    **CEL**: Flexible shard assignment via a CEL expression. The expression must produce
    **consecutive 0-based integer shard IDs** (e.g., ``shard_hash(key) % 100u`` yields
    IDs 0–99). A built-in ``shard_hash()`` function (wrapping xxh3_64) is available in
    CEL expressions. ``num_dbs`` is always discovered from data; it must not be provided
    explicitly. Optional ``boundaries`` enable ``bisect_right``-based routing.
    """

    strategy: ShardingStrategy = ShardingStrategy.HASH
    boundaries: list[BoundaryValue] | None = None
    cel_expr: str | None = None
    cel_columns: dict[str, str] | None = None
    max_keys_per_shard: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, ShardingStrategy):
            raise ValueError("strategy must be ShardingStrategy")
        if self.strategy == ShardingStrategy.CEL:
            if not self.cel_expr:
                raise ValueError("CEL strategy requires cel_expr")
            if not self.cel_columns:
                raise ValueError("CEL strategy requires cel_columns")
        elif self.cel_expr is not None:
            raise ValueError("cel_expr is only valid with CEL strategy")
        if self.boundaries is not None and self.strategy != ShardingStrategy.CEL:
            raise ValueError("boundaries are only valid with CEL strategy")
        if self.boundaries:
            validate_boundaries(self.boundaries)
        if (
            self.max_keys_per_shard is not None
            and self.strategy != ShardingStrategy.HASH
        ):
            raise ValueError("max_keys_per_shard is only valid with HASH strategy")
        if self.max_keys_per_shard is not None and self.max_keys_per_shard <= 0:
            raise ValueError("max_keys_per_shard must be > 0")

    def to_manifest_dict(self) -> dict[str, object]:
        """Return manifest-safe representation (Spark callables omitted)."""

        d: dict[str, object] = {
            "strategy": self.strategy.value,
            "boundaries": self.boundaries,
        }
        if self.cel_expr is not None:
            d["cel_expr"] = self.cel_expr
        if self.cel_columns is not None:
            d["cel_columns"] = self.cel_columns
        return d
