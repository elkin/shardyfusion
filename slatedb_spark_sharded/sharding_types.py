"""Shared sharding types used by both writer and reader paths."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

DB_ID_COL = "_slatedb_db_id"

BoundaryValue = int | float | str


class KeyEncoding(str, Enum):
    """Supported key serialization encodings."""

    U64BE = "u64be"
    U32BE = "u32be"

    @classmethod
    def from_value(cls, value: "KeyEncoding | str") -> Self:
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
    RANGE = "range"
    CUSTOM_EXPR = "custom_expr"

    @classmethod
    def from_value(cls, value: "ShardingStrategy | str") -> Self:
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
    """Configuration for mapping rows to shard database ids."""

    strategy: ShardingStrategy = ShardingStrategy.HASH
    boundaries: list[BoundaryValue] | None = None
    approx_quantile_rel_error: float = 0.01
    custom_expr: str | None = None
    custom_column_builder: Callable[[str], Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, ShardingStrategy):
            raise ValueError("strategy must be ShardingStrategy")

    def to_manifest_dict(self) -> dict[str, object]:
        """Return manifest-safe representation (Spark callables omitted)."""

        return {
            "strategy": self.strategy.value,
            "boundaries": self.boundaries,
            "approx_quantile_rel_error": self.approx_quantile_rel_error,
            "custom_expr": self.custom_expr,
        }
