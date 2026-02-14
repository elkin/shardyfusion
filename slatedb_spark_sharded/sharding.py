"""Sharding specs and Spark sharding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from pyspark import RDD
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from .errors import ShardAssignmentError

DB_ID_COL = "_slatedb_db_id"


class ShardingStrategy(str, Enum):
    """Supported sharding strategies."""

    HASH = "hash"
    RANGE = "range"
    CUSTOM_EXPR = "custom_expr"

    @classmethod
    def from_value(cls, value: "ShardingStrategy | str") -> "ShardingStrategy":
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

    strategy: ShardingStrategy | str = ShardingStrategy.HASH
    boundaries: list[float] | list[int] | list[str] | None = None
    approx_quantile_rel_error: float = 0.01
    custom_expr: str | None = None
    custom_column_builder: Callable[[str], Column] | None = None

    def __post_init__(self) -> None:
        self.strategy = ShardingStrategy.from_value(self.strategy)

    def to_manifest_dict(self) -> dict[str, object]:
        """Return manifest-safe representation (Spark callables omitted)."""

        return {
            "strategy": self.strategy.value,
            "boundaries": self.boundaries,
            "approx_quantile_rel_error": self.approx_quantile_rel_error,
            "custom_expr": self.custom_expr,
        }


def add_db_id_column(
    df: DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    sharding: ShardingSpec,
) -> tuple[DataFrame, ShardingSpec]:
    """Add deterministic db id column and return resolved sharding spec."""

    resolved = ShardingSpec(
        strategy=sharding.strategy,
        boundaries=sharding.boundaries if sharding.boundaries is not None else None,
        approx_quantile_rel_error=sharding.approx_quantile_rel_error,
        custom_expr=sharding.custom_expr,
        custom_column_builder=sharding.custom_column_builder,
    )

    match sharding.strategy:
        case ShardingStrategy.HASH:
            db_expr = F.pmod(F.hash(F.col(key_col)), F.lit(num_dbs))
        case ShardingStrategy.RANGE:
            boundaries = _resolve_boundaries(df, key_col, num_dbs, sharding)
            resolved.boundaries = boundaries
            db_expr = _range_bucket_expr(F.col(key_col), boundaries)
        case ShardingStrategy.CUSTOM_EXPR:
            db_expr = _custom_expr(sharding, key_col)
        case _:  # pragma: no cover - guarded by ShardingSpec.__post_init__
            raise ShardAssignmentError(
                f"Unsupported sharding strategy: {sharding.strategy}"
            )

    df_with_db_id = df.withColumn(DB_ID_COL, db_expr.cast("int"))

    invalid_count = (
        df_with_db_id.where(
            (F.col(DB_ID_COL).isNull())
            | (F.col(DB_ID_COL) < 0)
            | (F.col(DB_ID_COL) >= num_dbs)
        )
        .limit(1)
        .count()
    )
    if invalid_count > 0:
        raise ShardAssignmentError("Computed db_id out of range [0, num_dbs-1].")

    return df_with_db_id, resolved


def prepare_partitioned_rdd(
    df_with_db_id: DataFrame,
    *,
    num_dbs: int,
    key_col: str,
    sort_within_partitions: bool,
) -> RDD[tuple[int, object]]:
    """Return pair RDD partitioned so partition index matches db id."""

    prepared = df_with_db_id
    if sort_within_partitions:
        prepared = prepared.sortWithinPartitions(key_col)

    pair_rdd = prepared.rdd.map(lambda row: (int(row[DB_ID_COL]), row))
    return pair_rdd.partitionBy(num_dbs, lambda key: int(key))


def _resolve_boundaries(
    df: DataFrame,
    key_col: str,
    num_dbs: int,
    sharding: ShardingSpec,
) -> list[float] | list[int] | list[str]:
    if sharding.boundaries is not None:
        boundaries = sharding.boundaries
        expected = max(num_dbs - 1, 0)
        if len(boundaries) != expected:
            raise ShardAssignmentError(
                f"Range sharding expects {expected} boundaries for num_dbs={num_dbs}, got {len(boundaries)}"
            )
        return boundaries

    probabilities = [idx / num_dbs for idx in range(1, num_dbs)]
    boundaries = df.approxQuantile(
        key_col,
        probabilities,
        sharding.approx_quantile_rel_error,
    )
    return boundaries


def _range_bucket_expr(
    col: Column, boundaries: list[float] | list[int] | list[str]
) -> Column:
    expr: Column = F.lit(len(boundaries))
    for idx in range(len(boundaries) - 1, -1, -1):
        expr = F.when(col < F.lit(boundaries[idx]), F.lit(idx)).otherwise(expr)
    return expr


def _custom_expr(sharding: ShardingSpec, key_col: str) -> Column:
    if sharding.custom_expr:
        return F.expr(sharding.custom_expr)
    if sharding.custom_column_builder is not None:
        return sharding.custom_column_builder(key_col)
    raise ShardAssignmentError(
        "custom_expr sharding requires either `custom_expr` SQL text or `custom_column_builder`."
    )
