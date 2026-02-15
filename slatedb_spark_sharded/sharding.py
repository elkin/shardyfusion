"""Sharding specs and Spark sharding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
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

    strategy: ShardingStrategy = ShardingStrategy.HASH
    boundaries: list[float] | list[int] | list[str] | None = None
    approx_quantile_rel_error: float = 0.01
    custom_expr: str | None = None
    custom_column_builder: Callable[[str], Column] | None = None

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

    df_with_db_id: DataFrame
    match sharding.strategy:
        case ShardingStrategy.HASH:
            db_expr = F.pmod(F.hash(F.col(key_col)), F.lit(num_dbs))
            df_with_db_id = df.withColumn(DB_ID_COL, db_expr.cast("int"))
        case ShardingStrategy.RANGE:
            boundaries = _resolve_boundaries(df, key_col, num_dbs, sharding)
            resolved.boundaries = boundaries
            if _boundaries_are_numeric(boundaries):
                df_with_db_id = _range_bucketize_df(df, key_col, boundaries)
            else:
                db_expr = _range_bucket_expr(key_col, boundaries)
                df_with_db_id = df.withColumn(DB_ID_COL, db_expr.cast("int"))
        case ShardingStrategy.CUSTOM_EXPR:
            db_expr = _custom_expr(sharding, key_col)
            df_with_db_id = df.withColumn(DB_ID_COL, db_expr.cast("int"))
        case _:  # pragma: no cover - guarded by ShardingSpec.__post_init__
            raise ShardAssignmentError(
                f"Unsupported sharding strategy: {sharding.strategy}"
            )

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
        _validate_boundaries(boundaries)
        return boundaries

    probabilities = [idx / num_dbs for idx in range(1, num_dbs)]
    boundaries = df.approxQuantile(
        key_col,
        probabilities,
        sharding.approx_quantile_rel_error,
    )
    _validate_boundaries(boundaries)
    return boundaries


def _range_bucket_expr(
    key_col: str, boundaries: list[float] | list[int] | list[str]
) -> Column:
    """Build range-bucket expression using pure Spark SQL functions."""

    if not boundaries:
        return F.lit(0)

    escaped_col = key_col.replace("`", "``")
    boundary_sql = ", ".join(_sql_literal(value) for value in boundaries)
    # bucket id = number of boundaries less-than-or-equal-to value.
    # Example boundaries [10, 20]:
    #   value < 10  -> 0
    #   10..19      -> 1
    #   >= 20       -> 2
    return F.expr(
        f"size(filter(array({boundary_sql}), boundary -> `{escaped_col}` >= boundary))"
    )


def _boundaries_are_numeric(boundaries: list[float] | list[int] | list[str]) -> bool:
    return all(
        isinstance(boundary, (int, float)) and not isinstance(boundary, bool)
        for boundary in boundaries
    )


def _range_bucketize_df(
    df: DataFrame,
    key_col: str,
    boundaries: list[float] | list[int] | list[str],
) -> DataFrame:
    """Apply range bucketing with Spark ML Bucketizer for numeric boundaries."""

    try:
        from pyspark.ml.feature import Bucketizer
    except Exception:  # pragma: no cover - dependency/environment dependent
        db_expr = _range_bucket_expr(key_col, boundaries)
        return df.withColumn(DB_ID_COL, db_expr.cast("int"))

    splits = [
        -float("inf"),
        *[float(boundary) for boundary in boundaries],
        float("inf"),
    ]
    bucketizer = Bucketizer(
        splits=splits,
        inputCol=key_col,
        outputCol=DB_ID_COL,
        handleInvalid="error",
    )
    return bucketizer.transform(df).withColumn(DB_ID_COL, F.col(DB_ID_COL).cast("int"))


def _custom_expr(sharding: ShardingSpec, key_col: str) -> Column:
    if sharding.custom_expr:
        return F.expr(sharding.custom_expr)
    if sharding.custom_column_builder is not None:
        return sharding.custom_column_builder(key_col)
    raise ShardAssignmentError(
        "custom_expr sharding requires either `custom_expr` SQL text or `custom_column_builder`."
    )


def _validate_boundaries(boundaries: list[float] | list[int] | list[str]) -> None:
    """Validate boundaries are non-null and strictly increasing."""

    if any(boundary is None for boundary in boundaries):
        raise ShardAssignmentError("Range boundaries must not contain null values")
    if any(isinstance(boundary, bool) for boundary in boundaries):
        raise ShardAssignmentError("Range boundaries must not be boolean values")

    for idx in range(1, len(boundaries)):
        left = boundaries[idx - 1]
        right = boundaries[idx]
        try:
            if not (left < right):
                raise ShardAssignmentError(
                    "Range boundaries must be strictly increasing; "
                    f"got boundaries[{idx - 1}]={left!r}, boundaries[{idx}]={right!r}"
                )
        except TypeError as exc:
            raise ShardAssignmentError(
                "Range boundaries contain non-comparable values; "
                f"got boundaries[{idx - 1}]={left!r}, boundaries[{idx}]={right!r}"
            ) from exc


def _sql_literal(value: float | int | str) -> str:
    """Render a safe SQL literal for range boundary values."""

    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return "'" + escaped + "'"
    if isinstance(value, bool):
        raise ShardAssignmentError("Boolean boundary literals are not supported")
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "CAST('NaN' AS DOUBLE)"
        if math.isinf(value):
            return (
                "CAST('Infinity' AS DOUBLE)"
                if value > 0
                else "CAST('-Infinity' AS DOUBLE)"
            )
        return repr(value)
    raise ShardAssignmentError(f"Unsupported boundary literal type: {type(value)!r}")
