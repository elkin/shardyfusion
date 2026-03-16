"""Sharding specs and Spark sharding helpers."""

import math
from collections.abc import Sequence
from typing import cast

from pyspark import RDD
from pyspark.ml.feature import Bucketizer
from pyspark.sql import Column, DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, StringType

from shardyfusion.errors import ShardAssignmentError
from shardyfusion.ordering import compare_ordered
from shardyfusion.sharding_types import (
    DB_ID_COL,
    BoundaryValue,
    ShardingSpec,
    ShardingStrategy,
)


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
        boundaries=sharding.boundaries,
        approx_quantile_rel_error=sharding.approx_quantile_rel_error,
        cel_expr=sharding.cel_expr,
        cel_columns=sharding.cel_columns,
    )

    # Validate key column type for HASH/RANGE (CEL validates its own columns).
    validate_key_col_type(
        df=df,
        key_col=key_col,
        strategy=sharding.strategy,
    )

    df_with_db_id: DataFrame
    match sharding.strategy:
        case ShardingStrategy.HASH:
            # SHARDING INVARIANT: This expression MUST produce identical results
            # to routing.xxhash64_db_id(key, num_dbs, key_encoding).
            # Verified at runtime by verify_routing_agreement() and
            # cross-checked by tests/unit/writer/test_routing_contract.py.
            db_expr = F.pmod(F.xxhash64(F.col(key_col).cast("long")), F.lit(num_dbs))
            df_with_db_id = df.withColumn(DB_ID_COL, db_expr.cast("int"))
        case ShardingStrategy.RANGE:
            boundaries = _resolve_boundaries(df, key_col, num_dbs, sharding)
            resolved.boundaries = boundaries
            if _boundaries_are_numeric(boundaries):
                df_with_db_id = _range_bucketize_df(df, key_col, boundaries)
            else:
                db_expr = _range_bucket_expr(key_col, boundaries)
                df_with_db_id = df.withColumn(DB_ID_COL, db_expr.cast("int"))
        case ShardingStrategy.CEL:
            from shardyfusion.cel import (
                compile_cel,
                pandas_rows_to_contexts,
                resolve_cel_boundaries,
            )

            assert sharding.cel_expr is not None and sharding.cel_columns is not None
            compiled = compile_cel(sharding.cel_expr, sharding.cel_columns)

            if sharding.boundaries is not None:
                boundaries_for_cel = list(sharding.boundaries)
            else:
                sample_pdf = (
                    df.select(*sharding.cel_columns.keys()).limit(10000).toPandas()
                )
                sampled = pandas_rows_to_contexts(sample_pdf, sharding.cel_columns)
                cel_resolved = resolve_cel_boundaries(
                    compiled, sampled, num_dbs, sharding
                )
                boundaries_for_cel = cel_resolved.boundaries or []
            resolved.boundaries = boundaries_for_cel
            resolved.cel_expr = sharding.cel_expr
            resolved.cel_columns = sharding.cel_columns

            # Capture serializable values (strings, list) — NOT the C++ compiled object.
            _cel_expr = sharding.cel_expr
            _cel_cols = dict(sharding.cel_columns)

            def _cel_map_arrow(iterator):  # type: ignore[no-untyped-def]
                import pyarrow as pa  # type: ignore[import-not-found]

                from shardyfusion.cel import compile_cel as _compile
                from shardyfusion.cel import route_cel_batch

                _compiled = _compile(_cel_expr, _cel_cols)
                for batch in iterator:
                    db_ids = route_cel_batch(_compiled, batch, boundaries_for_cel)
                    yield batch.append_column(
                        DB_ID_COL, pa.array(db_ids, type=pa.int32())
                    )

            # Apply CEL + boundary assignment via mapInArrow
            from pyspark.sql.types import StructField, StructType

            output_schema = StructType(
                list(df.schema.fields) + [StructField(DB_ID_COL, IntegerType(), False)]
            )
            df_with_db_id = df.mapInArrow(_cel_map_arrow, output_schema)
        case _:
            raise ShardAssignmentError(
                f"Unsupported sharding strategy: {sharding.strategy!r}"
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
) -> RDD[tuple[int, Row]]:
    """Return pair RDD partitioned so partition index matches db id."""

    prepared = df_with_db_id
    if sort_within_partitions:
        prepared = prepared.sortWithinPartitions(key_col)

    pair_rdd = cast(RDD[Row], prepared.rdd).map(lambda row: (int(row[DB_ID_COL]), row))
    return pair_rdd.partitionBy(num_dbs, lambda key: int(key))


def validate_key_col_type(
    *,
    df: DataFrame,
    key_col: str,
    strategy: ShardingStrategy,
) -> None:
    # CEL validates its own columns — key_col may not exist in the schema.
    if strategy == ShardingStrategy.CEL:
        return

    try:
        dtype = df.schema[key_col].dataType
    except KeyError as exc:
        raise ShardAssignmentError(
            f"Key column `{key_col}` was not found in DataFrame schema"
        ) from exc

    match strategy:
        case ShardingStrategy.HASH:
            allowed_hash = (IntegerType, LongType)
            if not isinstance(dtype, allowed_hash):
                raise ShardAssignmentError(
                    "Hash sharding requires key column type IntegerType or LongType; "
                    f"got {type(dtype).__name__} for `{key_col}`"
                )
            return

        case ShardingStrategy.RANGE:
            allowed_range = (IntegerType, LongType, StringType)
            if not isinstance(dtype, allowed_range):
                raise ShardAssignmentError(
                    "Range sharding requires key column type one of "
                    "IntegerType, LongType, StringType; "
                    f"got {type(dtype).__name__} for `{key_col}`"
                )
            return

        # CEL handled by early return above.


def _resolve_boundaries(
    df: DataFrame,
    key_col: str,
    num_dbs: int,
    sharding: ShardingSpec,
) -> list[BoundaryValue]:
    expected = max(num_dbs - 1, 0)
    if sharding.boundaries is not None:
        boundaries = list(sharding.boundaries)
        if len(boundaries) != expected:
            raise ShardAssignmentError(
                f"Range sharding expects {expected} boundaries for num_dbs={num_dbs}, got {len(boundaries)}"
            )
        _validate_boundaries(boundaries)
        return boundaries

    probabilities = [idx / num_dbs for idx in range(1, num_dbs)]
    quantiles = df.approxQuantile(
        key_col,
        probabilities,
        sharding.approx_quantile_rel_error,
    )
    if len(quantiles) != expected:
        raise ShardAssignmentError(
            "Range sharding could not derive the expected number of boundaries from "
            f"approxQuantile: expected {expected}, got {len(quantiles)}"
        )
    resolved: list[BoundaryValue] = list(quantiles)
    _validate_boundaries(resolved)
    return resolved


def _range_bucket_expr(key_col: str, boundaries: Sequence[BoundaryValue]) -> Column:
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


def _boundaries_are_numeric(boundaries: Sequence[BoundaryValue]) -> bool:
    return all(
        isinstance(boundary, (int, float)) and not isinstance(boundary, bool)
        for boundary in boundaries
    )


def _range_bucketize_df(
    df: DataFrame,
    key_col: str,
    boundaries: Sequence[BoundaryValue],
) -> DataFrame:
    """Apply range bucketing with Spark ML Bucketizer for numeric boundaries."""

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


def _validate_boundaries(boundaries: Sequence[BoundaryValue]) -> None:
    """Validate boundaries are non-null and strictly increasing."""

    if any(boundary is None for boundary in boundaries):
        raise ShardAssignmentError("Range boundaries must not contain null values")
    if any(isinstance(boundary, bool) for boundary in boundaries):
        raise ShardAssignmentError("Range boundaries must not be boolean values")

    for idx in range(1, len(boundaries)):
        left = boundaries[idx - 1]
        right = boundaries[idx]
        if type(left) is not type(right):
            raise ShardAssignmentError(
                "Range boundaries must all share one type; "
                f"got boundaries[{idx - 1}]={left!r}, boundaries[{idx}]={right!r}"
            )
        mismatch_message = (
            "Range boundaries contain non-comparable values; "
            f"got boundaries[{idx - 1}]={left!r}, boundaries[{idx}]={right!r}"
        )
        try:
            is_increasing = (
                compare_ordered(
                    left,
                    right,
                    mismatch_message=mismatch_message,
                )
                < 0
            )
        except ValueError as exc:
            raise ShardAssignmentError(str(exc)) from exc
        if not is_increasing:
            raise ShardAssignmentError(
                "Range boundaries must be strictly increasing; "
                f"got boundaries[{idx - 1}]={left!r}, boundaries[{idx}]={right!r}"
            )


def _sql_literal(value: BoundaryValue) -> str:
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
