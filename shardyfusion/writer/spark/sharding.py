"""Sharding specs and Spark sharding helpers."""

from typing import cast

from pyspark import RDD
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType

from shardyfusion.errors import ShardAssignmentError
from shardyfusion.sharding_types import (
    DB_ID_COL,
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
        cel_expr=sharding.cel_expr,
        cel_columns=sharding.cel_columns,
    )

    output_schema = StructType(
        list(df.schema.fields) + [StructField(DB_ID_COL, IntegerType(), False)]
    )

    df_with_db_id: DataFrame
    match sharding.strategy:
        case ShardingStrategy.HASH:
            _key_col = key_col
            _num_dbs = num_dbs

            def _hash_map_arrow(iterator):  # type: ignore[no-untyped-def]
                import pyarrow as pa  # type: ignore[import-not-found]

                from shardyfusion.routing import xxh3_db_id

                for batch in iterator:
                    keys = batch.column(_key_col).to_pylist()
                    db_ids = [xxh3_db_id(k, _num_dbs) for k in keys]
                    yield batch.append_column(
                        DB_ID_COL, pa.array(db_ids, type=pa.int32())
                    )

            df_with_db_id = df.mapInArrow(_hash_map_arrow, output_schema)

        case ShardingStrategy.CEL:
            from shardyfusion.cel import compile_cel

            assert sharding.cel_expr is not None and sharding.cel_columns is not None
            boundaries_for_cel = (
                list(sharding.boundaries) if sharding.boundaries is not None else None
            )
            resolved.boundaries = boundaries_for_cel
            resolved.cel_expr = sharding.cel_expr
            resolved.cel_columns = sharding.cel_columns

            _cel_expr = sharding.cel_expr
            _cel_cols = dict(sharding.cel_columns)
            _cel_boundaries = boundaries_for_cel

            compile_cel(_cel_expr, _cel_cols)  # validate eagerly on driver

            def _cel_map_arrow(iterator):  # type: ignore[no-untyped-def]
                import pyarrow as pa  # type: ignore[import-not-found]

                from shardyfusion.cel import compile_cel as _compile
                from shardyfusion.cel import route_cel_batch

                _compiled = _compile(_cel_expr, _cel_cols)
                for batch in iterator:
                    db_ids = route_cel_batch(_compiled, batch, _cel_boundaries)
                    yield batch.append_column(
                        DB_ID_COL, pa.array(db_ids, type=pa.int32())
                    )

            df_with_db_id = df.mapInArrow(_cel_map_arrow, output_schema)

        case _:
            raise ShardAssignmentError(
                f"Unsupported sharding strategy: {sharding.strategy!r}"
            )

    # Validate db_id range (skip for CEL direct mode where num_dbs may be 0/unknown)
    if num_dbs > 0:
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
