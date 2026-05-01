"""Sharding specs and Spark sharding helpers."""

from typing import cast

import numpy as np
from pyspark import RDD
from pyspark.sql import DataFrame, Row
from pyspark.sql.types import IntegerType, StructField, StructType

from shardyfusion._writer_core import build_categorical_routing_values
from shardyfusion.errors import ShardAssignmentError
from shardyfusion.sharding_types import DB_ID_COL
from shardyfusion.vector._distributed import ResolvedVectorRouting
from shardyfusion.vector.sharding import cluster_assign, lsh_assign
from shardyfusion.vector.types import VectorShardingStrategy as VecStrategy


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


def _discover_categorical_routing_values(
    df: DataFrame,
    *,
    cel_expr: str,
    cel_columns: dict[str, str],
) -> list[int | str | bytes]:
    selected = df.select(*cel_columns.keys())
    _cel_expr = cel_expr
    _cel_cols = dict(cel_columns)

    def _partition_tokens(rows):  # type: ignore[no-untyped-def]
        from shardyfusion.cel import compile_cel

        compiled = compile_cel(_cel_expr, _cel_cols)
        for row in rows:
            yield compiled.evaluate(row.asDict(recursive=False))

    distinct_tokens = selected.rdd.mapPartitions(_partition_tokens).distinct().collect()
    return build_categorical_routing_values(distinct_tokens)


VECTOR_DB_ID_COL = "_vector_db_id"


def add_vector_db_id_column(
    df: DataFrame,
    *,
    vector_col: str,
    routing: ResolvedVectorRouting,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
) -> tuple[DataFrame, int]:
    """Add deterministic _vector_db_id column based on vector routing.

    Uses mapInArrow with per-row Python routing to match the Python writer's
    assign_vector_shard() behavior.

    Returns:
        Tuple of (modified DataFrame, num_dbs)
    """
    strategy = routing.strategy
    num_dbs = routing.num_dbs

    output_schema = StructType(
        list(df.schema.fields) + [StructField(VECTOR_DB_ID_COL, IntegerType(), False)]
    )

    if strategy == VecStrategy.EXPLICIT:
        assert shard_id_col is not None, "shard_id_col required for EXPLICIT"

        def _explicit_map_arrow(iterator):  # type: ignore[no-untyped-def]
            import pyarrow as pa

            for batch in iterator:
                shard_ids = batch.column(shard_id_col).to_pylist()
                yield batch.append_column(
                    VECTOR_DB_ID_COL, pa.array(shard_ids, type=pa.int32())
                )

        df_with_id = df.mapInArrow(_explicit_map_arrow, output_schema)

    elif strategy == VecStrategy.CLUSTER:
        assert routing.centroids is not None, "centroids required for CLUSTER"
        _centroids = routing.centroids
        _metric = routing.metric if routing.metric else "cosine"

        def _cluster_map_arrow(iterator):  # type: ignore[no-untyped-def]
            import pyarrow as pa

            from shardyfusion.vector.types import DistanceMetric

            metric = _metric
            if isinstance(metric, str):
                metric = DistanceMetric(metric)

            for batch in iterator:
                vectors = batch.column(vector_col).to_pylist()
                db_ids = [
                    cluster_assign(np.asarray(v, dtype=np.float32), _centroids, metric)
                    for v in vectors
                ]
                yield batch.append_column(
                    VECTOR_DB_ID_COL, pa.array(db_ids, type=pa.int32())
                )

        df_with_id = df.mapInArrow(_cluster_map_arrow, output_schema)

    elif strategy == VecStrategy.LSH:
        assert routing.hyperplanes is not None, "hyperplanes required for LSH"
        _hyperplanes = routing.hyperplanes

        def _lsh_map_arrow(iterator):  # type: ignore[no-untyped-def]
            import pyarrow as pa

            for batch in iterator:
                vectors = batch.column(vector_col).to_pylist()
                db_ids = [
                    lsh_assign(np.asarray(v, dtype=np.float32), _hyperplanes, num_dbs)
                    for v in vectors
                ]
                yield batch.append_column(
                    VECTOR_DB_ID_COL, pa.array(db_ids, type=pa.int32())
                )

        df_with_id = df.mapInArrow(_lsh_map_arrow, output_schema)

    elif strategy == VecStrategy.CEL:
        assert routing.compiled_cel is not None, "compiled_cel required for CEL"
        assert routing.cel_expr is not None, "cel_expr required for CEL"
        assert routing_context_cols is not None, "routing_context_cols required for CEL"

        from shardyfusion.cel import compile_cel as _compile

        _cel_expr = routing.cel_expr
        _cel_cols = dict(routing_context_cols)
        _routing_values = routing.routing_values

        _compile(_cel_expr, _cel_cols)  # validate eagerly on driver

        def _cel_map_arrow(iterator):  # type: ignore[no-untyped-def]
            import pyarrow as pa

            from shardyfusion.cel import compile_cel as _compile_inner
            from shardyfusion.cel import route_cel_batch

            _compiled = _compile_inner(_cel_expr, _cel_cols)
            for batch in iterator:
                db_ids = route_cel_batch(
                    _compiled,
                    batch,
                    _routing_values,
                )
                yield batch.append_column(
                    VECTOR_DB_ID_COL, pa.array(db_ids, type=pa.int32())
                )

        df_with_id = df.mapInArrow(_cel_map_arrow, output_schema)

    else:
        raise ShardAssignmentError(f"Unsupported vector strategy: {strategy}")

    return df_with_id, num_dbs
