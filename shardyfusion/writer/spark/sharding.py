"""Sharding specs and Spark sharding helpers."""

from typing import cast

import numpy as np
from pyspark import RDD
from pyspark.sql import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StructField, StructType

from shardyfusion._writer_core import build_categorical_routing_values
from shardyfusion.errors import ShardAssignmentError
from shardyfusion.sharding_types import (
    DB_ID_COL,
    CelShardingSpec,
    HashShardingSpec,
    ShardingSpec,
)
from shardyfusion.vector._distributed import ResolvedVectorRouting
from shardyfusion.vector.sharding import cluster_assign, lsh_assign
from shardyfusion.vector.types import VectorShardingStrategy as VecStrategy


def add_db_id_column(
    df: DataFrame,
    *,
    key_col: str,
    num_dbs: int | None,
    sharding: ShardingSpec,
) -> tuple[DataFrame, ShardingSpec]:
    """Add deterministic db id column and return resolved sharding spec."""

    output_schema = StructType(
        list(df.schema.fields) + [StructField(DB_ID_COL, IntegerType(), False)]
    )

    df_with_db_id: DataFrame
    if isinstance(sharding, HashShardingSpec):
        assert num_dbs is not None, "num_dbs required for HASH sharding"
        _key_col = key_col
        _num_dbs = num_dbs
        _hash_algorithm = sharding.hash_algorithm
        resolved: ShardingSpec = HashShardingSpec(
            hash_algorithm=sharding.hash_algorithm,
            max_keys_per_shard=sharding.max_keys_per_shard,
        )

        def _hash_map_arrow(iterator):  # type: ignore[no-untyped-def]
            import pyarrow as pa  # type: ignore[import-not-found]

            from shardyfusion.routing import hash_db_id

            for batch in iterator:
                keys = batch.column(_key_col).to_pylist()
                db_ids = [hash_db_id(k, _num_dbs, _hash_algorithm) for k in keys]
                yield batch.append_column(DB_ID_COL, pa.array(db_ids, type=pa.int32()))

        df_with_db_id = df.mapInArrow(_hash_map_arrow, output_schema)

    elif isinstance(sharding, CelShardingSpec):
        from shardyfusion.cel import compile_cel

        assert sharding.cel_expr is not None and sharding.cel_columns is not None
        routing_values_for_cel = (
            _discover_categorical_routing_values(
                df,
                cel_expr=sharding.cel_expr,
                cel_columns=dict(sharding.cel_columns),
            )
            if sharding.infer_routing_values_from_data
            else (
                list(sharding.routing_values)
                if sharding.routing_values is not None
                else None
            )
        )
        resolved = CelShardingSpec(
            cel_expr=sharding.cel_expr,
            cel_columns=dict(sharding.cel_columns),
            routing_values=routing_values_for_cel,
            infer_routing_values_from_data=False,
        )

        _cel_expr = sharding.cel_expr
        _cel_cols = dict(sharding.cel_columns)
        _cel_routing_values = routing_values_for_cel

        compile_cel(_cel_expr, _cel_cols)  # validate eagerly on driver

        def _cel_map_arrow(iterator):  # type: ignore[no-untyped-def]
            import pyarrow as pa  # type: ignore[import-not-found]

            from shardyfusion.cel import compile_cel as _compile
            from shardyfusion.cel import route_cel_batch

            _compiled = _compile(_cel_expr, _cel_cols)
            for batch in iterator:
                db_ids = route_cel_batch(
                    _compiled,
                    batch,
                    _cel_routing_values,
                )
                yield batch.append_column(DB_ID_COL, pa.array(db_ids, type=pa.int32()))

        df_with_db_id = df.mapInArrow(_cel_map_arrow, output_schema)

    else:
        raise ShardAssignmentError(
            f"Unsupported sharding strategy: {type(sharding).__name__}"
        )

    # Validate db_id range (skip for CEL direct mode where num_dbs is None)
    if num_dbs is not None:
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
