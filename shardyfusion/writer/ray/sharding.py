"""Ray Data-native sharding helpers."""

import numpy as np
import pyarrow as pa
import ray.data

from shardyfusion._writer_core import build_categorical_routing_values, route_hash
from shardyfusion.sharding_types import (
    DB_ID_COL,
    CelShardingSpec,
    ShardHashAlgorithm,
)
from shardyfusion.vector._distributed import ResolvedVectorRouting
from shardyfusion.vector.sharding import cluster_assign, lsh_assign
from shardyfusion.vector.types import VectorShardingStrategy as VecStrategy


def add_db_id_column_hash(
    ds: ray.data.Dataset,
    *,
    key_col: str,
    num_dbs: int,
    hash_algorithm: ShardHashAlgorithm,
) -> ray.data.Dataset:
    """Add deterministic ``_shard_id`` column via hash routing."""

    def _apply_routing(table: pa.Table) -> pa.Table:
        keys = table.column(key_col).to_pylist()
        db_ids = [
            route_hash(key, num_dbs=num_dbs, hash_algorithm=hash_algorithm)
            for key in keys
        ]
        return table.append_column(DB_ID_COL, pa.array(db_ids, type=pa.int64()))

    return ds.map_batches(_apply_routing, batch_format="pyarrow", zero_copy_batch=True)


def add_db_id_column_cel(
    ds: ray.data.Dataset,
    *,
    cel_expr: str,
    cel_columns: dict[str, str],
    routing_values: list[int | str | bytes] | None,
    infer_routing_values_from_data: bool,
) -> tuple[ray.data.Dataset, CelShardingSpec]:
    """Add deterministic ``_shard_id`` column via CEL routing.

    Uses Arrow batch format to avoid Arrow->pandas->Arrow round-trip.
    Delegates to ``route_cel_batch()`` which evaluates the CEL expression
    with the full row context (supporting non-key columns).
    """

    resolved = CelShardingSpec(
        cel_expr=cel_expr,
        cel_columns=dict(cel_columns),
        routing_values=list(routing_values) if routing_values is not None else None,
        infer_routing_values_from_data=False,
    )
    _cel_expr = resolved.cel_expr
    _cel_cols = dict(resolved.cel_columns)
    assert _cel_expr is not None and _cel_cols is not None
    if infer_routing_values_from_data:
        resolved.routing_values = _discover_categorical_routing_values(
            ds,
            cel_expr=_cel_expr,
            cel_columns=_cel_cols,
        )

    _routing_values = (
        list(resolved.routing_values) if resolved.routing_values is not None else None
    )

    def _apply_cel_routing(table: pa.Table) -> pa.Table:
        from shardyfusion.cel import compile_cel, route_cel_batch

        compiled = compile_cel(_cel_expr, _cel_cols)
        db_ids = route_cel_batch(compiled, table, _routing_values)
        return table.append_column(DB_ID_COL, pa.array(db_ids, type=pa.int64()))

    return (
        ds.map_batches(
            _apply_cel_routing, batch_format="pyarrow", zero_copy_batch=True
        ),
        resolved,
    )


def _discover_categorical_routing_values(
    ds: ray.data.Dataset,
    *,
    cel_expr: str,
    cel_columns: dict[str, str],
) -> list[int | str | bytes]:
    def _extract_tokens(table: pa.Table) -> pa.Table:
        from shardyfusion.cel import compile_cel, evaluate_cel_arrow_batch

        compiled = compile_cel(cel_expr, cel_columns)
        tokens = evaluate_cel_arrow_batch(compiled, table)
        return pa.table({"routing_token": tokens})

    token_ds = ds.map_batches(
        _extract_tokens,
        batch_format="pyarrow",
        zero_copy_batch=True,
    )
    return build_categorical_routing_values(token_ds.unique("routing_token"))


VECTOR_DB_ID_COL = "_vector_db_id"


def add_vector_db_id_column(
    ds: ray.data.Dataset,
    *,
    vector_col: str,
    routing: ResolvedVectorRouting,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
) -> tuple[ray.data.Dataset, int]:
    """Add deterministic _vector_db_id column based on vector routing.

    Uses map_batches with Arrow batch format to match the Python writer's
    assign_vector_shard() behavior.

    Returns:
        Tuple of (modified Dataset, num_dbs)
    """
    strategy = routing.strategy
    num_dbs = routing.num_dbs

    if strategy == VecStrategy.EXPLICIT:
        assert shard_id_col is not None, "shard_id_col required for EXPLICIT"

        def _apply_explicit(table: pa.Table) -> pa.Table:
            shard_ids = table.column(shard_id_col).to_pylist()
            return table.append_column(
                VECTOR_DB_ID_COL, pa.array(shard_ids, type=pa.int64())
            )

        return (
            ds.map_batches(
                _apply_explicit, batch_format="pyarrow", zero_copy_batch=True
            ),
            num_dbs,
        )

    elif strategy == VecStrategy.CLUSTER:
        assert routing.centroids is not None, "centroids required for CLUSTER"
        _centroids = routing.centroids
        _metric = routing.metric

        def _apply_cluster(table: pa.Table) -> pa.Table:
            from shardyfusion.vector.types import DistanceMetric

            metric = _metric
            if isinstance(metric, str):
                metric = DistanceMetric(metric)
            vectors = table.column(vector_col).to_pylist()
            db_ids = [
                cluster_assign(np.asarray(v, dtype=np.float32), _centroids, metric)
                for v in vectors
            ]
            return table.append_column(
                VECTOR_DB_ID_COL, pa.array(db_ids, type=pa.int64())
            )

        return (
            ds.map_batches(
                _apply_cluster, batch_format="pyarrow", zero_copy_batch=True
            ),
            num_dbs,
        )

    elif strategy == VecStrategy.LSH:
        assert routing.hyperplanes is not None, "hyperplanes required for LSH"
        _hyperplanes = routing.hyperplanes

        def _apply_lsh(table: pa.Table) -> pa.Table:
            vectors = table.column(vector_col).to_pylist()
            db_ids = [
                lsh_assign(np.asarray(v, dtype=np.float32), _hyperplanes, num_dbs)
                for v in vectors
            ]
            return table.append_column(
                VECTOR_DB_ID_COL, pa.array(db_ids, type=pa.int64())
            )

        return (
            ds.map_batches(_apply_lsh, batch_format="pyarrow", zero_copy_batch=True),
            num_dbs,
        )

    elif strategy == VecStrategy.CEL:
        assert routing.compiled_cel is not None, "compiled_cel required for CEL"
        assert routing.cel_expr is not None, "cel_expr required for CEL"
        assert routing_context_cols is not None, "routing_context_cols required for CEL"

        from shardyfusion.cel import compile_cel, route_cel_batch

        _cel_expr = routing.cel_expr
        _cel_cols = dict(routing_context_cols)
        _routing_values = routing.routing_values

        def _apply_cel(table: pa.Table) -> pa.Table:
            _compiled = compile_cel(_cel_expr, _cel_cols)
            db_ids = route_cel_batch(_compiled, table, _routing_values)
            return table.append_column(
                VECTOR_DB_ID_COL, pa.array(db_ids, type=pa.int64())
            )

        return (
            ds.map_batches(_apply_cel, batch_format="pyarrow", zero_copy_batch=True),
            num_dbs,
        )

    else:
        raise ValueError(f"Unsupported vector strategy: {strategy}")
