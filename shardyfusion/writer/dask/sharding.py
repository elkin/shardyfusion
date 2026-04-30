"""Dask-native sharding helpers."""

import dask.dataframe as dd
import numpy as np
import pandas as pd

from shardyfusion._writer_core import build_categorical_routing_values, route_hash
from shardyfusion.sharding_types import (
    DB_ID_COL,
    CelShardingSpec,
    HashShardingSpec,
    KeyEncoding,
    ShardingSpec,
)
from shardyfusion.vector._distributed import (
    ResolvedVectorRouting,
    coerce_vector_value,
)
from shardyfusion.vector.sharding import cluster_assign, lsh_assign
from shardyfusion.vector.types import VectorShardingStrategy as VecStrategy


def add_db_id_column(
    ddf: dd.DataFrame,
    *,
    key_col: str,
    num_dbs: int | None,
    sharding: ShardingSpec,
    key_encoding: KeyEncoding,
) -> tuple[dd.DataFrame, ShardingSpec]:
    """Add deterministic ``_shard_id`` column via Python routing function.

    Uses the same ``route_hash()`` / ``route_cel()`` as the reader,
    guaranteeing the sharding invariant without reimplementation.

    For CEL sharding, builds per-row routing contexts from the CEL columns
    so that expressions referencing non-key columns evaluate correctly.
    """

    if isinstance(sharding, CelShardingSpec):
        return _add_db_id_cel(ddf, sharding=sharding)
    assert num_dbs is not None, "num_dbs required for HASH sharding"
    assert isinstance(sharding, HashShardingSpec)
    return _add_db_id_hash(
        ddf,
        key_col=key_col,
        num_dbs=num_dbs,
        sharding=sharding,
        key_encoding=key_encoding,
    )


def _add_db_id_hash(
    ddf: dd.DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    sharding: HashShardingSpec,
    key_encoding: KeyEncoding,
) -> tuple[dd.DataFrame, ShardingSpec]:
    def _apply_routing(pdf: pd.DataFrame) -> pd.DataFrame:
        db_ids = pdf[key_col].apply(
            lambda key: route_hash(
                key, num_dbs=num_dbs, hash_algorithm=sharding.hash_algorithm
            )
        )
        return pdf.assign(**{DB_ID_COL: db_ids})

    meta = ddf._meta.assign(**{DB_ID_COL: 0})
    return ddf.map_partitions(_apply_routing, meta=meta), sharding


def _add_db_id_cel(
    ddf: dd.DataFrame,
    *,
    sharding: ShardingSpec,
) -> tuple[dd.DataFrame, ShardingSpec]:
    assert isinstance(sharding, CelShardingSpec)
    assert sharding.cel_expr is not None and sharding.cel_columns is not None
    resolved = CelShardingSpec(
        cel_expr=sharding.cel_expr,
        cel_columns=dict(sharding.cel_columns),
        routing_values=(
            list(sharding.routing_values)
            if sharding.routing_values is not None
            else None
        ),
        infer_routing_values_from_data=False,
    )
    cel_expr = resolved.cel_expr
    cel_columns = resolved.cel_columns
    assert cel_expr is not None and cel_columns is not None
    if sharding.infer_routing_values_from_data:
        resolved.routing_values = _discover_categorical_routing_values(
            ddf,
            cel_expr=cel_expr,
            cel_columns=cel_columns,
        )

    _cel_expr = cel_expr
    _cel_cols = dict(cel_columns)
    _routing_values = (
        list(resolved.routing_values) if resolved.routing_values is not None else None
    )

    def _apply_cel_routing(pdf: pd.DataFrame) -> pd.DataFrame:
        from shardyfusion.cel import (
            build_categorical_routing_lookup,
            compile_cel,
            pandas_rows_to_contexts,
            route_cel,
        )

        compiled = compile_cel(_cel_expr, _cel_cols)
        contexts = pandas_rows_to_contexts(pdf, _cel_cols)
        routing_lookup = (
            build_categorical_routing_lookup(_routing_values)
            if _routing_values is not None
            else None
        )
        db_ids = [
            route_cel(
                compiled,
                ctx,
                _routing_values,
                lookup=routing_lookup,
            )
            for ctx in contexts
        ]
        return pdf.assign(**{DB_ID_COL: db_ids})

    meta = ddf._meta.assign(**{DB_ID_COL: 0})
    return ddf.map_partitions(_apply_cel_routing, meta=meta), resolved


def _discover_categorical_routing_values(
    ddf: dd.DataFrame,
    *,
    cel_expr: str,
    cel_columns: dict[str, str],
) -> list[int | str | bytes]:
    def _extract_tokens(pdf: pd.DataFrame) -> pd.Series:
        from shardyfusion.cel import compile_cel, pandas_rows_to_contexts

        compiled = compile_cel(cel_expr, cel_columns)
        contexts = pandas_rows_to_contexts(pdf, cel_columns)
        tokens = [compiled.evaluate(ctx) for ctx in contexts]
        return pd.Series(tokens, dtype="object")

    tokens = ddf.map_partitions(_extract_tokens, meta=("routing_token", "object"))
    distinct_tokens = tokens.drop_duplicates()
    return build_categorical_routing_values(distinct_tokens.compute().tolist())


VECTOR_DB_ID_COL = "_vector_db_id"


def _stack_vector_values(values: list[object]) -> np.ndarray:
    """Stack vector-like values into a 2D float32 array."""

    if not values:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([coerce_vector_value(value) for value in values]).astype(
        np.float32, copy=False
    )


def add_vector_db_id_column(
    ddf: dd.DataFrame,
    *,
    vector_col: str,
    routing: ResolvedVectorRouting,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
) -> tuple[dd.DataFrame, int]:
    """Add deterministic _vector_db_id column based on vector routing.

    Uses map_partitions with pandas apply to match the Python writer's
    assign_vector_shard() behavior.

    Returns:
        Tuple of (modified DataFrame, num_dbs)
    """
    strategy = routing.strategy
    num_dbs = routing.num_dbs

    if strategy == VecStrategy.EXPLICIT:
        assert shard_id_col is not None, "shard_id_col required for EXPLICIT"

        def _apply_explicit(pdf: pd.DataFrame) -> pd.DataFrame:
            return pdf.assign(**{VECTOR_DB_ID_COL: pdf[shard_id_col]})

        meta = ddf._meta.assign(**{VECTOR_DB_ID_COL: 0})
        return ddf.map_partitions(_apply_explicit, meta=meta), num_dbs

    elif strategy == VecStrategy.CLUSTER:
        assert routing.centroids is not None, "centroids required for CLUSTER"
        _centroids = routing.centroids
        _metric = routing.metric

        def _apply_cluster(pdf: pd.DataFrame) -> pd.DataFrame:
            from shardyfusion.vector.types import DistanceMetric

            metric = _metric
            if isinstance(metric, str):
                metric = DistanceMetric(metric)
            vectors = _stack_vector_values(pdf[vector_col].tolist())
            db_ids = [cluster_assign(v, _centroids, metric) for v in vectors]
            return pdf.assign(**{VECTOR_DB_ID_COL: db_ids})

        meta = ddf._meta.assign(**{VECTOR_DB_ID_COL: 0})
        return ddf.map_partitions(_apply_cluster, meta=meta), num_dbs

    elif strategy == VecStrategy.LSH:
        assert routing.hyperplanes is not None, "hyperplanes required for LSH"
        _hyperplanes = routing.hyperplanes

        def _apply_lsh(pdf: pd.DataFrame) -> pd.DataFrame:
            vectors = _stack_vector_values(pdf[vector_col].tolist())
            db_ids = [lsh_assign(v, _hyperplanes, num_dbs) for v in vectors]
            return pdf.assign(**{VECTOR_DB_ID_COL: db_ids})

        meta = ddf._meta.assign(**{VECTOR_DB_ID_COL: 0})
        return ddf.map_partitions(_apply_lsh, meta=meta), num_dbs

    elif strategy == VecStrategy.CEL:
        assert routing.compiled_cel is not None, "compiled_cel required for CEL"
        assert routing.cel_expr is not None, "cel_expr required for CEL"
        assert routing_context_cols is not None, "routing_context_cols required for CEL"

        from shardyfusion.cel import (
            build_categorical_routing_lookup,
            compile_cel,
            pandas_rows_to_contexts,
            route_cel,
        )

        _cel_expr = routing.cel_expr
        _cel_cols = dict(routing_context_cols)
        _routing_values = routing.routing_values
        _cel_lookup = (
            build_categorical_routing_lookup(_routing_values)
            if _routing_values is not None
            else None
        )

        def _apply_cel(pdf: pd.DataFrame) -> pd.DataFrame:
            _compiled = compile_cel(_cel_expr, _cel_cols)
            contexts = pandas_rows_to_contexts(pdf, _cel_cols)
            db_ids = [
                route_cel(
                    _compiled,
                    ctx,
                    _routing_values,
                    lookup=_cel_lookup,
                )
                for ctx in contexts
            ]
            return pdf.assign(**{VECTOR_DB_ID_COL: db_ids})

        meta = ddf._meta.assign(**{VECTOR_DB_ID_COL: 0})
        return ddf.map_partitions(_apply_cel, meta=meta), num_dbs

    else:
        raise ValueError(f"Unsupported vector strategy: {strategy}")
