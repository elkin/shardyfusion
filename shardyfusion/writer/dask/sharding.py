"""Dask-native sharding helpers."""

import dask.dataframe as dd
import pandas as pd

from shardyfusion._writer_core import build_categorical_routing_values, route_key
from shardyfusion.sharding_types import (
    DB_ID_COL,
    KeyEncoding,
    ShardingSpec,
    ShardingStrategy,
)


def add_db_id_column(
    ddf: dd.DataFrame,
    *,
    key_col: str,
    num_dbs: int | None,
    sharding: ShardingSpec,
    key_encoding: KeyEncoding,
) -> tuple[dd.DataFrame, ShardingSpec]:
    """Add deterministic ``_shard_id`` column via Python routing function.

    Uses the same ``route_key()`` as the reader, guaranteeing the
    sharding invariant without reimplementation.

    For CEL sharding, builds per-row routing contexts from the CEL columns
    so that expressions referencing non-key columns evaluate correctly.
    """

    if sharding.strategy == ShardingStrategy.CEL:
        return _add_db_id_cel(ddf, sharding=sharding)
    assert num_dbs is not None, "num_dbs required for HASH sharding"
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
    sharding: ShardingSpec,
    key_encoding: KeyEncoding,
) -> tuple[dd.DataFrame, ShardingSpec]:
    def _apply_routing(pdf: pd.DataFrame) -> pd.DataFrame:
        db_ids = pdf[key_col].apply(
            lambda key: route_key(
                key,
                num_dbs=num_dbs,
                sharding=sharding,
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
    assert sharding.cel_expr is not None and sharding.cel_columns is not None
    resolved = ShardingSpec(
        strategy=ShardingStrategy.CEL,
        routing_values=list(sharding.routing_values)
        if sharding.routing_values is not None
        else None,
        cel_expr=sharding.cel_expr,
        cel_columns=dict(sharding.cel_columns),
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
