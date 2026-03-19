"""Dask-native sharding helpers."""

import dask.dataframe as dd
import pandas as pd

from shardyfusion._writer_core import route_key
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
    num_dbs: int,
    sharding: ShardingSpec,
    key_encoding: KeyEncoding,
) -> dd.DataFrame:
    """Add deterministic ``_slatedb_db_id`` column via Python routing function.

    Uses the same ``route_key()`` as the reader, guaranteeing the
    sharding invariant without reimplementation.

    For CEL sharding, builds per-row routing contexts from the CEL columns
    so that expressions referencing non-key columns evaluate correctly.
    """

    if sharding.strategy == ShardingStrategy.CEL:
        return _add_db_id_cel(ddf, sharding=sharding)
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
) -> dd.DataFrame:
    def _apply_routing(pdf: pd.DataFrame) -> pd.DataFrame:
        db_ids = pdf[key_col].apply(
            lambda key: route_key(
                key,
                num_dbs=num_dbs,
                sharding=sharding,
                key_encoding=key_encoding,
            )
        )
        return pdf.assign(**{DB_ID_COL: db_ids})

    meta = ddf._meta.assign(**{DB_ID_COL: 0})
    return ddf.map_partitions(_apply_routing, meta=meta)


def _add_db_id_cel(
    ddf: dd.DataFrame,
    *,
    sharding: ShardingSpec,
) -> dd.DataFrame:
    assert sharding.cel_expr is not None and sharding.cel_columns is not None
    _cel_expr = sharding.cel_expr
    _cel_cols = dict(sharding.cel_columns)
    _col_names = list(_cel_cols)
    _boundaries = list(sharding.boundaries) if sharding.boundaries is not None else None

    def _apply_cel_routing(pdf: pd.DataFrame) -> pd.DataFrame:
        from shardyfusion.cel import compile_cel, pandas_rows_to_contexts, route_cel

        compiled = compile_cel(_cel_expr, _cel_cols)
        contexts = pandas_rows_to_contexts(pdf, _cel_cols)
        db_ids = [route_cel(compiled, ctx, _boundaries) for ctx in contexts]
        return pdf.assign(**{DB_ID_COL: db_ids})

    meta = ddf._meta.assign(**{DB_ID_COL: 0})
    return ddf.map_partitions(_apply_cel_routing, meta=meta)
