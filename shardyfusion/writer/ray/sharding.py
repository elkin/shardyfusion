"""Ray Data-native sharding helpers."""

import pyarrow as pa
import ray.data

from shardyfusion._writer_core import route_key
from shardyfusion.sharding_types import (
    DB_ID_COL,
    KeyEncoding,
    ShardingSpec,
    ShardingStrategy,
)


def add_db_id_column(
    ds: ray.data.Dataset,
    *,
    key_col: str,
    num_dbs: int | None,
    sharding: ShardingSpec,
    key_encoding: KeyEncoding,
) -> ray.data.Dataset:
    """Add deterministic ``_shard_id`` column via Python routing function.

    Uses the same ``route_key()`` as the reader, guaranteeing the
    sharding invariant without reimplementation.

    Uses Arrow batch format to avoid Arrow->pandas->Arrow round-trip.
    For CEL sharding, delegates to ``route_cel_batch()`` which evaluates
    the CEL expression with the full row context (supporting non-key columns).
    """

    if sharding.strategy == ShardingStrategy.CEL:
        return _add_db_id_cel(ds, sharding=sharding)
    assert num_dbs is not None, "num_dbs required for HASH sharding"
    return _add_db_id_hash(
        ds,
        key_col=key_col,
        num_dbs=num_dbs,
        sharding=sharding,
        key_encoding=key_encoding,
    )


def _add_db_id_hash(
    ds: ray.data.Dataset,
    *,
    key_col: str,
    num_dbs: int,
    sharding: ShardingSpec,
    key_encoding: KeyEncoding,
) -> ray.data.Dataset:
    def _apply_routing(table: pa.Table) -> pa.Table:
        keys = table.column(key_col).to_pylist()
        db_ids = [
            route_key(
                key,
                num_dbs=num_dbs,
                sharding=sharding,
                key_encoding=key_encoding,
            )
            for key in keys
        ]
        return table.append_column(DB_ID_COL, pa.array(db_ids, type=pa.int64()))

    return ds.map_batches(_apply_routing, batch_format="pyarrow", zero_copy_batch=True)


def _add_db_id_cel(
    ds: ray.data.Dataset,
    *,
    sharding: ShardingSpec,
) -> ray.data.Dataset:
    assert sharding.cel_expr is not None and sharding.cel_columns is not None
    _cel_expr = sharding.cel_expr
    _cel_cols = dict(sharding.cel_columns)
    _boundaries = list(sharding.boundaries) if sharding.boundaries is not None else None

    def _apply_cel_routing(table: pa.Table) -> pa.Table:
        from shardyfusion.cel import compile_cel, route_cel_batch

        compiled = compile_cel(_cel_expr, _cel_cols)
        db_ids = route_cel_batch(compiled, table, _boundaries)
        return table.append_column(DB_ID_COL, pa.array(db_ids, type=pa.int64()))

    return ds.map_batches(
        _apply_cel_routing, batch_format="pyarrow", zero_copy_batch=True
    )
