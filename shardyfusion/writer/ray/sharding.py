"""Ray Data-native sharding helpers."""

import pyarrow as pa
import ray.data

from shardyfusion._writer_core import build_categorical_routing_values, route_key
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
) -> tuple[ray.data.Dataset, ShardingSpec]:
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
) -> tuple[ray.data.Dataset, ShardingSpec]:
    def _apply_routing(table: pa.Table) -> pa.Table:
        keys = table.column(key_col).to_pylist()
        db_ids = [
            route_key(
                key,
                num_dbs=num_dbs,
                sharding=sharding,
            )
            for key in keys
        ]
        return table.append_column(DB_ID_COL, pa.array(db_ids, type=pa.int64()))

    return (
        ds.map_batches(_apply_routing, batch_format="pyarrow", zero_copy_batch=True),
        sharding,
    )


def _add_db_id_cel(
    ds: ray.data.Dataset,
    *,
    sharding: ShardingSpec,
) -> tuple[ray.data.Dataset, ShardingSpec]:
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
            ds,
            cel_expr=cel_expr,
            cel_columns=cel_columns,
        )

    _cel_expr = cel_expr
    _cel_cols = dict(cel_columns)
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
