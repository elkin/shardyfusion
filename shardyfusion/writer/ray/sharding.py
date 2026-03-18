"""Ray Data-native sharding helpers."""

import pyarrow as pa
import ray.data

from shardyfusion._writer_core import route_key
from shardyfusion.sharding_types import (
    DB_ID_COL,
    KeyEncoding,
    ShardingSpec,
)


def add_db_id_column(
    ds: ray.data.Dataset,
    *,
    key_col: str,
    num_dbs: int,
    sharding: ShardingSpec,
    key_encoding: KeyEncoding,
) -> ray.data.Dataset:
    """Add deterministic ``_slatedb_db_id`` column via Python routing function.

    Uses the same ``route_key()`` as the reader, guaranteeing the
    sharding invariant without reimplementation.

    Uses Arrow batch format to avoid Arrow->pandas->Arrow round-trip.
    """

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
