"""Ray Data-native sharding helpers."""

import pyarrow as pa
import ray.data

from shardyfusion._writer_core import route_key
from shardyfusion.errors import ShardAssignmentError
from shardyfusion.sharding_types import (
    DB_ID_COL,
    BoundaryValue,
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


def compute_range_boundaries(
    ds: ray.data.Dataset,
    *,
    key_col: str,
    num_dbs: int,
    rel_error: float = 0.01,
) -> list[BoundaryValue]:
    """Compute approximate quantile boundaries using sampling.

    Takes a sample of the dataset and computes quantiles locally,
    requiring O(1) passes over the dataset instead of O(num_dbs).

    The ``rel_error`` parameter is accepted for API consistency with
    Spark's ``approxQuantile`` but does not directly control the
    sampling accuracy.
    """

    _ = rel_error  # accepted for API consistency

    expected = max(num_dbs - 1, 0)
    if expected == 0:
        return []

    # Sample-based approach: take a sample and compute quantiles locally.
    count = ds.count()
    sample_size = min(10_000, count)

    if sample_size == 0:
        raise ShardAssignmentError(
            "Range sharding requires a non-empty dataset to compute boundaries"
        )

    # Take a sample of keys and sort them locally
    sample_rows = ds.random_shuffle().take(sample_size)
    values = sorted(row[key_col] for row in sample_rows)

    # Compute quantile boundaries from sorted sample
    boundaries: list[BoundaryValue] = []
    for idx in range(1, num_dbs):
        pos = int(idx * len(values) / num_dbs)
        pos = min(pos, len(values) - 1)
        val = values[pos]
        # Convert numpy scalars to Python native types for manifest compatibility
        boundaries.append(val.item() if hasattr(val, "item") else val)

    if len(boundaries) != expected:
        raise ShardAssignmentError(
            f"Range sharding could not derive the expected number of boundaries from "
            f"quantile: expected {expected}, got {len(boundaries)}"
        )

    return boundaries
