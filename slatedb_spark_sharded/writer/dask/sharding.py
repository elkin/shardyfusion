"""Dask-native sharding helpers."""

import dask.dataframe as dd
import pandas as pd

from slatedb_spark_sharded._writer_core import _route_key
from slatedb_spark_sharded.errors import ShardAssignmentError
from slatedb_spark_sharded.sharding_types import (
    DB_ID_COL,
    BoundaryValue,
    KeyEncoding,
    ShardingSpec,
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

    Uses the same ``_route_key()`` as the reader, guaranteeing the
    sharding invariant without reimplementation.
    """

    def _apply_routing(pdf: pd.DataFrame) -> pd.DataFrame:
        db_ids = pdf[key_col].apply(
            lambda key: _route_key(
                key,
                num_dbs=num_dbs,
                sharding=sharding,
                key_encoding=key_encoding,
            )
        )
        return pdf.assign(**{DB_ID_COL: db_ids})

    meta = ddf._meta.assign(**{DB_ID_COL: 0})
    return ddf.map_partitions(_apply_routing, meta=meta)


def compute_range_boundaries(
    ddf: dd.DataFrame,
    *,
    key_col: str,
    num_dbs: int,
    rel_error: float = 0.01,
) -> list[BoundaryValue]:
    """Compute approximate quantile boundaries using Dask.

    The ``rel_error`` parameter is accepted for API consistency with
    Spark's ``approxQuantile`` but is not directly passed to Dask's
    quantile implementation (which uses its own internal approximation).
    """

    _ = rel_error  # Dask quantile uses its own approximation method

    expected = max(num_dbs - 1, 0)
    if expected == 0:
        return []

    probabilities = [idx / num_dbs for idx in range(1, num_dbs)]
    quantiles_series = ddf[key_col].quantile(probabilities).compute()

    boundaries: list[BoundaryValue] = []
    for val in quantiles_series:
        # Convert numpy scalars to Python native types for manifest compatibility
        boundaries.append(val.item() if hasattr(val, "item") else val)

    if len(boundaries) != expected:
        raise ShardAssignmentError(
            f"Range sharding could not derive the expected number of boundaries from "
            f"quantile: expected {expected}, got {len(boundaries)}"
        )

    return boundaries
