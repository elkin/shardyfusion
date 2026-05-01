"""Contract tests verifying Dask-computed db_id matches Python routing.

Since the Dask writer uses ``route_hash()`` directly (not Spark SQL),
this verifies end-to-end correctness of the ``add_db_id_column_hash()``
function against the canonical routing function.
"""

from __future__ import annotations

import dask.dataframe as dd
import pandas as pd
import pytest

from shardyfusion._writer_core import route_hash
from shardyfusion.sharding_types import (
    DB_ID_COL,
    ShardHashAlgorithm,
)
from shardyfusion.writer.dask.sharding import add_db_id_column_hash

# Reuse edge-case keys from the main routing contract test suite.
from tests.unit.writer.core.test_routing_contract import (
    EDGE_CASE_KEYS,
    U32_EDGE_CASE_KEYS,
)


@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 7, 8, 16, 64, 128])
def test_dask_python_hash_agreement_u64be(num_dbs: int) -> None:
    """Dask add_db_id_column_hash matches route_hash for u64be hash sharding."""

    pdf = pd.DataFrame({"id": EDGE_CASE_KEYS})
    ddf = dd.from_pandas(pdf, npartitions=4)

    result_ddf = add_db_id_column_hash(
        ddf,
        key_col="id",
        num_dbs=num_dbs,
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
    )
    result = result_ddf.compute()

    for _, row in result.iterrows():
        key = int(row["id"])  # convert numpy scalar to Python int
        dask_db_id = int(row[DB_ID_COL])
        python_db_id = route_hash(
            key,
            num_dbs=num_dbs,
        )
        assert dask_db_id == python_db_id, (
            f"key={key}, num_dbs={num_dbs}: Dask={dask_db_id}, Python={python_db_id}"
        )


@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 8, 16, 64])
def test_dask_python_hash_agreement_u32be(num_dbs: int) -> None:
    """Dask add_db_id_column_hash matches route_hash for u32be hash sharding."""

    pdf = pd.DataFrame({"id": U32_EDGE_CASE_KEYS})
    ddf = dd.from_pandas(pdf, npartitions=4)

    result_ddf = add_db_id_column_hash(
        ddf,
        key_col="id",
        num_dbs=num_dbs,
        hash_algorithm=ShardHashAlgorithm.XXH3_64,
    )
    result = result_ddf.compute()

    for _, row in result.iterrows():
        key = int(row["id"])  # convert numpy scalar to Python int
        dask_db_id = int(row[DB_ID_COL])
        python_db_id = route_hash(
            key,
            num_dbs=num_dbs,
        )
        assert dask_db_id == python_db_id, (
            f"key={key}, num_dbs={num_dbs}: Dask(u32be)={dask_db_id}, Python={python_db_id}"
        )
