"""Contract tests verifying Dask-computed db_id matches Python routing.

Since the Dask writer uses ``_route_key()`` directly (not Spark SQL),
this verifies end-to-end correctness of the ``add_db_id_column()``
function against the canonical routing function.
"""

from __future__ import annotations

import pandas as pd
import pytest

dd = pytest.importorskip("dask.dataframe")
import dask  # noqa: E402

from slatedb_spark_sharded._writer_core import _route_key  # noqa: E402
from slatedb_spark_sharded.sharding_types import (  # noqa: E402
    DB_ID_COL,
    KeyEncoding,
    ShardingSpec,
    ShardingStrategy,
)
from slatedb_spark_sharded.writer.dask.sharding import add_db_id_column  # noqa: E402

# Reuse edge-case keys from the main routing contract test suite.
from tests.unit.writer.test_routing_contract import (  # noqa: E402
    EDGE_CASE_KEYS,
    U32_EDGE_CASE_KEYS,
)


@pytest.fixture(autouse=True)
def _synchronous_scheduler():
    with dask.config.set(scheduler="synchronous"):
        yield


@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 7, 8, 16, 64, 128])
def test_dask_python_hash_agreement_u64be(num_dbs: int) -> None:
    """Dask add_db_id_column matches _route_key for u64be hash sharding."""

    pdf = pd.DataFrame({"id": EDGE_CASE_KEYS})
    ddf = dd.from_pandas(pdf, npartitions=4)
    sharding = ShardingSpec(strategy=ShardingStrategy.HASH)

    result = add_db_id_column(
        ddf,
        key_col="id",
        num_dbs=num_dbs,
        sharding=sharding,
        key_encoding=KeyEncoding.U64BE,
    ).compute()

    for _, row in result.iterrows():
        key = int(row["id"])  # convert numpy scalar to Python int
        dask_db_id = int(row[DB_ID_COL])
        python_db_id = _route_key(
            key,
            num_dbs=num_dbs,
            sharding=sharding,
            key_encoding=KeyEncoding.U64BE,
        )
        assert dask_db_id == python_db_id, (
            f"key={key}, num_dbs={num_dbs}: Dask={dask_db_id}, Python={python_db_id}"
        )


@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 8, 16, 64])
def test_dask_python_hash_agreement_u32be(num_dbs: int) -> None:
    """Dask add_db_id_column matches _route_key for u32be hash sharding."""

    pdf = pd.DataFrame({"id": U32_EDGE_CASE_KEYS})
    ddf = dd.from_pandas(pdf, npartitions=4)
    sharding = ShardingSpec(strategy=ShardingStrategy.HASH)

    result = add_db_id_column(
        ddf,
        key_col="id",
        num_dbs=num_dbs,
        sharding=sharding,
        key_encoding=KeyEncoding.U32BE,
    ).compute()

    for _, row in result.iterrows():
        key = int(row["id"])  # convert numpy scalar to Python int
        dask_db_id = int(row[DB_ID_COL])
        python_db_id = _route_key(
            key,
            num_dbs=num_dbs,
            sharding=sharding,
            key_encoding=KeyEncoding.U32BE,
        )
        assert dask_db_id == python_db_id, (
            f"key={key}, num_dbs={num_dbs}: Dask(u32be)={dask_db_id}, Python={python_db_id}"
        )


@pytest.mark.parametrize(
    "boundaries",
    [
        [10],
        [10, 20],
        [10, 20, 35, 50],
        [0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
    ],
)
def test_dask_range_sharding_matches_route_key(boundaries: list[int]) -> None:
    """Dask add_db_id_column matches _route_key for range sharding."""

    num_dbs = len(boundaries) + 1
    sharding = ShardingSpec(
        strategy=ShardingStrategy.RANGE,
        boundaries=boundaries,
    )

    # Build test keys: boundary values and their neighbors
    keys = sorted(
        set(
            [-1, 0, 1, 5, 9, 10, 11, 15, 19, 20, 21, 35, 50, 51, 100, 500, 1000]
            + boundaries
            + [b - 1 for b in boundaries]
            + [b + 1 for b in boundaries]
        )
    )

    pdf = pd.DataFrame({"id": keys})
    ddf = dd.from_pandas(pdf, npartitions=2)

    result = add_db_id_column(
        ddf,
        key_col="id",
        num_dbs=num_dbs,
        sharding=sharding,
        key_encoding=KeyEncoding.U64BE,
    ).compute()

    for _, row in result.iterrows():
        key = int(row["id"])  # convert numpy scalar to Python int
        dask_db_id = int(row[DB_ID_COL])
        python_db_id = _route_key(
            key,
            num_dbs=num_dbs,
            sharding=sharding,
            key_encoding=KeyEncoding.U64BE,
        )
        assert dask_db_id == python_db_id, (
            f"key={key}, boundaries={boundaries}: "
            f"Dask={dask_db_id}, Python={python_db_id}"
        )
