"""Contract tests verifying Ray-computed db_id matches Python routing.

Since the Ray writer uses ``route_hash()`` directly (not Spark SQL),
this verifies end-to-end correctness of the ``add_db_id_column()``
function against the canonical routing function.
"""

from __future__ import annotations

import pytest
import ray
import ray.data

from shardyfusion._writer_core import route_hash
from shardyfusion.sharding_types import (
    DB_ID_COL,
    HashShardingSpec,
    KeyEncoding,
)
from shardyfusion.writer.ray.sharding import add_db_id_column

# Reuse edge-case keys from the main routing contract test suite.
from tests.unit.writer.core.test_routing_contract import (
    EDGE_CASE_KEYS,
    U32_EDGE_CASE_KEYS,
)


@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 7, 8, 16, 64, 128])
def test_ray_python_hash_agreement_u64be(num_dbs: int) -> None:
    """Ray add_db_id_column matches route_hash for u64be hash sharding."""

    ds = ray.data.from_items([{"id": k} for k in EDGE_CASE_KEYS], override_num_blocks=4)
    sharding = HashShardingSpec()

    result_ds, _ = add_db_id_column(
        ds,
        key_col="id",
        num_dbs=num_dbs,
        sharding=sharding,
        key_encoding=KeyEncoding.U64BE,
    )

    for row in result_ds.take_all():
        key = int(row["id"])
        ray_db_id = int(row[DB_ID_COL])
        python_db_id = route_hash(
            key,
            num_dbs=num_dbs,
        )
        assert ray_db_id == python_db_id, (
            f"key={key}, num_dbs={num_dbs}: Ray={ray_db_id}, Python={python_db_id}"
        )


@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 8, 16, 64])
def test_ray_python_hash_agreement_u32be(num_dbs: int) -> None:
    """Ray add_db_id_column matches route_hash for u32be hash sharding."""

    ds = ray.data.from_items(
        [{"id": k} for k in U32_EDGE_CASE_KEYS], override_num_blocks=4
    )
    sharding = HashShardingSpec()

    result_ds, _ = add_db_id_column(
        ds,
        key_col="id",
        num_dbs=num_dbs,
        sharding=sharding,
        key_encoding=KeyEncoding.U32BE,
    )

    for row in result_ds.take_all():
        key = int(row["id"])
        ray_db_id = int(row[DB_ID_COL])
        python_db_id = route_hash(
            key,
            num_dbs=num_dbs,
        )
        assert ray_db_id == python_db_id, (
            f"key={key}, num_dbs={num_dbs}: Ray(u32be)={ray_db_id}, Python={python_db_id}"
        )
