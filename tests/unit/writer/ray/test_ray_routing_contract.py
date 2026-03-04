"""Contract tests verifying Ray-computed db_id matches Python routing.

Since the Ray writer uses ``route_key()`` directly (not Spark SQL),
this verifies end-to-end correctness of the ``add_db_id_column()``
function against the canonical routing function.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
ray_data = pytest.importorskip("ray.data")
import ray  # noqa: E402

from shardyfusion._writer_core import route_key  # noqa: E402
from shardyfusion.sharding_types import (  # noqa: E402
    DB_ID_COL,
    KeyEncoding,
    ShardingSpec,
    ShardingStrategy,
)
from shardyfusion.writer.ray.sharding import add_db_id_column  # noqa: E402

# Reuse edge-case keys from the main routing contract test suite.
from tests.unit.writer.test_routing_contract import (  # noqa: E402
    EDGE_CASE_KEYS,
    U32_EDGE_CASE_KEYS,
)


@pytest.fixture(scope="session", autouse=True)
def _ray_init():
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 7, 8, 16, 64, 128])
def test_ray_python_hash_agreement_u64be(num_dbs: int) -> None:
    """Ray add_db_id_column matches route_key for u64be hash sharding."""

    ds = ray.data.from_items([{"id": k} for k in EDGE_CASE_KEYS], override_num_blocks=4)
    sharding = ShardingSpec(strategy=ShardingStrategy.HASH)

    result_ds = add_db_id_column(
        ds,
        key_col="id",
        num_dbs=num_dbs,
        sharding=sharding,
        key_encoding=KeyEncoding.U64BE,
    )

    for row in result_ds.take_all():
        key = int(row["id"])
        ray_db_id = int(row[DB_ID_COL])
        python_db_id = route_key(
            key,
            num_dbs=num_dbs,
            sharding=sharding,
            key_encoding=KeyEncoding.U64BE,
        )
        assert ray_db_id == python_db_id, (
            f"key={key}, num_dbs={num_dbs}: Ray={ray_db_id}, Python={python_db_id}"
        )


@pytest.mark.parametrize("num_dbs", [1, 2, 3, 5, 8, 16, 64])
def test_ray_python_hash_agreement_u32be(num_dbs: int) -> None:
    """Ray add_db_id_column matches route_key for u32be hash sharding."""

    ds = ray.data.from_items(
        [{"id": k} for k in U32_EDGE_CASE_KEYS], override_num_blocks=4
    )
    sharding = ShardingSpec(strategy=ShardingStrategy.HASH)

    result_ds = add_db_id_column(
        ds,
        key_col="id",
        num_dbs=num_dbs,
        sharding=sharding,
        key_encoding=KeyEncoding.U32BE,
    )

    for row in result_ds.take_all():
        key = int(row["id"])
        ray_db_id = int(row[DB_ID_COL])
        python_db_id = route_key(
            key,
            num_dbs=num_dbs,
            sharding=sharding,
            key_encoding=KeyEncoding.U32BE,
        )
        assert ray_db_id == python_db_id, (
            f"key={key}, num_dbs={num_dbs}: Ray(u32be)={ray_db_id}, Python={python_db_id}"
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
def test_ray_range_sharding_matches_route_key(boundaries: list[int]) -> None:
    """Ray add_db_id_column matches route_key for range sharding."""

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

    ds = ray.data.from_items([{"id": k} for k in keys], override_num_blocks=2)

    result_ds = add_db_id_column(
        ds,
        key_col="id",
        num_dbs=num_dbs,
        sharding=sharding,
        key_encoding=KeyEncoding.U64BE,
    )

    for row in result_ds.take_all():
        key = int(row["id"])
        ray_db_id = int(row[DB_ID_COL])
        python_db_id = route_key(
            key,
            num_dbs=num_dbs,
            sharding=sharding,
            key_encoding=KeyEncoding.U64BE,
        )
        assert ray_db_id == python_db_id, (
            f"key={key}, boundaries={boundaries}: "
            f"Ray={ray_db_id}, Python={python_db_id}"
        )
