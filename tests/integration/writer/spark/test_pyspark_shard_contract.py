from __future__ import annotations

import json

import pytest

from slatedb_spark_sharded.config import ManifestOptions, OutputOptions, WriteConfig
from slatedb_spark_sharded.serde import ValueSpec
from slatedb_spark_sharded.testing import real_file_adapter_factory
from slatedb_spark_sharded.writer.spark import write_sharded
from tests.helpers.tracking import InMemoryPublisher


@pytest.mark.spark
def test_sharded_writer_contract_holds_for_pyspark(spark, tmp_path) -> None:
    rows = [(i, f"payload-{i}".encode("utf-8")) for i in range(71)]
    df = spark.createDataFrame(rows, ["id", "payload"])

    publisher = InMemoryPublisher()
    config = WriteConfig(
        num_dbs=5,
        s3_prefix="s3://bucket/prefix",
        manifest=ManifestOptions(publisher=publisher),
        adapter_factory=real_file_adapter_factory(str(tmp_path / "object-store")),
        output=OutputOptions(run_id="run-contract-1"),
    )

    result = write_sharded(
        df,
        config,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
    )

    assert len(result.winners) == 5
    assert [winner.db_id for winner in result.winners] == [0, 1, 2, 3, 4]
    assert len({winner.db_url for winner in result.winners}) == 5
    assert sum(winner.row_count for winner in result.winners) == len(rows)
    assert all("run_id=run-contract-1" in winner.db_url for winner in result.winners)
    assert all("/attempt=" in winner.db_url for winner in result.winners)

    manifest = json.loads(result.manifest_artifact.payload.decode("utf-8"))
    assert manifest["required"]["num_dbs"] == 5
    assert manifest["required"]["run_id"] == "run-contract-1"
    assert len(manifest["shards"]) == 5
