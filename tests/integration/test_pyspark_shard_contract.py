from __future__ import annotations

import json

import pytest

from slatedb_spark_sharded.config import EngineOptions, ManifestOptions, OutputOptions, SlateDbConfig
from slatedb_spark_sharded.manifest import ManifestArtifact
from slatedb_spark_sharded.publish import ManifestPublisher
from slatedb_spark_sharded.serde import ValueSpec
from slatedb_spark_sharded.testing import fake_adapter_factory
from slatedb_spark_sharded.writer import write_sharded_slatedb


class InMemoryPublisher(ManifestPublisher):
    def __init__(self) -> None:
        self.objects: dict[str, ManifestArtifact] = {}

    def publish_manifest(self, *, name: str, artifact: ManifestArtifact, run_id: str) -> str:
        ref = f"mem://manifests/run_id={run_id}/{name}"
        self.objects[ref] = artifact
        return ref

    def publish_current(self, *, name: str, artifact: ManifestArtifact) -> str | None:
        ref = f"mem://{name}"
        self.objects[ref] = artifact
        return ref


@pytest.mark.spark
def test_sharded_writer_contract_holds_for_pyspark(spark) -> None:
    rows = [(i, f"payload-{i}".encode("utf-8")) for i in range(71)]
    df = spark.createDataFrame(rows, ["id", "payload"])

    publisher = InMemoryPublisher()
    config = SlateDbConfig(
        num_dbs=5,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        manifest=ManifestOptions(publisher=publisher),
        engine=EngineOptions(slatedb_adapter_factory=fake_adapter_factory),
        output=OutputOptions(run_id="run-contract-1"),
    )

    result = write_sharded_slatedb(df, config)

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
