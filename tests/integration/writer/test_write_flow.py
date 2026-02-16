from __future__ import annotations

import json

import pytest

from slatedb_spark_sharded.config import (
    EngineOptions,
    ManifestOptions,
    OutputOptions,
    SlateDbConfig,
)
from slatedb_spark_sharded.manifest import BuildStats, ManifestArtifact
from slatedb_spark_sharded.publish import ManifestPublisher
from slatedb_spark_sharded.serde import ValueSpec
from slatedb_spark_sharded.testing import real_file_adapter_factory
from slatedb_spark_sharded.writer import write_sharded_slatedb


class InMemoryPublisher(ManifestPublisher):
    def __init__(self) -> None:
        self.objects: dict[str, ManifestArtifact] = {}

    def publish_manifest(
        self, *, name: str, artifact: ManifestArtifact, run_id: str
    ) -> str:
        ref = f"mem://manifests/run_id={run_id}/{name}"
        self.objects[ref] = artifact
        return ref

    def publish_current(self, *, name: str, artifact: ManifestArtifact) -> str | None:
        ref = f"mem://{name}"
        self.objects[ref] = artifact
        return ref


@pytest.mark.spark
def test_write_sharded_flow_with_in_memory_publisher(spark, tmp_path) -> None:
    rows = [(i, f"v{i}".encode("utf-8")) for i in range(40)]
    df = spark.createDataFrame(rows, ["id", "payload"])

    publisher = InMemoryPublisher()
    config = SlateDbConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        manifest=ManifestOptions(publisher=publisher),
        engine=EngineOptions(
            slatedb_adapter_factory=real_file_adapter_factory(
                str(tmp_path / "object-store")
            )
        ),
        output=OutputOptions(run_id="run-test-1"),
    )

    result = write_sharded_slatedb(df, config)

    assert result.run_id == "run-test-1"
    assert len(result.winners) == 4
    assert [item.db_id for item in result.winners] == [0, 1, 2, 3]
    assert result.manifest_ref.startswith("mem://manifests/")
    assert result.current_ref == "mem://_CURRENT"
    assert isinstance(result.stats, BuildStats)
    assert result.stats.num_winners == 4

    current_artifact = publisher.objects[result.current_ref]
    current_payload = json.loads(current_artifact.payload.decode("utf-8"))
    assert current_payload["manifest_ref"] == result.manifest_ref
    assert current_payload["run_id"] == result.run_id
