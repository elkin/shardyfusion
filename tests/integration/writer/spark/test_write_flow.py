from __future__ import annotations

import json

import pytest

from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
from shardyfusion.manifest import BuildStats
from shardyfusion.serde import ValueSpec
from shardyfusion.testing import real_file_adapter_factory
from shardyfusion.writer.spark import write_sharded
from tests.helpers.tracking import InMemoryPublisher


@pytest.mark.spark
def test_write_sharded_flow_with_in_memory_publisher(spark, tmp_path) -> None:
    rows = [(i, f"v{i}".encode()) for i in range(40)]
    df = spark.createDataFrame(rows, ["id", "payload"])

    publisher = InMemoryPublisher()
    config = WriteConfig(
        num_dbs=4,
        s3_prefix="s3://bucket/prefix",
        manifest=ManifestOptions(publisher=publisher),
        adapter_factory=real_file_adapter_factory(str(tmp_path / "object-store")),
        output=OutputOptions(run_id="run-test-1"),
    )

    result = write_sharded(
        df,
        config,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
    )

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
