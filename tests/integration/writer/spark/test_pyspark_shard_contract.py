from __future__ import annotations

import pytest

from shardyfusion.config import HashWriteConfig, ManifestOptions, OutputOptions
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.serde import ValueSpec
from shardyfusion.testing import real_file_adapter_factory
from shardyfusion.writer.spark import write_sharded_by_hash


@pytest.mark.spark
def test_sharded_writer_contract_holds_for_pyspark(spark, tmp_path) -> None:
    rows = [(i, f"payload-{i}".encode()) for i in range(71)]
    df = spark.createDataFrame(rows, ["id", "payload"])

    store = InMemoryManifestStore()
    config = HashWriteConfig(
        num_dbs=5,
        s3_prefix="s3://bucket/prefix",
        manifest=ManifestOptions(store=store),
        adapter_factory=real_file_adapter_factory(str(tmp_path / "object-store")),
        output=OutputOptions(run_id="run-contract-1"),
    )

    result = write_sharded_by_hash(
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

    # Verify manifest content via store
    parsed = store.load_manifest(result.manifest_ref)
    assert parsed.required_build.num_dbs == 5
    assert parsed.required_build.run_id == "run-contract-1"
    assert len(parsed.shards) == 5
