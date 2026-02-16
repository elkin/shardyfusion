from __future__ import annotations

import pytest
import slatedb

from slatedb_spark_sharded.config import (
    EngineOptions,
    ManifestOptions,
    OutputOptions,
    ShardingOptions,
    SlateDbConfig,
)
from slatedb_spark_sharded.reader import SlateShardedReader
from slatedb_spark_sharded.serde import ValueSpec
from slatedb_spark_sharded.sharding import ShardingSpec, ShardingStrategy
from slatedb_spark_sharded.testing import (
    map_s3_db_url_to_file_url,
    real_file_adapter_factory,
    writer_local_dir_for_db_url,
)
from slatedb_spark_sharded.writer import write_sharded_slatedb


@pytest.mark.spark
def test_reader_refreshes_after_new_writer_batch(
    monkeypatch, spark, tmp_path, local_s3_service
) -> None:
    bucket = local_s3_service["bucket"]
    endpoint_url = local_s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/writer-reader-refresh"
    local_root = str(tmp_path / "writer-local")
    object_store_root = str(tmp_path / "object-store")

    def build_config(run_id: str) -> SlateDbConfig:
        return SlateDbConfig(
            num_dbs=4,
            s3_prefix=s3_prefix,
            key_col="id",
            value_spec=ValueSpec.binary_col("payload"),
            output=OutputOptions(
                run_id=run_id,
                local_root=local_root,
            ),
            sharding=ShardingOptions(
                spec=ShardingSpec(
                    strategy=ShardingStrategy.RANGE,
                    boundaries=[8, 16, 24],
                )
            ),
            engine=EngineOptions(
                slatedb_adapter_factory=real_file_adapter_factory(object_store_root),
            ),
            manifest=ManifestOptions(
                s3_client_config={
                    "endpoint_url": endpoint_url,
                    "region_name": local_s3_service["region_name"],
                    "aws_access_key_id": local_s3_service["aws_access_key_id"],
                    "aws_secret_access_key": local_s3_service["aws_secret_access_key"],
                }
            ),
        )

    df_v1 = spark.createDataFrame(
        [(i, f"old-{i}".encode("utf-8")) for i in range(32)],
        ["id", "payload"],
    )
    result_v1 = write_sharded_slatedb(df_v1, build_config("refresh-run-1"))

    def open_real_reader(*, local_path, db_url, checkpoint_id, env_file, settings):
        _ = (local_path, env_file, settings)
        return slatedb.SlateDBReader(
            writer_local_dir_for_db_url(db_url, local_root),
            url=map_s3_db_url_to_file_url(db_url, object_store_root),
            checkpoint_id=checkpoint_id,
        )

    monkeypatch.setattr(
        "slatedb_spark_sharded.reader._open_slatedb_reader", open_real_reader
    )

    reader = SlateShardedReader(
        s3_prefix=s3_prefix,
        local_root=str(tmp_path / "reader-cache"),
    )
    try:
        assert reader.get(7) == b"old-7"

        df_v2 = spark.createDataFrame(
            [(i, f"new-{i}".encode("utf-8")) for i in range(32)],
            ["id", "payload"],
        )
        result_v2 = write_sharded_slatedb(df_v2, build_config("refresh-run-2"))

        assert result_v1.manifest_ref != result_v2.manifest_ref

        changed = reader.refresh()
        assert changed is True
        assert reader.get(7) == b"new-7"

        unchanged = reader.refresh()
        assert unchanged is False
    finally:
        reader.close()
