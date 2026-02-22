"""Shared S3 test scenarios used by both moto integration and Garage e2e suites.

Each function contains the full test logic (setup, write, assert) but is
S3-backend agnostic. The caller passes the ``LocalS3Service`` dict produced
by the fixture for their backend, plus an optional ``s3_client_config`` for
backends that need explicit connection options (e.g. path-style addressing).

Writer-specific imports (pyspark, writer module) are deferred to function
bodies so that the reader scenario can be collected without pyspark installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import slatedb

from slatedb_spark_sharded.manifest import (
    CurrentPointer,
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from slatedb_spark_sharded.manifest_readers import DefaultS3ManifestReader
from slatedb_spark_sharded.reader import SlateShardedReader
from slatedb_spark_sharded.sharding_types import KeyEncoding, ShardingStrategy
from slatedb_spark_sharded.type_defs import S3ClientConfig

if TYPE_CHECKING:
    from ..conftest import LocalS3Service


# ---------------------------------------------------------------------------
# Scenario 1: Reader loads manifest from S3
# ---------------------------------------------------------------------------


def run_reader_loads_manifest_scenario(
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    s3_client_config: S3ClientConfig | None = None,
) -> None:
    """Reader loads CURRENT + manifest from S3 and performs get/multi_get."""

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/reader-only"
    manifest_ref = f"{s3_prefix}/manifests/run_id=reader-local/manifest"
    current_ref = f"{s3_prefix}/_CURRENT"
    local_root = tmp_path / "reader-cache"
    object_store_root = tmp_path / "object-store"
    object_store_root.mkdir(parents=True, exist_ok=True)

    # Create two SlateDB shards locally
    db0_local = local_root / "shard=00000"
    db1_local = local_root / "shard=00001"
    db0_local.mkdir(parents=True, exist_ok=True)
    db1_local.mkdir(parents=True, exist_ok=True)
    db0_url = f"file://{(object_store_root / 'db0').as_posix()}"
    db1_url = f"file://{(object_store_root / 'db1').as_posix()}"

    db0 = slatedb.SlateDB(str(db0_local), url=db0_url)
    db0.put((1).to_bytes(8, "big", signed=False), b"v1")
    db0.put((8).to_bytes(8, "big", signed=False), b"v8")
    db0.flush_with_options("wal")
    db0_ckpt = db0.create_checkpoint(scope="durable")["id"]
    db0.close()

    db1 = slatedb.SlateDB(str(db1_local), url=db1_url)
    db1.put((10).to_bytes(8, "big", signed=False), b"v10")
    db1.put((15).to_bytes(8, "big", signed=False), b"v15")
    db1.flush_with_options("wal")
    db1_ckpt = db1.create_checkpoint(scope="durable")["id"]
    db1.close()

    # Build manifest + CURRENT payloads
    required = RequiredBuildMeta(
        run_id="reader-local",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=2,
        s3_prefix=s3_prefix,
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.RANGE, boundaries=[10]),
        db_path_template="db={db_id:05d}",
        tmp_prefix="_tmp",
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url=db0_url,
            attempt=0,
            row_count=2,
            min_key=0,
            max_key=9,
            checkpoint_id=db0_ckpt,
            writer_info={},
        ),
        RequiredShardMeta(
            db_id=1,
            db_url=db1_url,
            attempt=0,
            row_count=2,
            min_key=10,
            max_key=None,
            checkpoint_id=db1_ckpt,
            writer_info={},
        ),
    ]
    manifest_payload = json.dumps(
        {
            "required": required.model_dump(mode="json"),
            "shards": [shard.model_dump(mode="json") for shard in shards],
            "custom": {},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    current_payload = json.dumps(
        CurrentPointer(
            manifest_ref=manifest_ref,
            manifest_content_type="application/json",
            run_id="reader-local",
            updated_at="2026-01-01T00:00:00+00:00",
        ).model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    # Upload manifest + CURRENT to S3
    client = s3_service["client"]
    client.put_object(
        Bucket=bucket,
        Key=manifest_ref.split(f"s3://{bucket}/", 1)[1],
        Body=manifest_payload,
        ContentType="application/json",
    )
    client.put_object(
        Bucket=bucket,
        Key=current_ref.split(f"s3://{bucket}/", 1)[1],
        Body=current_payload,
        ContentType="application/json",
    )

    # Build reader kwargs — inject manifest_reader for path-style addressing
    reader_kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(local_root),
    }
    if s3_client_config is not None:
        reader_kwargs["manifest_reader"] = DefaultS3ManifestReader(
            s3_prefix,
            s3_client_config=s3_client_config,
        )

    with SlateShardedReader(**reader_kwargs) as reader:
        assert reader.get(1) == b"v1"
        assert reader.get(10) == b"v10"
        got = reader.multi_get([8, 15, 1, 10])
        assert got[8] == b"v8"
        assert got[15] == b"v15"
        assert got[1] == b"v1"
        assert got[10] == b"v10"


# ---------------------------------------------------------------------------
# Scenario 2: Writer publishes manifest to S3
# ---------------------------------------------------------------------------


def run_writer_publishes_manifest_scenario(
    spark: Any,
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    s3_client_config: S3ClientConfig | None = None,
) -> None:
    """Writer publishes manifest + CURRENT to S3, then reads shards back."""

    from slatedb_spark_sharded.config import ManifestOptions, OutputOptions, WriteConfig
    from slatedb_spark_sharded.serde import ValueSpec
    from slatedb_spark_sharded.sharding_types import ShardingSpec
    from slatedb_spark_sharded.testing import (
        map_s3_db_url_to_file_url,
        real_file_adapter_factory,
        writer_local_dir_for_db_url,
    )
    from slatedb_spark_sharded.writer.spark import write_sharded_spark

    rows = [(i, f"v{i}".encode("utf-8")) for i in range(24)]
    df = spark.createDataFrame(rows, ["id", "payload"])

    bucket = s3_service["bucket"]
    endpoint_url = s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/writer-only"
    local_root = str(tmp_path / "writer-local")
    object_store_root = str(tmp_path / "object-store")

    manifest_s3_config: S3ClientConfig = s3_client_config or {
        "endpoint_url": endpoint_url,
        "region_name": s3_service["region_name"],
        "access_key_id": s3_service["access_key_id"],
        "secret_access_key": s3_service["secret_access_key"],
    }

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.RANGE,
            boundaries=[6, 12, 18],
        ),
        adapter_factory=real_file_adapter_factory(object_store_root),
        manifest=ManifestOptions(s3_client_config=manifest_s3_config),
        output=OutputOptions(
            run_id="writer-local-s3",
            local_root=local_root,
        ),
    )

    result = write_sharded_spark(
        df,
        config,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
    )

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(f"s3://{bucket}/writer-only/manifests/")
    assert result.current_ref == f"s3://{bucket}/writer-only/_CURRENT"

    manifest_bucket, manifest_key = (
        bucket,
        result.manifest_ref.split(f"s3://{bucket}/", 1)[1],
    )
    current_key = result.current_ref.split(f"s3://{bucket}/", 1)[1]

    client = s3_service["client"]
    manifest_obj = client.get_object(Bucket=manifest_bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = json.loads(manifest_obj["Body"].read().decode("utf-8"))
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))

    assert manifest_payload["required"]["run_id"] == "writer-local-s3"
    assert manifest_payload["required"]["num_dbs"] == 4
    assert len(manifest_payload["shards"]) == 4
    assert current_payload["manifest_ref"] == result.manifest_ref
    assert current_payload["manifest_content_type"] == "application/json"

    # Verify each shard was physically written and can be read via real SlateDB.
    for winner in result.winners:
        reader = slatedb.SlateDBReader(
            writer_local_dir_for_db_url(winner.db_url, local_root),
            url=map_s3_db_url_to_file_url(winner.db_url, object_store_root),
            checkpoint_id=winner.checkpoint_id,
        )
        try:
            probe_key = winner.db_id * 6
            got = reader.get(probe_key.to_bytes(8, "big", signed=False))
            assert got == f"v{probe_key}".encode("utf-8")
        finally:
            reader.close()


# ---------------------------------------------------------------------------
# Scenario 3: Reader refreshes after new writer batch
# ---------------------------------------------------------------------------


def run_writer_reader_refresh_scenario(
    spark: Any,
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    s3_client_config: S3ClientConfig | None = None,
) -> None:
    """Writer publishes v1, reader opens, writer publishes v2, reader refreshes."""

    from slatedb_spark_sharded.config import ManifestOptions, OutputOptions, WriteConfig
    from slatedb_spark_sharded.serde import ValueSpec
    from slatedb_spark_sharded.sharding_types import ShardingSpec
    from slatedb_spark_sharded.testing import (
        map_s3_db_url_to_file_url,
        real_file_adapter_factory,
        writer_local_dir_for_db_url,
    )
    from slatedb_spark_sharded.writer.spark import write_sharded_spark

    bucket = s3_service["bucket"]
    endpoint_url = s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/writer-reader-refresh"
    local_root = str(tmp_path / "writer-local")
    object_store_root = str(tmp_path / "object-store")

    manifest_s3_config: S3ClientConfig = s3_client_config or {
        "endpoint_url": endpoint_url,
        "region_name": s3_service["region_name"],
        "access_key_id": s3_service["access_key_id"],
        "secret_access_key": s3_service["secret_access_key"],
    }

    def build_config(run_id: str) -> WriteConfig:
        return WriteConfig(
            num_dbs=4,
            s3_prefix=s3_prefix,
            output=OutputOptions(
                run_id=run_id,
                local_root=local_root,
            ),
            sharding=ShardingSpec(
                strategy=ShardingStrategy.RANGE,
                boundaries=[8, 16, 24],
            ),
            adapter_factory=real_file_adapter_factory(object_store_root),
            manifest=ManifestOptions(s3_client_config=manifest_s3_config),
        )

    df_v1 = spark.createDataFrame(
        [(i, f"old-{i}".encode("utf-8")) for i in range(32)],
        ["id", "payload"],
    )
    result_v1 = write_sharded_spark(
        df_v1,
        build_config("refresh-run-1"),
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
    )

    def open_real_reader(
        *, db_url: str, local_dir: str, checkpoint_id: str | None
    ) -> slatedb.SlateDBReader:
        return slatedb.SlateDBReader(
            writer_local_dir_for_db_url(db_url, local_root),
            url=map_s3_db_url_to_file_url(db_url, object_store_root),
            checkpoint_id=checkpoint_id,
        )

    # Build reader kwargs — inject manifest_reader for path-style addressing
    reader_kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(tmp_path / "reader-cache"),
        "reader_factory": open_real_reader,
    }
    if s3_client_config is not None:
        reader_kwargs["manifest_reader"] = DefaultS3ManifestReader(
            s3_prefix,
            s3_client_config=s3_client_config,
        )

    with SlateShardedReader(**reader_kwargs) as reader:
        assert reader.get(7) == b"old-7"

        df_v2 = spark.createDataFrame(
            [(i, f"new-{i}".encode("utf-8")) for i in range(32)],
            ["id", "payload"],
        )
        result_v2 = write_sharded_spark(
            df_v2,
            build_config("refresh-run-2"),
            key_col="id",
            value_spec=ValueSpec.binary_col("payload"),
        )

        assert result_v1.manifest_ref != result_v2.manifest_ref

        changed = reader.refresh()
        assert changed is True
        assert reader.get(7) == b"new-7"

        unchanged = reader.refresh()
        assert unchanged is False


# ---------------------------------------------------------------------------
# Scenario 4: Python writer publishes manifest to S3
# ---------------------------------------------------------------------------


def run_python_writer_publishes_manifest_scenario(
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    parallel: bool = False,
    s3_client_config: S3ClientConfig | None = None,
) -> None:
    """Python writer publishes manifest + CURRENT to S3, then reads shards back."""

    from slatedb_spark_sharded.config import ManifestOptions, OutputOptions, WriteConfig
    from slatedb_spark_sharded.sharding_types import ShardingSpec
    from slatedb_spark_sharded.testing import (
        map_s3_db_url_to_file_url,
        real_file_adapter_factory,
        writer_local_dir_for_db_url,
    )
    from slatedb_spark_sharded.writer.python import write_sharded

    mode_label = "parallel" if parallel else "sequential"
    records = list(range(24))

    bucket = s3_service["bucket"]
    endpoint_url = s3_service["endpoint_url"]
    s3_prefix = f"s3://{bucket}/python-writer-{mode_label}"
    local_root = str(tmp_path / f"python-writer-local-{mode_label}")
    object_store_root = str(tmp_path / f"python-object-store-{mode_label}")

    manifest_s3_config: S3ClientConfig = s3_client_config or {
        "endpoint_url": endpoint_url,
        "region_name": s3_service["region_name"],
        "access_key_id": s3_service["access_key_id"],
        "secret_access_key": s3_service["secret_access_key"],
    }

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.RANGE,
            boundaries=[6, 12, 18],
        ),
        adapter_factory=real_file_adapter_factory(object_store_root),
        manifest=ManifestOptions(s3_client_config=manifest_s3_config),
        output=OutputOptions(
            run_id=f"python-writer-{mode_label}",
            local_root=local_root,
        ),
    )

    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode("utf-8"),
        parallel=parallel,
    )

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(
        f"s3://{bucket}/python-writer-{mode_label}/manifests/"
    )
    assert result.current_ref == f"s3://{bucket}/python-writer-{mode_label}/_CURRENT"

    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = result.current_ref.split(f"s3://{bucket}/", 1)[1]

    client = s3_service["client"]
    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = json.loads(manifest_obj["Body"].read().decode("utf-8"))
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))

    assert manifest_payload["required"]["run_id"] == f"python-writer-{mode_label}"
    assert manifest_payload["required"]["num_dbs"] == 4
    assert len(manifest_payload["shards"]) == 4
    assert current_payload["manifest_ref"] == result.manifest_ref
    assert current_payload["manifest_content_type"] == "application/json"

    # Verify each shard was physically written and can be read via real SlateDB.
    total_rows = 0
    for winner in result.winners:
        reader = slatedb.SlateDBReader(
            writer_local_dir_for_db_url(winner.db_url, local_root),
            url=map_s3_db_url_to_file_url(winner.db_url, object_store_root),
            checkpoint_id=winner.checkpoint_id,
        )
        try:
            probe_key = winner.db_id * 6
            got = reader.get(probe_key.to_bytes(8, "big", signed=False))
            assert got == f"v{probe_key}".encode("utf-8")
            total_rows += winner.row_count
        finally:
            reader.close()

    assert total_rows == 24
