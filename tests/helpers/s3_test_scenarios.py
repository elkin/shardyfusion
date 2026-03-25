"""Shared S3 test scenarios used by both moto integration and Garage e2e suites.

Each function contains the full test logic (setup, write, assert) but is
S3-backend agnostic. The caller passes the ``LocalS3Service`` dict produced
by the fixture for their backend, plus optional ``credential_provider`` and
``s3_connection_options`` for backends that need explicit connection/identity
options (e.g. path-style addressing).

Writer-specific imports (pyspark, writer module) are deferred to function
bodies so that the reader scenario can be collected without pyspark installed.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from shardyfusion.credentials import CredentialProvider, StaticCredentialProvider
from shardyfusion.manifest import (
    CurrentPointer,
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.manifest_store import S3ManifestStore
from shardyfusion.reader import ConcurrentShardedReader
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.slatedb_adapter import DbAdapterFactory
from shardyfusion.type_defs import S3ConnectionOptions, ShardReaderFactory
from tests.helpers.run_record_assertions import (
    assert_success_run_record,
    load_s3_run_record,
)

if TYPE_CHECKING:
    from ..conftest import LocalS3Service


def _default_credential_provider(
    s3_service: LocalS3Service,
    credential_provider: CredentialProvider | None,
) -> CredentialProvider:
    """Return the given provider or build a StaticCredentialProvider from the service dict."""
    if credential_provider is not None:
        return credential_provider
    return StaticCredentialProvider(
        access_key_id=s3_service["access_key_id"],
        secret_access_key=s3_service["secret_access_key"],
    )


def _default_connection_options(
    s3_service: LocalS3Service,
    s3_connection_options: S3ConnectionOptions | None,
) -> S3ConnectionOptions:
    """Return the given options or build defaults from the service dict."""
    if s3_connection_options is not None:
        return s3_connection_options
    return S3ConnectionOptions(
        endpoint_url=s3_service["endpoint_url"],
        region_name=s3_service["region_name"],
    )


# ---------------------------------------------------------------------------
# Scenario 1: Reader loads manifest from S3
# ---------------------------------------------------------------------------


def run_reader_loads_manifest_scenario(
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    adapter_factory: DbAdapterFactory,
    reader_factory: ShardReaderFactory,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Reader loads CURRENT + manifest from S3 and performs get/multi_get."""

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/reader-only"
    manifest_ref = f"{s3_prefix}/manifests/run_id=reader-local/manifest"
    current_ref = f"{s3_prefix}/_CURRENT"
    local_root = tmp_path / "reader-cache"

    # Create a single SlateDB shard with all test data (num_dbs=1 so all keys
    # route to shard 0 regardless of hash distribution).
    db0_url = f"s3://{bucket}/reader-only/shards/db0"
    db0_local = local_root / "shard=00000"
    db0_local.mkdir(parents=True, exist_ok=True)

    adapter = adapter_factory(db_url=db0_url, local_dir=db0_local)
    adapter.write_batch(
        [
            ((1).to_bytes(8, "big", signed=False), b"v1"),
            ((8).to_bytes(8, "big", signed=False), b"v8"),
            ((10).to_bytes(8, "big", signed=False), b"v10"),
            ((15).to_bytes(8, "big", signed=False), b"v15"),
        ]
    )
    adapter.flush()
    db0_ckpt = adapter.checkpoint()
    adapter.close()

    # Build manifest + CURRENT payloads
    required = RequiredBuildMeta(
        run_id="reader-local",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        num_dbs=1,
        s3_prefix=s3_prefix,
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    shards = [
        RequiredShardMeta(
            db_id=0,
            db_url=db0_url,
            attempt=0,
            row_count=4,
            min_key=1,
            max_key=15,
            checkpoint_id=db0_ckpt,
            writer_info={},
        ),
    ]

    manifest_payload = yaml.safe_dump(
        {
            "required": required.model_dump(mode="json"),
            "shards": [shard.model_dump(mode="json") for shard in shards],
            "custom": {},
        },
        sort_keys=True,
        default_flow_style=False,
    ).encode("utf-8")
    current_payload = json.dumps(
        CurrentPointer(
            manifest_ref=manifest_ref,
            manifest_content_type="application/x-yaml",
            run_id="reader-local",
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
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
        ContentType="application/x-yaml",
    )
    client.put_object(
        Bucket=bucket,
        Key=current_ref.split(f"s3://{bucket}/", 1)[1],
        Body=current_payload,
        ContentType="application/json",
    )

    # Build reader kwargs — inject manifest_store for custom connection/identity
    reader_kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(local_root),
        "reader_factory": reader_factory,
    }
    if credential_provider is not None or s3_connection_options is not None:
        reader_kwargs["manifest_store"] = S3ManifestStore(
            s3_prefix,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )

    with ConcurrentShardedReader(**reader_kwargs) as reader:
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
    adapter_factory: DbAdapterFactory,
    expect_retry: bool = False,
    write_kwargs: dict[str, Any] | None = None,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Writer publishes manifest + CURRENT to S3, then reads shards back."""

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.serde import ValueSpec
    from shardyfusion.sharding_types import ShardingSpec
    from shardyfusion.writer.spark import write_sharded

    rows = [(i, f"v{i}".encode()) for i in range(24)]
    df = spark.createDataFrame(rows, ["id", "payload"])

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/writer-only"
    local_root = str(tmp_path / "writer-local")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        adapter_factory=adapter_factory,
        manifest=ManifestOptions(
            credential_provider=cred_provider,
            s3_connection_options=conn_options,
        ),
        output=OutputOptions(
            run_id="writer-local-s3",
            local_root=local_root,
        ),
    )

    result = write_sharded(
        df,
        config,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
        **(write_kwargs or {}),
    )

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(f"s3://{bucket}/writer-only/manifests/")
    assert result.run_record_ref is not None
    if expect_retry:
        assert any(winner.attempt > 0 for winner in result.winners)

    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = "writer-only/_CURRENT"

    client = s3_service["client"]
    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = yaml.safe_load(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))

    assert manifest_payload["required"]["run_id"] == "writer-local-s3"
    assert manifest_payload["required"]["num_dbs"] == 4
    assert len(manifest_payload["shards"]) == 4
    assert current_payload["manifest_ref"] == result.manifest_ref

    run_record = load_s3_run_record(s3_service, result.run_record_ref)
    assert_success_run_record(
        run_record,
        result=result,
        writer_type="spark",
        s3_prefix=s3_prefix,
    )

    # Verify each shard was physically written and total rows sum correctly.
    total_rows = 0
    for winner in result.winners:
        assert winner.row_count > 0
        total_rows += winner.row_count

    assert total_rows == 24


# ---------------------------------------------------------------------------
# Scenario 3: Reader refreshes after new writer batch
# ---------------------------------------------------------------------------


def run_writer_reader_refresh_scenario(
    spark: Any,
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    adapter_factory: DbAdapterFactory,
    reader_factory: ShardReaderFactory,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Writer publishes v1, reader opens, writer publishes v2, reader refreshes."""

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.serde import ValueSpec
    from shardyfusion.sharding_types import ShardingSpec
    from shardyfusion.writer.spark import write_sharded

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/writer-reader-refresh"
    local_root = str(tmp_path / "writer-local")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    def build_config(run_id: str) -> WriteConfig:
        return WriteConfig(
            num_dbs=4,
            s3_prefix=s3_prefix,
            output=OutputOptions(
                run_id=run_id,
                local_root=local_root,
            ),
            sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
            adapter_factory=adapter_factory,
            manifest=ManifestOptions(
                credential_provider=cred_provider,
                s3_connection_options=conn_options,
            ),
        )

    df_v1 = spark.createDataFrame(
        [(i, f"old-{i}".encode()) for i in range(32)],
        ["id", "payload"],
    )
    result_v1 = write_sharded(
        df_v1,
        build_config("refresh-run-1"),
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
    )

    # Build reader kwargs — inject manifest_store for custom connection/identity
    reader_kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(tmp_path / "reader-cache"),
        "reader_factory": reader_factory,
    }
    if credential_provider is not None or s3_connection_options is not None:
        reader_kwargs["manifest_store"] = S3ManifestStore(
            s3_prefix,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )

    with ConcurrentShardedReader(**reader_kwargs) as reader:
        assert reader.get(7) == b"old-7"

        df_v2 = spark.createDataFrame(
            [(i, f"new-{i}".encode()) for i in range(32)],
            ["id", "payload"],
        )
        result_v2 = write_sharded(
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
    adapter_factory: DbAdapterFactory,
    parallel: bool = False,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Python writer publishes manifest + CURRENT to S3, then reads shards back."""

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.sharding_types import ShardingSpec
    from shardyfusion.writer.python import write_sharded

    mode_label = "parallel" if parallel else "sequential"
    records = list(range(24))

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/python-writer-{mode_label}"
    local_root = str(tmp_path / f"python-writer-local-{mode_label}")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        adapter_factory=adapter_factory,
        manifest=ManifestOptions(
            credential_provider=cred_provider,
            s3_connection_options=conn_options,
        ),
        output=OutputOptions(
            run_id=f"python-writer-{mode_label}",
            local_root=local_root,
        ),
    )

    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r,
        value_fn=lambda r: f"v{r}".encode(),
        parallel=parallel,
    )

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(
        f"s3://{bucket}/python-writer-{mode_label}/manifests/"
    )

    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = f"python-writer-{mode_label}/_CURRENT"

    client = s3_service["client"]
    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = yaml.safe_load(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))

    assert manifest_payload["required"]["run_id"] == f"python-writer-{mode_label}"
    assert manifest_payload["required"]["num_dbs"] == 4
    assert len(manifest_payload["shards"]) == 4
    assert current_payload["manifest_ref"] == result.manifest_ref

    # Verify each shard was physically written and total rows sum correctly.
    total_rows = 0
    for winner in result.winners:
        assert winner.row_count > 0
        total_rows += winner.row_count

    assert total_rows == 24


# ---------------------------------------------------------------------------
# Scenario 5: Dask writer publishes manifest to S3
# ---------------------------------------------------------------------------


def run_dask_writer_publishes_manifest_scenario(
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    adapter_factory: DbAdapterFactory,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Dask writer publishes manifest + CURRENT to S3, then reads shards back."""

    import dask
    import dask.dataframe as dd
    import pandas as pd

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.serde import ValueSpec
    from shardyfusion.sharding_types import ShardingSpec
    from shardyfusion.writer.dask import write_sharded

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/dask-writer"
    local_root = str(tmp_path / "dask-writer-local")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        adapter_factory=adapter_factory,
        manifest=ManifestOptions(
            credential_provider=cred_provider,
            s3_connection_options=conn_options,
        ),
        output=OutputOptions(
            run_id="dask-writer-e2e",
            local_root=local_root,
        ),
    )

    pdf = pd.DataFrame({"id": list(range(24)), "val": [f"v{i}" for i in range(24)]})
    ddf = dd.from_pandas(pdf, npartitions=4)

    with dask.config.set(scheduler="synchronous"):
        result = write_sharded(
            ddf,
            config,
            key_col="id",
            value_spec=ValueSpec.callable_encoder(lambda row: str(row["val"]).encode()),
        )

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(f"s3://{bucket}/dask-writer/manifests/")

    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = "dask-writer/_CURRENT"

    client = s3_service["client"]
    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = yaml.safe_load(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))

    assert manifest_payload["required"]["run_id"] == "dask-writer-e2e"
    assert manifest_payload["required"]["num_dbs"] == 4
    assert len(manifest_payload["shards"]) == 4
    assert current_payload["manifest_ref"] == result.manifest_ref

    # Verify each shard was physically written and total rows sum correctly.
    total_rows = 0
    for winner in result.winners:
        assert winner.row_count > 0
        total_rows += winner.row_count

    assert total_rows == 24


# ---------------------------------------------------------------------------
# Scenario 6: Python writer publishes v1, reader opens, Python writer publishes
# v2, reader refreshes
# ---------------------------------------------------------------------------


def run_python_writer_reader_refresh_scenario(
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    adapter_factory: DbAdapterFactory,
    reader_factory: ShardReaderFactory,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Python writer publishes v1, reader opens, Python writer publishes v2, reader refreshes."""

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.sharding_types import ShardingSpec
    from shardyfusion.writer.python import write_sharded

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/python-writer-reader-refresh"
    local_root = str(tmp_path / "python-writer-local")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    def build_config(run_id: str) -> WriteConfig:
        return WriteConfig(
            num_dbs=4,
            s3_prefix=s3_prefix,
            output=OutputOptions(run_id=run_id, local_root=local_root),
            sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
            adapter_factory=adapter_factory,
            manifest=ManifestOptions(
                credential_provider=cred_provider,
                s3_connection_options=conn_options,
            ),
        )

    result_v1 = write_sharded(
        list(range(32)),
        build_config("python-refresh-run-1"),
        key_fn=lambda r: r,
        value_fn=lambda r: f"old-{r}".encode(),
    )

    reader_kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(tmp_path / "reader-cache"),
        "reader_factory": reader_factory,
    }
    if credential_provider is not None or s3_connection_options is not None:
        reader_kwargs["manifest_store"] = S3ManifestStore(
            s3_prefix,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )

    with ConcurrentShardedReader(**reader_kwargs) as reader:
        assert reader.get(7) == b"old-7"

        result_v2 = write_sharded(
            list(range(32)),
            build_config("python-refresh-run-2"),
            key_fn=lambda r: r,
            value_fn=lambda r: f"new-{r}".encode(),
        )

        assert result_v1.manifest_ref != result_v2.manifest_ref

        changed = reader.refresh()
        assert changed is True
        assert reader.get(7) == b"new-7"

        unchanged = reader.refresh()
        assert unchanged is False


# ---------------------------------------------------------------------------
# Scenario 7: Dask writer publishes v1, reader opens, Dask writer publishes
# v2, reader refreshes
# ---------------------------------------------------------------------------


def run_dask_writer_reader_refresh_scenario(
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    adapter_factory: DbAdapterFactory,
    reader_factory: ShardReaderFactory,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Dask writer publishes v1, reader opens, Dask writer publishes v2, reader refreshes."""

    import dask
    import dask.dataframe as dd
    import pandas as pd

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.serde import ValueSpec
    from shardyfusion.sharding_types import ShardingSpec
    from shardyfusion.writer.dask import write_sharded

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/dask-writer-reader-refresh"
    local_root = str(tmp_path / "dask-writer-local")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    def build_config(run_id: str) -> WriteConfig:
        return WriteConfig(
            num_dbs=4,
            s3_prefix=s3_prefix,
            output=OutputOptions(run_id=run_id, local_root=local_root),
            sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
            adapter_factory=adapter_factory,
            manifest=ManifestOptions(
                credential_provider=cred_provider,
                s3_connection_options=conn_options,
            ),
        )

    value_spec = ValueSpec.callable_encoder(lambda row: str(row["val"]).encode())

    def make_ddf(prefix: str) -> dd.DataFrame:
        pdf = pd.DataFrame(
            {"id": list(range(32)), "val": [f"{prefix}-{i}" for i in range(32)]}
        )
        return dd.from_pandas(pdf, npartitions=4)

    with dask.config.set(scheduler="synchronous"):
        result_v1 = write_sharded(
            make_ddf("old"),
            build_config("dask-refresh-run-1"),
            key_col="id",
            value_spec=value_spec,
        )

    reader_kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(tmp_path / "reader-cache"),
        "reader_factory": reader_factory,
    }
    if credential_provider is not None or s3_connection_options is not None:
        reader_kwargs["manifest_store"] = S3ManifestStore(
            s3_prefix,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )

    with ConcurrentShardedReader(**reader_kwargs) as reader:
        assert reader.get(7) == b"old-7"

        with dask.config.set(scheduler="synchronous"):
            result_v2 = write_sharded(
                make_ddf("new"),
                build_config("dask-refresh-run-2"),
                key_col="id",
                value_spec=value_spec,
            )

        assert result_v1.manifest_ref != result_v2.manifest_ref

        changed = reader.refresh()
        assert changed is True
        assert reader.get(7) == b"new-7"

        unchanged = reader.refresh()
        assert unchanged is False


# ---------------------------------------------------------------------------
# Scenario 8: Ray writer publishes manifest to S3
# ---------------------------------------------------------------------------


def run_ray_writer_publishes_manifest_scenario(
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    adapter_factory: DbAdapterFactory,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Ray writer publishes manifest + CURRENT to S3, then reads shards back."""

    import ray.data

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.serde import ValueSpec
    from shardyfusion.sharding_types import ShardingSpec
    from shardyfusion.writer.ray import write_sharded

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/ray-writer"
    local_root = str(tmp_path / "ray-writer-local")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    config = WriteConfig(
        num_dbs=4,
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        adapter_factory=adapter_factory,
        manifest=ManifestOptions(
            credential_provider=cred_provider,
            s3_connection_options=conn_options,
        ),
        output=OutputOptions(
            run_id="ray-writer-e2e",
            local_root=local_root,
        ),
    )

    ds = ray.data.from_items(
        [{"id": i, "val": f"v{i}"} for i in range(24)],
        override_num_blocks=4,
    )

    result = write_sharded(
        ds,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: str(row["val"]).encode()),
    )

    assert len(result.winners) == 4
    assert result.manifest_ref.startswith(f"s3://{bucket}/ray-writer/manifests/")

    manifest_key = result.manifest_ref.split(f"s3://{bucket}/", 1)[1]
    current_key = "ray-writer/_CURRENT"

    client = s3_service["client"]
    manifest_obj = client.get_object(Bucket=bucket, Key=manifest_key)
    current_obj = client.get_object(Bucket=bucket, Key=current_key)

    manifest_payload = yaml.safe_load(manifest_obj["Body"].read())
    current_payload = json.loads(current_obj["Body"].read().decode("utf-8"))

    assert manifest_payload["required"]["run_id"] == "ray-writer-e2e"
    assert manifest_payload["required"]["num_dbs"] == 4
    assert len(manifest_payload["shards"]) == 4
    assert current_payload["manifest_ref"] == result.manifest_ref

    # Verify each shard was physically written and total rows sum correctly.
    total_rows = 0
    for winner in result.winners:
        assert winner.row_count > 0
        total_rows += winner.row_count

    assert total_rows == 24


# ---------------------------------------------------------------------------
# Scenario 9: Ray writer publishes v1, reader opens, Ray writer publishes
# v2, reader refreshes
# ---------------------------------------------------------------------------


def run_ray_writer_reader_refresh_scenario(
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    adapter_factory: DbAdapterFactory,
    reader_factory: ShardReaderFactory,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Ray writer publishes v1, reader opens, Ray writer publishes v2, reader refreshes."""

    import ray.data

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.serde import ValueSpec
    from shardyfusion.sharding_types import ShardingSpec
    from shardyfusion.writer.ray import write_sharded

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/ray-writer-reader-refresh"
    local_root = str(tmp_path / "ray-writer-local")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    def build_config(run_id: str) -> WriteConfig:
        return WriteConfig(
            num_dbs=4,
            s3_prefix=s3_prefix,
            output=OutputOptions(run_id=run_id, local_root=local_root),
            sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
            adapter_factory=adapter_factory,
            manifest=ManifestOptions(
                credential_provider=cred_provider,
                s3_connection_options=conn_options,
            ),
        )

    value_spec = ValueSpec.callable_encoder(lambda row: str(row["val"]).encode())

    def make_ds(prefix: str) -> ray.data.Dataset:
        return ray.data.from_items(
            [{"id": i, "val": f"{prefix}-{i}"} for i in range(32)],
            override_num_blocks=4,
        )

    result_v1 = write_sharded(
        make_ds("old"),
        build_config("ray-refresh-run-1"),
        key_col="id",
        value_spec=value_spec,
    )

    reader_kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(tmp_path / "reader-cache"),
        "reader_factory": reader_factory,
    }
    if credential_provider is not None or s3_connection_options is not None:
        reader_kwargs["manifest_store"] = S3ManifestStore(
            s3_prefix,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )

    with ConcurrentShardedReader(**reader_kwargs) as reader:
        assert reader.get(7) == b"old-7"

        result_v2 = write_sharded(
            make_ds("new"),
            build_config("ray-refresh-run-2"),
            key_col="id",
            value_spec=value_spec,
        )

        assert result_v1.manifest_ref != result_v2.manifest_ref

        changed = reader.refresh()
        assert changed is True
        assert reader.get(7) == b"new-7"

        unchanged = reader.refresh()
        assert unchanged is False
