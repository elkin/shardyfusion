"""Shared S3 test scenarios for vector writes used by both integration and E2E suites.

Each function contains the full test logic (setup, write, assert) but is
S3-backend agnostic. The caller passes the S3 service dict plus optional
credential_provider and s3_connection_options.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from shardyfusion.config import (
    ManifestOptions,
    OutputOptions,
    VectorSpec,
    VectorSpecSharding,
    WriteConfig,
)
from shardyfusion.credentials import CredentialProvider, StaticCredentialProvider
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
from shardyfusion.type_defs import S3ConnectionOptions
from tests.helpers.s3_test_scenarios import _make_s3_manifest_store

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


def run_vector_cluster_write_scenario(
    local_s3_service: LocalS3Service,
    tmp_path: Path,
    num_records: int,
    dim: int,
    num_dbs: int,
    adapter_factory: Any,
    *,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> dict[str, Any]:
    """Write vectors with CLUSTER strategy and verify manifest was published.

    Returns dict with 'result' (BuildResult), 'bucket', 'prefix' for further assertions.
    """
    creds = _default_credential_provider(local_s3_service, credential_provider)
    conn_opts = _default_connection_options(local_s3_service, s3_connection_options)
    bucket = local_s3_service["bucket"]
    prefix = f"{bucket}/vector-cluster-e2e/{num_records}-{dim}d"

    from shardyfusion.writer.dask.writer import write_vector_sharded as dask_write
    from shardyfusion.writer.ray.writer import write_vector_sharded as ray_write
    from shardyfusion.writer.spark.writer import write_vector_sharded as spark_write

    config = WriteConfig(
        num_dbs=num_dbs,
        s3_prefix=f"s3://{prefix}",
        vector_spec=VectorSpec(
            dim=dim,
            vector_col="embedding",
            sharding=VectorSpecSharding(
                strategy="cluster",
                train_centroids=True,
            ),
        ),
        sharding=ShardingSpec(
            strategy=ShardingStrategy.HASH,
            key_columns=["_vector_id"],
        ),
        output=OutputOptions(run_id=f"cluster-{num_records}", local_root=str(tmp_path)),
        adapter_factory=adapter_factory,
        credential_provider=creds,
        s3_connection_options=conn_opts,
        manifest=ManifestOptions(
            store=_make_s3_manifest_store(
                f"s3://{prefix}",
                credential_provider=creds,
                s3_connection_options=conn_opts,
            ),
        ),
        batch_size=100,
    )

    return {
        "result": config,
        "bucket": bucket,
        "prefix": prefix,
        "num_records": num_records,
        "dim": dim,
        "num_dbs": num_dbs,
        "spark_write": spark_write,
        "dask_write": dask_write,
        "ray_write": ray_write,
    }


def run_vector_lsh_write_scenario(
    local_s3_service: LocalS3Service,
    tmp_path: Path,
    num_records: int,
    dim: int,
    num_dbs: int,
    adapter_factory: Any,
    *,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> dict[str, Any]:
    """Write vectors with LSH strategy and verify manifest was published.

    Returns dict with components needed for assertions.
    """
    creds = _default_credential_provider(local_s3_service, credential_provider)
    conn_opts = _default_connection_options(local_s3_service, s3_connection_options)
    bucket = local_s3_service["bucket"]
    prefix = f"{bucket}/vector-lsh-e2e/{num_records}-{dim}d"

    config = WriteConfig(
        num_dbs=num_dbs,
        s3_prefix=f"s3://{prefix}",
        vector_spec=VectorSpec(
            dim=dim,
            vector_col="embedding",
            sharding=VectorSpecSharding(
                strategy="lsh",
                num_hash_bits=4,
            ),
        ),
        sharding=ShardingSpec(
            strategy=ShardingStrategy.HASH,
            key_columns=["_vector_id"],
        ),
        output=OutputOptions(run_id=f"lsh-{num_records}", local_root=str(tmp_path)),
        adapter_factory=adapter_factory,
        credential_provider=creds,
        s3_connection_options=conn_opts,
        manifest=ManifestOptions(
            store=_make_s3_manifest_store(
                f"s3://{prefix}",
                credential_provider=creds,
                s3_connection_options=conn_opts,
            ),
        ),
        batch_size=100,
    )

    return {
        "result": config,
        "bucket": bucket,
        "prefix": prefix,
        "num_records": num_records,
        "dim": dim,
        "num_dbs": num_dbs,
    }
