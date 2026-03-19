"""Smoke E2E scenario: write simple 2-column data, read back with all sync readers.

The input and expected output are defined here so reviewers only need to
check this one file to verify the data flow.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import slatedb

from shardyfusion.credentials import CredentialProvider, StaticCredentialProvider
from shardyfusion.manifest_store import S3ManifestStore
from shardyfusion.type_defs import S3ConnectionOptions

if TYPE_CHECKING:
    from shardyfusion.config import WriteConfig
    from shardyfusion.manifest import BuildResult

    from tests.conftest import LocalS3Service

# ---- Input ----
# 10 rows, 2 columns: (key: int, value: bytes)
SMOKE_DATA: list[tuple[int, bytes]] = [
    (0, b"zero"),
    (1, b"one"),
    (2, b"two"),
    (3, b"three"),
    (4, b"four"),
    (5, b"five"),
    (6, b"six"),
    (7, b"seven"),
    (8, b"eight"),
    (9, b"nine"),
]

# ---- Expected output ----
# After writing, every key must round-trip: reader.multi_get(keys) == {k: v for k, v in SMOKE_DATA}
EXPECTED: dict[int, bytes] = dict(SMOKE_DATA)

# Type alias for the engine-specific write callback.
WriteFn = Callable[["list[tuple[int, bytes]]", "WriteConfig"], "BuildResult"]


def _default_credential_provider(
    s3_service: LocalS3Service,
    credential_provider: CredentialProvider | None,
) -> CredentialProvider:
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
    if s3_connection_options is not None:
        return s3_connection_options
    return S3ConnectionOptions(
        endpoint_url=s3_service["endpoint_url"],
        region_name=s3_service["region_name"],
    )


def run_smoke_write_then_read_scenario(
    write_fn: WriteFn,
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Write SMOKE_DATA via *write_fn*, read back with ShardedReader & ConcurrentShardedReader."""

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.reader import ConcurrentShardedReader
    from shardyfusion.reader.reader import ShardedReader
    from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
    from shardyfusion.testing import (
        map_s3_db_url_to_file_url,
        real_file_adapter_factory,
        writer_local_dir_for_db_url,
    )

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/smoke-{tmp_path.name}"
    local_root = str(tmp_path / "writer-local")
    object_store_root = str(tmp_path / "object-store")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    config = WriteConfig(
        num_dbs=3,
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(strategy=ShardingStrategy.HASH),
        adapter_factory=real_file_adapter_factory(object_store_root),
        manifest=ManifestOptions(
            credential_provider=cred_provider,
            s3_connection_options=conn_options,
        ),
        output=OutputOptions(run_id="smoke-run", local_root=local_root),
    )

    # ---- Write ----
    result = write_fn(SMOKE_DATA, config)

    assert len(result.winners) == 3
    total_rows = sum(w.row_count for w in result.winners)
    assert total_rows == len(SMOKE_DATA)

    # ---- Read helpers ----
    def open_real_reader(
        *, db_url: str, local_dir: Path, checkpoint_id: str | None
    ) -> slatedb.SlateDBReader:
        return slatedb.SlateDBReader(
            writer_local_dir_for_db_url(db_url, local_root),
            url=map_s3_db_url_to_file_url(db_url, object_store_root),
            checkpoint_id=checkpoint_id,
        )

    reader_kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(tmp_path / "reader-cache"),
        "reader_factory": open_real_reader,
    }
    if credential_provider is not None or s3_connection_options is not None:
        reader_kwargs["manifest_store"] = S3ManifestStore(
            s3_prefix,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )

    keys = [k for k, _ in SMOKE_DATA]

    # ---- Read with ShardedReader ----
    with ShardedReader(**reader_kwargs) as reader:
        got = reader.multi_get(keys)
        for key, expected_value in SMOKE_DATA:
            assert got[key] == expected_value, f"ShardedReader: key={key}"

    # ---- Read with ConcurrentShardedReader ----
    with ConcurrentShardedReader(**reader_kwargs) as reader:
        got = reader.multi_get(keys)
        for key, expected_value in SMOKE_DATA:
            assert got[key] == expected_value, f"ConcurrentShardedReader: key={key}"
