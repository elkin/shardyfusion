"""Smoke E2E scenarios: write simple data, read back with all sync readers.

The input and expected output are defined here so reviewers only need to
check this one file to verify the data flow.

Scenarios cover HASH sharding (with num_dbs and max_keys_per_shard),
and CEL sharding (key modulo, shard_hash, and routing-context column).
"""

from __future__ import annotations

from collections import defaultdict
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

# ---------------------------------------------------------------------------
# Input data — 10 rows, 3 columns: (key: int, value: bytes, group: str)
#
# The "group" column has 2 distinct values ("a" for even keys, "b" for odd).
# HASH tests ignore it; CEL routing-context tests use it.
# ---------------------------------------------------------------------------

SMOKE_DATA: list[tuple[int, bytes, str]] = [
    (0, b"zero", "a"),
    (1, b"one", "b"),
    (2, b"two", "a"),
    (3, b"three", "b"),
    (4, b"four", "a"),
    (5, b"five", "b"),
    (6, b"six", "a"),
    (7, b"seven", "b"),
    (8, b"eight", "a"),
    (9, b"nine", "b"),
]

# Expected output: every key must round-trip to its value.
EXPECTED: dict[int, bytes] = {k: v for k, v, _ in SMOKE_DATA}

# Type alias for engine-specific write callbacks.
WriteFn = Callable[["list[tuple[int, bytes, str]]", "WriteConfig"], "BuildResult"]

# Type alias for a function that returns the routing_context for a given row.
RoutingContextFn = Callable[["tuple[int, bytes, str]", dict[str, object] | None]]


# ---------------------------------------------------------------------------
# S3 credential/connection helpers (same as s3_test_scenarios.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_reader_kwargs(
    *,
    s3_prefix: str,
    tmp_path: Path,
    local_root: str,
    object_store_root: str,
    credential_provider: CredentialProvider | None,
    s3_connection_options: S3ConnectionOptions | None,
) -> dict[str, Any]:
    """Build kwargs dict for ShardedReader / ConcurrentShardedReader."""

    from shardyfusion.testing import (
        map_s3_db_url_to_file_url,
        writer_local_dir_for_db_url,
    )

    def open_real_reader(
        *, db_url: str, local_dir: Path, checkpoint_id: str | None
    ) -> slatedb.SlateDBReader:
        return slatedb.SlateDBReader(
            writer_local_dir_for_db_url(db_url, local_root),
            url=map_s3_db_url_to_file_url(db_url, object_store_root),
            checkpoint_id=checkpoint_id,
        )

    kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(tmp_path / "reader-cache"),
        "reader_factory": open_real_reader,
    }
    if credential_provider is not None or s3_connection_options is not None:
        kwargs["manifest_store"] = S3ManifestStore(
            s3_prefix,
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        )
    return kwargs


def _verify_reads(
    reader_kwargs: dict[str, Any],
    *,
    routing_context_fn: RoutingContextFn | None = None,
) -> None:
    """Read back all SMOKE_DATA keys with ShardedReader & ConcurrentShardedReader."""

    from shardyfusion.reader import ConcurrentShardedReader
    from shardyfusion.reader.reader import ShardedReader

    for reader_cls_name, reader_cls in [
        ("ShardedReader", ShardedReader),
        ("ConcurrentShardedReader", ConcurrentShardedReader),
    ]:
        with reader_cls(**reader_kwargs) as reader:
            if routing_context_fn is None:
                # No routing context — one multi_get for all keys.
                keys = [k for k, _, _ in SMOKE_DATA]
                got = reader.multi_get(keys)
                for key, expected_value, _ in SMOKE_DATA:
                    assert got[key] == expected_value, f"{reader_cls_name}: key={key}"
            else:
                # Group keys by routing context and call multi_get per group.
                groups: dict[tuple[tuple[str, object], ...], list[tuple[int, bytes, str]]] = (
                    defaultdict(list)
                )
                for row in SMOKE_DATA:
                    ctx = routing_context_fn(row)
                    ctx_key = tuple(sorted(ctx.items())) if ctx else ()
                    groups[ctx_key].append(row)

                for ctx_key, rows in groups.items():
                    ctx = dict(ctx_key) if ctx_key else None
                    keys = [r[0] for r in rows]
                    got = reader.multi_get(keys, routing_context=ctx)
                    for key, expected_value, _ in rows:
                        assert got[key] == expected_value, f"{reader_cls_name}: key={key}"


# ---------------------------------------------------------------------------
# Scenario: HASH sharding (configurable num_dbs / max_keys_per_shard)
# ---------------------------------------------------------------------------


def run_smoke_write_then_read_scenario(
    write_fn: WriteFn,
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    num_dbs: int = 3,
    max_keys_per_shard: int | None = None,
    expected_num_shards: int | None = None,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Write SMOKE_DATA with HASH sharding, read back with both sync readers.

    Args:
        num_dbs: Explicit shard count (set to 0 when using max_keys_per_shard).
        max_keys_per_shard: If set, num_dbs is computed as ceil(10 / max_keys_per_shard).
        expected_num_shards: Expected number of non-empty shards in the result.
            Defaults to *num_dbs* when num_dbs > 0, or computed from max_keys_per_shard.
    """

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
    from shardyfusion.testing import real_file_adapter_factory

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/smoke-hash-{tmp_path.name}"
    local_root = str(tmp_path / "writer-local")
    object_store_root = str(tmp_path / "object-store")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    config = WriteConfig(
        num_dbs=num_dbs,
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.HASH,
            max_keys_per_shard=max_keys_per_shard,
        ),
        adapter_factory=real_file_adapter_factory(object_store_root),
        manifest=ManifestOptions(
            credential_provider=cred_provider,
            s3_connection_options=conn_options,
        ),
        output=OutputOptions(run_id="smoke-hash", local_root=local_root),
    )

    # ---- Write ----
    result = write_fn(SMOKE_DATA, config)

    if expected_num_shards is not None:
        assert len(result.winners) == expected_num_shards
    total_rows = sum(w.row_count for w in result.winners)
    assert total_rows == len(SMOKE_DATA)

    # ---- Read ----
    reader_kwargs = _build_reader_kwargs(
        s3_prefix=s3_prefix,
        tmp_path=tmp_path,
        local_root=local_root,
        object_store_root=object_store_root,
        credential_provider=credential_provider,
        s3_connection_options=s3_connection_options,
    )
    _verify_reads(reader_kwargs)


# ---------------------------------------------------------------------------
# Scenario: CEL sharding (key-based expression)
# ---------------------------------------------------------------------------


def run_smoke_cel_scenario(
    write_fn: WriteFn,
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    cel_expr: str,
    cel_columns: dict[str, str],
    boundaries: list[int | str] | None = None,
    expected_num_shards: int,
    routing_context_fn: RoutingContextFn | None = None,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Write SMOKE_DATA with CEL sharding, read back with both sync readers.

    Args:
        cel_expr: The CEL expression (e.g. ``"key % 3"``, ``"shard_hash(key) % 3u"``).
        cel_columns: Column name → CEL type mapping (e.g. ``{"key": "int"}``).
        boundaries: Optional boundary list for bisect_right routing.
        expected_num_shards: How many non-empty shards the expression should produce.
        routing_context_fn: If the CEL expression uses columns other than "key",
            provide a function ``(row) -> {"col": value}`` so the reader knows
            how to route each key.
    """

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
    from shardyfusion.testing import real_file_adapter_factory

    bucket = s3_service["bucket"]
    # Include a short hash of the expression in the prefix to avoid collisions.
    safe_suffix = cel_expr.replace(" ", "").replace("%", "mod").replace("(", "").replace(")", "")[:20]
    s3_prefix = f"s3://{bucket}/smoke-cel-{safe_suffix}-{tmp_path.name}"
    local_root = str(tmp_path / "writer-local")
    object_store_root = str(tmp_path / "object-store")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    config = WriteConfig(
        num_dbs=0,  # CEL always discovers from data
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr=cel_expr,
            cel_columns=cel_columns,
            boundaries=boundaries,
        ),
        adapter_factory=real_file_adapter_factory(object_store_root),
        manifest=ManifestOptions(
            credential_provider=cred_provider,
            s3_connection_options=conn_options,
        ),
        output=OutputOptions(run_id="smoke-cel", local_root=local_root),
    )

    # ---- Write ----
    result = write_fn(SMOKE_DATA, config)

    assert len(result.winners) == expected_num_shards
    total_rows = sum(w.row_count for w in result.winners)
    assert total_rows == len(SMOKE_DATA)

    # ---- Read ----
    reader_kwargs = _build_reader_kwargs(
        s3_prefix=s3_prefix,
        tmp_path=tmp_path,
        local_root=local_root,
        object_store_root=object_store_root,
        credential_provider=credential_provider,
        s3_connection_options=s3_connection_options,
    )
    _verify_reads(reader_kwargs, routing_context_fn=routing_context_fn)
