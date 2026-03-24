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

from shardyfusion.credentials import CredentialProvider
from shardyfusion.manifest_store import S3ManifestStore
from shardyfusion.serde import make_key_encoder
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.slatedb_adapter import DbAdapterFactory
from shardyfusion.type_defs import S3ConnectionOptions, ShardReaderFactory
from tests.helpers.s3_test_scenarios import (
    _default_connection_options,
    _default_credential_provider,
)

if TYPE_CHECKING:
    from shardyfusion.config import WriteConfig
    from shardyfusion.manifest import BuildResult, RequiredShardMeta
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
RoutingContextFn = Callable[["tuple[int, bytes, str]"], "dict[str, object] | None"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_reader_kwargs(
    *,
    s3_prefix: str,
    tmp_path: Path,
    reader_factory: ShardReaderFactory,
    credential_provider: CredentialProvider | None,
    s3_connection_options: S3ConnectionOptions | None,
) -> dict[str, Any]:
    """Build kwargs dict for ShardedReader / ConcurrentShardedReader."""

    kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(tmp_path / "reader-cache"),
        "reader_factory": reader_factory,
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
                groups: dict[
                    tuple[tuple[str, object], ...], list[tuple[int, bytes, str]]
                ] = defaultdict(list)
                for row in SMOKE_DATA:
                    ctx = routing_context_fn(row)
                    ctx_key = tuple(sorted(ctx.items())) if ctx else ()
                    groups[ctx_key].append(row)

                for ctx_key, rows in groups.items():
                    ctx = dict(ctx_key) if ctx_key else None
                    keys = [r[0] for r in rows]
                    got = reader.multi_get(keys, routing_context=ctx)
                    for key, expected_value, _ in rows:
                        assert got[key] == expected_value, (
                            f"{reader_cls_name}: key={key}"
                        )


def _verify_shard_placement(
    result: BuildResult,
    *,
    route_fn: Callable[[int], int],
    reader_factory: ShardReaderFactory,
    local_root: str,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
) -> None:
    """Verify each SMOKE_DATA key landed in the shard predicted by route_fn.

    Opens raw shard readers per shard via *reader_factory* (bypassing
    ShardedReader), providing independent diagnostics for shard
    mis-routing and duplicate keys.
    """

    encode_key = make_key_encoder(key_encoding)

    winner_by_db_id: dict[int, RequiredShardMeta] = {w.db_id: w for w in result.winners}
    readers: dict[int, Any] = {}
    try:
        for db_id, winner in winner_by_db_id.items():
            assert winner.db_url is not None, f"shard {db_id} has no db_url"
            readers[db_id] = reader_factory(
                db_url=winner.db_url,
                local_dir=Path(local_root) / f"verify-{db_id}",
                checkpoint_id=winner.checkpoint_id,
            )

        for key, _value, _group in SMOKE_DATA:
            expected_db_id = route_fn(key)
            key_bytes = encode_key(key)

            assert expected_db_id in readers, (
                f"key {key} routes to shard {expected_db_id} "
                f"but no winner exists for that shard"
            )
            got = readers[expected_db_id].get(key_bytes)
            assert got is not None, (
                f"key {key} not found in expected shard {expected_db_id}"
            )

            # Verify key is NOT in any other shard (catches duplicates).
            for other_db_id, other_reader in readers.items():
                if other_db_id == expected_db_id:
                    continue
                dup = other_reader.get(key_bytes)
                assert dup is None, (
                    f"key {key} duplicated: found in shard {other_db_id} "
                    f"(expected only in shard {expected_db_id})"
                )
    finally:
        for reader in readers.values():
            reader.close()


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
    adapter_factory: DbAdapterFactory,
    reader_factory: ShardReaderFactory,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Write SMOKE_DATA with HASH sharding, read back with both sync readers.

    Args:
        num_dbs: Explicit shard count (set to 0 when using max_keys_per_shard).
        max_keys_per_shard: If set, num_dbs is computed as ceil(10 / max_keys_per_shard).
        expected_num_shards: Expected number of non-empty shards in the result.
        adapter_factory: Backend-specific factory for opening shard writers.
        reader_factory: Backend-specific factory for opening shard readers.
    """

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/smoke-hash-{tmp_path.name}"
    local_root = str(tmp_path / "writer-local")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    config = WriteConfig(
        num_dbs=num_dbs,
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.HASH,
            max_keys_per_shard=max_keys_per_shard,
        ),
        adapter_factory=adapter_factory,
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

    # ---- Shard placement (independent of reader) ----
    from shardyfusion.routing import xxh3_db_id

    effective_num_dbs = max(w.db_id for w in result.winners) + 1
    _verify_shard_placement(
        result,
        route_fn=lambda key: xxh3_db_id(key, effective_num_dbs),
        reader_factory=reader_factory,
        local_root=local_root,
    )

    # ---- Read ----
    reader_kwargs = _build_reader_kwargs(
        s3_prefix=s3_prefix,
        tmp_path=tmp_path,
        reader_factory=reader_factory,
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
    adapter_factory: DbAdapterFactory,
    reader_factory: ShardReaderFactory,
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
        adapter_factory: Backend-specific factory for opening shard writers.
        reader_factory: Backend-specific factory for opening shard readers.
        routing_context_fn: If the CEL expression uses columns other than "key",
            provide a function ``(row) -> {"col": value}`` so the reader knows
            how to route each key.
    """

    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy

    bucket = s3_service["bucket"]
    # Include a short hash of the expression in the prefix to avoid collisions.
    safe_suffix = (
        cel_expr.replace(" ", "")
        .replace("%", "mod")
        .replace("(", "")
        .replace(")", "")[:20]
    )
    s3_prefix = f"s3://{bucket}/smoke-cel-{safe_suffix}-{tmp_path.name}"
    local_root = str(tmp_path / "writer-local")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)

    config = WriteConfig(
        num_dbs=None,  # CEL always discovers from data
        s3_prefix=s3_prefix,
        sharding=ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr=cel_expr,
            cel_columns=cel_columns,
            boundaries=boundaries,
        ),
        adapter_factory=adapter_factory,
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

    # ---- Shard placement (independent of reader) ----
    from shardyfusion.cel import compile_cel, route_cel

    compiled = compile_cel(cel_expr, cel_columns)
    # Widen boundaries type for route_cel (list invariance).
    cel_boundaries: Any = boundaries
    if routing_context_fn is not None:
        key_to_ctx: dict[int, dict[str, Any]] = {}
        for row in SMOKE_DATA:
            ctx = routing_context_fn(row)
            assert ctx is not None, f"routing_context_fn returned None for key {row[0]}"
            key_to_ctx[row[0]] = ctx
        _verify_shard_placement(
            result,
            route_fn=lambda key: route_cel(compiled, key_to_ctx[key], cel_boundaries),
            reader_factory=reader_factory,
            local_root=local_root,
        )
    else:
        _verify_shard_placement(
            result,
            route_fn=lambda key: route_cel(compiled, {"key": key}, cel_boundaries),
            reader_factory=reader_factory,
            local_root=local_root,
        )

    # ---- Read ----
    reader_kwargs = _build_reader_kwargs(
        s3_prefix=s3_prefix,
        tmp_path=tmp_path,
        reader_factory=reader_factory,
        credential_provider=credential_provider,
        s3_connection_options=s3_connection_options,
    )
    _verify_reads(reader_kwargs, routing_context_fn=routing_context_fn)
