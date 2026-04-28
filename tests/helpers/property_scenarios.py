"""Property-based KV E2E scenarios shared by all writer frameworks."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from hypothesis import strategies as st

from shardyfusion.credentials import CredentialProvider
from shardyfusion.serde import make_key_encoder
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.slatedb_adapter import DbAdapterFactory
from shardyfusion.type_defs import S3ConnectionOptions, ShardReaderFactory
from tests.helpers.s3_test_scenarios import (
    _default_connection_options,
    _default_credential_provider,
    _make_s3_manifest_store,
)

if TYPE_CHECKING:
    from shardyfusion.config import WriteConfig
    from shardyfusion.manifest import BuildResult, ParsedManifest, RequiredShardMeta
    from tests.conftest import LocalS3Service


class PropertyCaseKind(str, Enum):
    HASH = "hash"
    HASH_MAX_KEYS = "hash_max_keys"
    CEL_CONTEXT = "cel_context"
    CEL_CATEGORICAL = "cel_categorical"


PROPERTY_CASE_KINDS: tuple[PropertyCaseKind, ...] = (
    PropertyCaseKind.HASH,
    PropertyCaseKind.HASH_MAX_KEYS,
    PropertyCaseKind.CEL_CONTEXT,
    PropertyCaseKind.CEL_CATEGORICAL,
)

GROUP_VALUES = ("alpha", "beta", "gamma")


@dataclass(frozen=True, slots=True)
class PropertyRow:
    key: int
    value: bytes
    tenant_id: int
    group: str


@dataclass(frozen=True, slots=True)
class PropertyInput:
    rows: list[PropertyRow]
    num_dbs: int
    max_keys_per_shard: int


PropertyWriteFn = Callable[[Iterable[PropertyRow], "WriteConfig"], "BuildResult"]


def property_input_strategy() -> st.SearchStrategy[PropertyInput]:
    rows = st.lists(
        st.builds(
            PropertyRow,
            key=st.integers(min_value=0, max_value=(1 << 32) - 1),
            value=st.binary(min_size=0, max_size=32),
            tenant_id=st.integers(min_value=0, max_value=8),
            group=st.sampled_from(GROUP_VALUES),
        ),
        min_size=1,
        max_size=12,
        unique_by=lambda row: row.key,
    )
    return st.builds(
        PropertyInput,
        rows=rows,
        num_dbs=st.integers(min_value=1, max_value=6),
        max_keys_per_shard=st.integers(min_value=1, max_value=6),
    )


def row_dicts(rows: Iterable[PropertyRow]) -> list[dict[str, object]]:
    return [
        {
            "key": row.key,
            "value": row.value,
            "tenant_id": row.tenant_id,
            "group": row.group,
        }
        for row in rows
    ]


def _build_config(
    *,
    kind: PropertyCaseKind,
    case: PropertyInput,
    s3_prefix: str,
    local_root: str,
    adapter_factory: DbAdapterFactory,
    credential_provider: CredentialProvider,
    s3_connection_options: S3ConnectionOptions,
) -> WriteConfig:
    from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
    from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy

    sharding: ShardingSpec
    num_dbs: int | None
    if kind == PropertyCaseKind.HASH:
        num_dbs = case.num_dbs
        sharding = ShardingSpec(strategy=ShardingStrategy.HASH)
    elif kind == PropertyCaseKind.HASH_MAX_KEYS:
        num_dbs = None
        sharding = ShardingSpec(
            strategy=ShardingStrategy.HASH,
            max_keys_per_shard=case.max_keys_per_shard,
        )
    elif kind == PropertyCaseKind.CEL_CONTEXT:
        num_dbs = None
        sharding = ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="tenant_id % 3",
            cel_columns={"tenant_id": "int"},
        )
    else:
        num_dbs = None
        sharding = ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="group",
            cel_columns={"group": "string"},
            routing_values=list(GROUP_VALUES),
        )

    return WriteConfig(
        num_dbs=num_dbs,
        s3_prefix=s3_prefix,
        sharding=sharding,
        adapter_factory=adapter_factory,
        manifest=ManifestOptions(
            credential_provider=credential_provider,
            s3_connection_options=s3_connection_options,
        ),
        output=OutputOptions(
            run_id=f"property-{kind.value}",
            local_root=local_root,
        ),
    )


def _route_fn(kind: PropertyCaseKind, case: PropertyInput) -> Callable[[PropertyRow], int]:
    from shardyfusion.cel import compile_cel, route_cel
    from shardyfusion.routing import xxh3_db_id

    if kind == PropertyCaseKind.HASH:
        return lambda row: xxh3_db_id(row.key, case.num_dbs)
    if kind == PropertyCaseKind.HASH_MAX_KEYS:
        num_dbs = ceil(len(case.rows) / case.max_keys_per_shard)
        return lambda row: xxh3_db_id(row.key, num_dbs)
    if kind == PropertyCaseKind.CEL_CONTEXT:
        compiled = compile_cel("tenant_id % 3", {"tenant_id": "int"})
        return lambda row: route_cel(compiled, {"tenant_id": row.tenant_id})

    compiled = compile_cel("group", {"group": "string"})
    return lambda row: route_cel(
        compiled,
        {"group": row.group},
        list(GROUP_VALUES),
    )


def _routing_context(
    kind: PropertyCaseKind, row: PropertyRow
) -> dict[str, object] | None:
    if kind == PropertyCaseKind.CEL_CONTEXT:
        return {"tenant_id": row.tenant_id}
    if kind == PropertyCaseKind.CEL_CATEGORICAL:
        return {"group": row.group}
    return None


def _expected_num_dbs(kind: PropertyCaseKind, case: PropertyInput) -> int:
    if kind == PropertyCaseKind.HASH:
        return case.num_dbs
    if kind == PropertyCaseKind.HASH_MAX_KEYS:
        return ceil(len(case.rows) / case.max_keys_per_shard)
    route = _route_fn(kind, case)
    return max(route(row) for row in case.rows) + 1


def _verify_direct_shards(
    *,
    case: PropertyInput,
    result: BuildResult,
    manifest: ParsedManifest,
    route: Callable[[PropertyRow], int],
    reader_factory: ShardReaderFactory,
    local_root: str,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
) -> None:
    encode_key = make_key_encoder(key_encoding)
    expected_counts = Counter(route(row) for row in case.rows)
    winner_by_db_id: dict[int, RequiredShardMeta] = {
        winner.db_id: winner for winner in result.winners
    }

    assert set(winner_by_db_id) == set(expected_counts)
    for db_id, expected_count in expected_counts.items():
        assert winner_by_db_id[db_id].row_count == expected_count

    readers: dict[int, Any] = {}
    try:
        for db_id, winner in winner_by_db_id.items():
            assert winner.db_url is not None, f"shard {db_id} has no db_url"
            readers[db_id] = reader_factory(
                db_url=winner.db_url,
                local_dir=Path(local_root) / f"property-verify-{db_id}",
                checkpoint_id=winner.checkpoint_id,
                manifest=manifest,
            )

        for row in case.rows:
            expected_db_id = route(row)
            key_bytes = encode_key(row.key)
            assert readers[expected_db_id].get(key_bytes) == row.value
            for other_db_id, other_reader in readers.items():
                if other_db_id == expected_db_id:
                    continue
                assert other_reader.get(key_bytes) is None
    finally:
        for reader in readers.values():
            reader.close()


def _verify_readers(
    *,
    kind: PropertyCaseKind,
    case: PropertyInput,
    route: Callable[[PropertyRow], int],
    s3_prefix: str,
    tmp_path: Path,
    reader_factory: ShardReaderFactory,
    credential_provider: CredentialProvider,
    s3_connection_options: S3ConnectionOptions,
) -> None:
    from shardyfusion.reader import ConcurrentShardedReader
    from shardyfusion.reader.reader import ShardedReader

    manifest_store = _make_s3_manifest_store(
        s3_prefix,
        credential_provider=credential_provider,
        s3_connection_options=s3_connection_options,
    )
    reader_kwargs: dict[str, Any] = {
        "s3_prefix": s3_prefix,
        "local_root": str(tmp_path / f"property-reader-cache-{uuid4().hex[:8]}"),
        "reader_factory": reader_factory,
        "manifest_store": manifest_store,
    }

    for reader_cls in (ShardedReader, ConcurrentShardedReader):
        with reader_cls(**reader_kwargs) as reader:
            for row in case.rows:
                ctx = _routing_context(kind, row)
                expected_db_id = route(row)
                assert reader.route_key(row.key, routing_context=ctx) == expected_db_id
                assert (
                    reader.shard_for_key(row.key, routing_context=ctx).db_id
                    == expected_db_id
                )
                assert reader.get(row.key, routing_context=ctx) == row.value

            groups: dict[
                tuple[tuple[str, object], ...], list[PropertyRow]
            ] = defaultdict(list)
            for row in case.rows:
                ctx = _routing_context(kind, row)
                ctx_key = tuple(sorted(ctx.items())) if ctx else ()
                groups[ctx_key].append(row)

            for ctx_key, rows in groups.items():
                ctx = dict(ctx_key) if ctx_key else None
                got = reader.multi_get([row.key for row in rows], routing_context=ctx)
                assert got == {row.key: row.value for row in rows}


def run_property_kv_e2e_scenario(
    write_fn: PropertyWriteFn,
    s3_service: LocalS3Service,
    tmp_path: Path,
    *,
    kind: PropertyCaseKind,
    case: PropertyInput,
    adapter_factory: DbAdapterFactory,
    reader_factory: ShardReaderFactory,
    credential_provider: CredentialProvider | None = None,
    s3_connection_options: S3ConnectionOptions | None = None,
) -> None:
    """Write generated KV rows, then validate readers and physical shard placement."""

    bucket = s3_service["bucket"]
    s3_prefix = f"s3://{bucket}/property-{kind.value}-{uuid4().hex[:12]}"
    local_root = str(tmp_path / f"property-writer-local-{uuid4().hex[:8]}")

    cred_provider = _default_credential_provider(s3_service, credential_provider)
    conn_options = _default_connection_options(s3_service, s3_connection_options)
    config = _build_config(
        kind=kind,
        case=case,
        s3_prefix=s3_prefix,
        local_root=local_root,
        adapter_factory=adapter_factory,
        credential_provider=cred_provider,
        s3_connection_options=conn_options,
    )

    result = write_fn(case.rows, config)
    manifest_store = _make_s3_manifest_store(
        s3_prefix,
        credential_provider=cred_provider,
        s3_connection_options=conn_options,
    )
    manifest = manifest_store.load_manifest(result.manifest_ref)
    route = _route_fn(kind, case)

    assert manifest.required_build.num_dbs == _expected_num_dbs(kind, case)
    assert sum(winner.row_count for winner in result.winners) == len(case.rows)
    assert sum(shard.row_count for shard in manifest.shards) == len(case.rows)

    _verify_direct_shards(
        case=case,
        result=result,
        manifest=manifest,
        route=route,
        reader_factory=reader_factory,
        local_root=local_root,
        key_encoding=manifest.required_build.key_encoding,
    )
    _verify_readers(
        kind=kind,
        case=case,
        route=route,
        s3_prefix=s3_prefix,
        tmp_path=tmp_path,
        reader_factory=reader_factory,
        credential_provider=cred_provider,
        s3_connection_options=conn_options,
    )
