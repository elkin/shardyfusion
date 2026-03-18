"""End-to-end tests for CEL sharding via the Python writer.

Tests both **unified** mode (CEL output = DB key) and **split** mode
(CEL routes, separate key_fn stores).  Exercises the full pipeline:
write → manifest → router → read-back verification.

Requires the ``cel`` extra (``cel-expr-python``).
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import pytest

cel_expr_python = pytest.importorskip("cel_expr_python")  # noqa: F841

from shardyfusion._writer_core import route_key
from shardyfusion.cel import compile_cel, route_cel
from shardyfusion.config import ManifestOptions, OutputOptions, WriteConfig
from shardyfusion.manifest import BuildResult
from shardyfusion.manifest_store import InMemoryManifestStore
from shardyfusion.routing import SnapshotRouter
from shardyfusion.serde import make_key_encoder
from shardyfusion.sharding_types import KeyEncoding, ShardingSpec, ShardingStrategy
from shardyfusion.testing import (
    file_backed_adapter_factory,
    file_backed_load_db,
)
from shardyfusion.writer.python import write_sharded

pytestmark = pytest.mark.cel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cel_config(
    tmp_path: Path,
    *,
    cel_expr: str,
    cel_columns: dict[str, str],
    boundaries: list[int | float | str],
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
    batch_size: int = 50_000,
) -> tuple[WriteConfig, str]:
    """Build a CEL WriteConfig with file-backed adapter for data verification."""
    root_dir = str(tmp_path / "file_backed")
    store = InMemoryManifestStore()
    config = WriteConfig(
        num_dbs=0,
        s3_prefix="s3://bucket/prefix",
        key_encoding=key_encoding,
        batch_size=batch_size,
        adapter_factory=file_backed_adapter_factory(root_dir),
        sharding=ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr=cel_expr,
            cel_columns=cel_columns,
            boundaries=boundaries,
        ),
        output=OutputOptions(run_id="test-cel"),
        manifest=ManifestOptions(store=store),
    )
    return config, root_dir


# ---------------------------------------------------------------------------
# CEL Unified Mode Tests
# ---------------------------------------------------------------------------


class TestCelUnifiedMode:
    """CEL unified mode: CEL output IS the routing key, used for both
    shard assignment and DB key encoding."""

    def test_write_and_read_modulo_routing(self, tmp_path: Path) -> None:
        """key % 4 routes 100 records into 4 shards correctly."""
        # Precompute boundaries for key%4 → values 0,1,2,3 → boundaries [1,2,3]
        config, root_dir = _make_cel_config(
            tmp_path,
            cel_expr="key % 4",
            cel_columns={"key": "int"},
            boundaries=[1, 2, 3],
        )

        records = list(range(100))
        result = write_sharded(
            records,
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: f"v{r}".encode(),
        )

        assert isinstance(result, BuildResult)
        assert result.stats.rows_written == 100
        assert len(result.winners) == 4
        assert sorted(w.db_id for w in result.winners) == [0, 1, 2, 3]

        # Verify each record is in the correct shard
        compiled = compile_cel("key % 4", {"key": "int"})
        for winner in result.winners:
            shard_data = file_backed_load_db(root_dir, winner.db_url)
            for key_bytes in shard_data:
                key_int = int.from_bytes(key_bytes, byteorder="big", signed=False)
                routing_key = compiled.evaluate({"key": key_int})
                expected_db_id = bisect_right([1, 2, 3], routing_key)
                assert expected_db_id == winner.db_id, (
                    f"key={key_int}, routing_key={routing_key}, "
                    f"expected_db_id={expected_db_id}, actual={winner.db_id}"
                )

    def test_manifest_contains_cel_metadata(self, tmp_path: Path) -> None:
        """Published manifest includes cel_expr and cel_columns."""
        store = InMemoryManifestStore()
        config, _ = _make_cel_config(
            tmp_path,
            cel_expr="key % 2",
            cel_columns={"key": "int"},
            boundaries=[1],
        )
        config.manifest.store = store

        result = write_sharded(
            list(range(20)),
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: b"v",
        )

        # Load the manifest and verify CEL metadata
        manifest = store.load_manifest(result.manifest_ref)
        sharding = manifest.required_build.sharding
        assert sharding.strategy == ShardingStrategy.CEL
        assert sharding.cel_expr == "key % 2"
        assert sharding.cel_columns == {"key": "int"}
        assert sharding.boundaries == [1]

    def test_router_reconstructed_from_manifest(self, tmp_path: Path) -> None:
        """SnapshotRouter built from CEL manifest routes keys correctly."""
        store = InMemoryManifestStore()
        config, _ = _make_cel_config(
            tmp_path,
            cel_expr="key % 3",
            cel_columns={"key": "int"},
            boundaries=[1, 2],
        )
        config.manifest.store = store

        result = write_sharded(
            list(range(30)),
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: b"v",
        )

        # Reconstruct router from manifest
        manifest = store.load_manifest(result.manifest_ref)
        router = SnapshotRouter(manifest.required_build, manifest.shards)

        # Verify routing matches write-time routing
        assert router.strategy == ShardingStrategy.CEL
        for key in range(30):
            write_db_id = route_key(
                key,
                num_dbs=3,
                sharding=config.sharding,
                key_encoding=config.key_encoding,
            )
            read_db_id = router.route_one(key)
            assert write_db_id == read_db_id, f"key={key}"

    def test_single_shard_all_keys(self, tmp_path: Path) -> None:
        """With no boundaries, all keys route to shard 0."""
        store = InMemoryManifestStore()
        root_dir = str(tmp_path / "file_backed")
        config = WriteConfig(
            num_dbs=0,
            s3_prefix="s3://bucket/prefix",
            key_encoding=KeyEncoding.U64BE,
            adapter_factory=file_backed_adapter_factory(root_dir),
            sharding=ShardingSpec(
                strategy=ShardingStrategy.CEL,
                cel_expr="key % 1",
                cel_columns={"key": "int"},
                boundaries=[],
            ),
            output=OutputOptions(run_id="test-single"),
            manifest=ManifestOptions(store=store),
        )

        result = write_sharded(
            list(range(50)),
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: b"v",
        )

        assert len(result.winners) == 1
        assert result.winners[0].db_id == 0
        assert result.winners[0].row_count == 50

    def test_empty_input(self, tmp_path: Path) -> None:
        config, _ = _make_cel_config(
            tmp_path,
            cel_expr="key % 4",
            cel_columns={"key": "int"},
            boundaries=[1, 2, 3],
        )

        result = write_sharded(
            [],
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: b"v",
        )

        assert result.stats.rows_written == 0
        assert len(result.winners) == 0

    def test_data_integrity_all_records_recoverable(self, tmp_path: Path) -> None:
        """Every written record can be read back from the correct shard."""
        config, root_dir = _make_cel_config(
            tmp_path,
            cel_expr="key % 4",
            cel_columns={"key": "int"},
            boundaries=[1, 2, 3],
        )

        records = list(range(200))
        result = write_sharded(
            records,
            config,
            key_fn=lambda r: r,
            value_fn=lambda r: f"value-{r}".encode(),
        )

        # Collect all KV from all shards
        all_kv: dict[bytes, bytes] = {}
        for winner in result.winners:
            shard_data = file_backed_load_db(root_dir, winner.db_url)
            all_kv.update(shard_data)

        assert len(all_kv) == 200
        encoder = make_key_encoder(config.key_encoding)
        for r in records:
            assert all_kv[encoder(r)] == f"value-{r}".encode()


# ---------------------------------------------------------------------------
# CEL Split Mode Tests (routing context)
# ---------------------------------------------------------------------------


class TestCelSplitMode:
    """CEL split mode: CEL evaluates over routing context columns to
    determine the shard, while key_fn provides the DB key separately.

    This simulates a real use case: sharding by region while storing
    user IDs as keys.
    """

    def test_split_mode_routes_by_context(self, tmp_path: Path) -> None:
        """Records route by 'region' context, stored by 'user_id' key."""
        # 3 regions → 3 shards, boundaries at "eu", "us" (sorted: ap < eu < us)
        store = InMemoryManifestStore()
        root_dir = str(tmp_path / "file_backed")
        config = WriteConfig(
            num_dbs=0,
            s3_prefix="s3://bucket/prefix",
            key_encoding=KeyEncoding.UTF8,
            adapter_factory=file_backed_adapter_factory(root_dir),
            sharding=ShardingSpec(
                strategy=ShardingStrategy.CEL,
                cel_expr="region",
                cel_columns={"region": "string"},
                boundaries=["eu", "us"],
            ),
            output=OutputOptions(run_id="test-split"),
            manifest=ManifestOptions(store=store),
        )

        # Records: (user_id, region)
        records = [
            ("alice", "ap"),
            ("bob", "eu"),
            ("carol", "us"),
            ("dave", "ap"),
            ("eve", "eu"),
            ("frank", "us"),
        ]

        result = write_sharded(
            records,
            config,
            key_fn=lambda r: r[0],
            value_fn=lambda r: f"{r[0]}@{r[1]}".encode(),
            columns_fn=lambda r: {"region": r[1]},
        )

        assert result.stats.rows_written == 6
        assert len(result.winners) == 3

        # Verify shard distribution: ap→0, eu→1, us→2
        compiled = compile_cel("region", {"region": "string"})
        for winner in result.winners:
            shard_data = file_backed_load_db(root_dir, winner.db_url)
            for key_bytes, val_bytes in shard_data.items():
                user_id = key_bytes.decode("utf-8")
                payload = val_bytes.decode("utf-8")
                region = payload.split("@")[1]
                expected_db_id = route_cel(compiled, {"region": region}, ["eu", "us"])
                assert expected_db_id == winner.db_id, (
                    f"user={user_id}, region={region}, "
                    f"expected={expected_db_id}, actual={winner.db_id}"
                )

    def test_split_mode_read_side_routing(self, tmp_path: Path) -> None:
        """Router.route_with_context produces same shard IDs as write time."""
        store = InMemoryManifestStore()
        root_dir = str(tmp_path / "file_backed")
        config = WriteConfig(
            num_dbs=0,
            s3_prefix="s3://bucket/prefix",
            key_encoding=KeyEncoding.UTF8,
            adapter_factory=file_backed_adapter_factory(root_dir),
            sharding=ShardingSpec(
                strategy=ShardingStrategy.CEL,
                cel_expr="region",
                cel_columns={"region": "string"},
                boundaries=["eu", "us"],
            ),
            output=OutputOptions(run_id="test-split-read"),
            manifest=ManifestOptions(store=store),
        )

        records = [
            ("user1", "ap"),
            ("user2", "eu"),
            ("user3", "us"),
        ]

        result = write_sharded(
            records,
            config,
            key_fn=lambda r: r[0],
            value_fn=lambda r: b"v",
            columns_fn=lambda r: {"region": r[1]},
        )

        # Reconstruct router and verify route_with_context
        manifest = store.load_manifest(result.manifest_ref)
        router = SnapshotRouter(manifest.required_build, manifest.shards)

        assert router.route_with_context({"region": "ap"}) == 0  # < "eu"
        assert router.route_with_context({"region": "eu"}) == 1  # >= "eu", < "us"
        assert router.route_with_context({"region": "us"}) == 2  # >= "us"

    def test_split_mode_uneven_distribution(self, tmp_path: Path) -> None:
        """Shards can have uneven record counts in split mode."""
        store = InMemoryManifestStore()
        root_dir = str(tmp_path / "file_backed")
        config = WriteConfig(
            num_dbs=0,
            s3_prefix="s3://bucket/prefix",
            key_encoding=KeyEncoding.UTF8,
            adapter_factory=file_backed_adapter_factory(root_dir),
            sharding=ShardingSpec(
                strategy=ShardingStrategy.CEL,
                cel_expr="region",
                cel_columns={"region": "string"},
                boundaries=["eu", "us"],
            ),
            output=OutputOptions(run_id="test-uneven"),
            manifest=ManifestOptions(store=store),
        )

        # Most records in "us" region
        records = (
            [("user" + str(i), "us") for i in range(8)]
            + [("eu_user", "eu")]
            + [("ap_user", "ap")]
        )

        result = write_sharded(
            records,
            config,
            key_fn=lambda r: r[0],
            value_fn=lambda r: b"v",
            columns_fn=lambda r: {"region": r[1]},
        )

        assert result.stats.rows_written == 10
        # Find the "us" shard (db_id=2) — should have 8 rows
        us_shard = next(w for w in result.winners if w.db_id == 2)
        assert us_shard.row_count == 8
