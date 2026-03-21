"""Tests for SnapshotRouter CEL routing modes."""

from __future__ import annotations

import pytest

from shardyfusion.manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.routing import SnapshotRouter
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy

# Guard CEL tests — skip if cel-expr-python not installed
cel_expr_python = pytest.importorskip("cel_expr_python")

pytestmark = pytest.mark.cel


def _build_required(
    *,
    num_dbs: int,
    cel_expr: str,
    cel_columns: dict[str, str],
    boundaries: list[int | str] | None = None,
    key_encoding: KeyEncoding = KeyEncoding.UTF8,
) -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id="run-cel",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="key",
        key_encoding=key_encoding,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr=cel_expr,
            cel_columns=cel_columns,
            boundaries=boundaries,
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )


def _make_shards(num_dbs: int) -> list[RequiredShardMeta]:
    return [
        RequiredShardMeta(
            db_id=i,
            db_url=f"s3://bucket/prefix/db={i:05d}",
            attempt=0,
            row_count=10,
        )
        for i in range(num_dbs)
    ]


class TestSnapshotRouterCelKeyOnly:
    """CEL key-only mode: auto-wraps key in {"key": value}."""

    def test_cel_direct_mode_no_boundaries(self) -> None:
        """CEL expression directly returns shard ID."""
        required = _build_required(
            num_dbs=4,
            cel_expr="key % 4",
            cel_columns={"key": "int"},
        )
        router = SnapshotRouter(required, _make_shards(4))

        assert router.route_one(0) == 0
        assert router.route_one(1) == 1
        assert router.route_one(5) == 1

    def test_cel_with_boundaries(self) -> None:
        """CEL expression + bisect_right boundaries."""
        required = _build_required(
            num_dbs=3,
            cel_expr="key",
            cel_columns={"key": "int"},
            boundaries=[10, 20],
        )
        router = SnapshotRouter(required, _make_shards(3))

        assert router.route_one(5) == 0
        assert router.route_one(15) == 1
        assert router.route_one(25) == 2


class TestSnapshotRouterCelMultiColumn:
    """CEL multi-column mode: requires routing_context."""

    def test_route_with_context(self) -> None:
        """route(key, routing_context=...) works for multi-column CEL."""
        required = _build_required(
            num_dbs=3,
            cel_expr="region",
            cel_columns={"region": "string"},
            boundaries=["eu", "us"],
        )
        router = SnapshotRouter(required, _make_shards(3))

        assert router.route_with_context({"region": "ap"}) == 0
        assert router.route_with_context({"region": "eu"}) == 1

    def test_route_one_raises_for_multi_column(self) -> None:
        """route_one(key) raises ValueError for multi-column CEL."""
        required = _build_required(
            num_dbs=3,
            cel_expr="region",
            cel_columns={"region": "string"},
            boundaries=["eu", "us"],
        )
        router = SnapshotRouter(required, _make_shards(3))

        with pytest.raises(ValueError, match="routing_context"):
            router.route_one("anything")


class TestSnapshotRouterBoundaryValidation:
    """Router rejects unsorted boundaries from manifest data."""

    def test_rejects_unsorted_boundaries(self) -> None:
        required = _build_required(
            num_dbs=3,
            cel_expr="key",
            cel_columns={"key": "int"},
            boundaries=[20, 10],
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            SnapshotRouter(required, _make_shards(3))

    def test_rejects_duplicate_boundaries(self) -> None:
        required = _build_required(
            num_dbs=3,
            cel_expr="key",
            cel_columns={"key": "int"},
            boundaries=[10, 10],
        )
        with pytest.raises(ValueError, match="strictly increasing"):
            SnapshotRouter(required, _make_shards(3))
