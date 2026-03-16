"""Tests for ShardingSpec with CEL fields and expanded types."""

import pytest

from shardyfusion.sharding_types import BoundaryValue, ShardingSpec, ShardingStrategy


class TestShardingSpecCel:
    def test_cel_requires_cel_expr(self) -> None:
        with pytest.raises(ValueError, match="cel_expr"):
            ShardingSpec(strategy=ShardingStrategy.CEL, cel_columns={"key": "int"})

    def test_cel_requires_cel_columns(self) -> None:
        with pytest.raises(ValueError, match="cel_columns"):
            ShardingSpec(strategy=ShardingStrategy.CEL, cel_expr="key % 10")

    def test_cel_valid(self) -> None:
        spec = ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="key % 10",
            cel_columns={"key": "int"},
        )
        assert spec.cel_expr == "key % 10"
        assert spec.cel_columns == {"key": "int"}

    def test_cel_expr_rejected_for_hash(self) -> None:
        with pytest.raises(ValueError, match="cel_expr is only valid with CEL"):
            ShardingSpec(
                strategy=ShardingStrategy.HASH,
                cel_expr="key % 10",
            )

    def test_max_keys_per_shard_positive(self) -> None:
        with pytest.raises(ValueError, match="max_keys_per_shard must be > 0"):
            ShardingSpec(
                strategy=ShardingStrategy.CEL,
                cel_expr="key",
                cel_columns={"key": "int"},
                max_keys_per_shard=0,
            )

    def test_max_keys_per_shard_valid(self) -> None:
        spec = ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="key",
            cel_columns={"key": "int"},
            max_keys_per_shard=1000,
        )
        assert spec.max_keys_per_shard == 1000


class TestBoundaryValueBytes:
    def test_bytes_boundary_type(self) -> None:
        boundaries: list[BoundaryValue] = [b"\x00", b"\x80", b"\xff"]
        assert all(isinstance(b, bytes) for b in boundaries)

    def test_mixed_boundary_types(self) -> None:
        boundaries: list[BoundaryValue] = [10, 20.5, "hello", b"\x00"]
        assert len(boundaries) == 4


class TestToManifestDict:
    def test_cel_fields_included(self) -> None:
        spec = ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="key % 10",
            cel_columns={"key": "int"},
            boundaries=[100, 200],
        )
        d = spec.to_manifest_dict()
        assert d["strategy"] == "cel"
        assert d["cel_expr"] == "key % 10"
        assert d["cel_columns"] == {"key": "int"}
        assert d["boundaries"] == [100, 200]

    def test_hash_no_cel_fields(self) -> None:
        spec = ShardingSpec(strategy=ShardingStrategy.HASH)
        d = spec.to_manifest_dict()
        assert "cel_expr" not in d
        assert "cel_columns" not in d
