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

    def test_max_keys_per_shard_rejected_for_cel(self) -> None:
        with pytest.raises(
            ValueError, match="max_keys_per_shard is only valid with HASH"
        ):
            ShardingSpec(
                strategy=ShardingStrategy.CEL,
                cel_expr="key",
                cel_columns={"key": "int"},
                max_keys_per_shard=1000,
            )

    def test_max_keys_per_shard_positive(self) -> None:
        with pytest.raises(ValueError, match="max_keys_per_shard must be > 0"):
            ShardingSpec(
                strategy=ShardingStrategy.HASH,
                max_keys_per_shard=0,
            )

    def test_max_keys_per_shard_valid(self) -> None:
        spec = ShardingSpec(
            strategy=ShardingStrategy.HASH,
            max_keys_per_shard=1000,
        )
        assert spec.max_keys_per_shard == 1000


class TestBoundaryValidation:
    """Boundaries must be strictly increasing, non-null, same-type, non-boolean."""

    def _cel_spec(self, boundaries: list[BoundaryValue] | None) -> ShardingSpec:
        return ShardingSpec(
            strategy=ShardingStrategy.CEL,
            cel_expr="key",
            cel_columns={"key": "int"},
            boundaries=boundaries,
        )

    def test_rejects_unsorted(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            self._cel_spec([20, 10])

    def test_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            self._cel_spec([10, 10])

    def test_rejects_nulls(self) -> None:
        with pytest.raises(ValueError, match="null"):
            self._cel_spec([10, None])  # type: ignore[list-item]

    def test_rejects_booleans(self) -> None:
        with pytest.raises(ValueError, match="boolean"):
            self._cel_spec([True])

    def test_rejects_mixed_types(self) -> None:
        with pytest.raises(ValueError, match="same type"):
            self._cel_spec([10, "hello"])  # type: ignore[list-item]

    def test_accepts_valid_ints(self) -> None:
        spec = self._cel_spec([10, 20, 30])
        assert spec.boundaries == [10, 20, 30]

    def test_accepts_valid_strings(self) -> None:
        spec = self._cel_spec(["a", "b", "c"])
        assert spec.boundaries == ["a", "b", "c"]

    def test_accepts_valid_bytes(self) -> None:
        spec = self._cel_spec([b"\x00", b"\x80", b"\xff"])
        assert spec.boundaries == [b"\x00", b"\x80", b"\xff"]

    def test_accepts_none_boundaries(self) -> None:
        spec = self._cel_spec(None)
        assert spec.boundaries is None

    def test_accepts_empty_boundaries(self) -> None:
        spec = self._cel_spec([])
        assert spec.boundaries == []

    def test_accepts_single_boundary(self) -> None:
        spec = self._cel_spec([42])
        assert spec.boundaries == [42]


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
