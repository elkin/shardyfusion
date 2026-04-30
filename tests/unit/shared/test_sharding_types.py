"""Tests for split sharding specs and categorical routing values."""

import pytest

from shardyfusion.sharding_types import (
    CelShardingSpec,
    HashShardingSpec,
    ShardHashAlgorithm,
)


class TestHashShardingSpec:
    def test_defaults(self) -> None:
        spec = HashShardingSpec()
        assert spec.hash_algorithm == ShardHashAlgorithm.XXH3_64
        assert spec.max_keys_per_shard is None

    def test_custom_hash_algorithm(self) -> None:
        spec = HashShardingSpec(hash_algorithm="xxh3_64")  # type: ignore[arg-type]
        assert spec.hash_algorithm == ShardHashAlgorithm.XXH3_64

    def test_max_keys_per_shard(self) -> None:
        spec = HashShardingSpec(max_keys_per_shard=1000)
        assert spec.max_keys_per_shard == 1000

    def test_rejects_non_positive_max_keys(self) -> None:
        with pytest.raises(ValueError, match="max_keys_per_shard must be > 0"):
            HashShardingSpec(max_keys_per_shard=0)

    def test_to_manifest_dict(self) -> None:
        spec = HashShardingSpec()
        d = spec.to_manifest_dict()
        assert d["strategy"] == "hash"
        assert d["hash_algorithm"] == "xxh3_64"


class TestCelShardingSpec:
    def test_requires_cel_expr(self) -> None:
        with pytest.raises(ValueError, match="cel_expr"):
            CelShardingSpec(cel_columns={"key": "int"})

    def test_requires_cel_columns(self) -> None:
        with pytest.raises(ValueError, match="cel_columns"):
            CelShardingSpec(cel_expr="key % 10")

    def test_valid(self) -> None:
        spec = CelShardingSpec(
            cel_expr="key % 10",
            cel_columns={"key": "int"},
        )
        assert spec.cel_expr == "key % 10"
        assert spec.cel_columns == {"key": "int"}

    def test_accepts_routing_values(self) -> None:
        spec = CelShardingSpec(
            cel_expr="region",
            cel_columns={"region": "string"},
            routing_values=["ap", "eu", "us"],
        )
        assert spec.routing_values == ["ap", "eu", "us"]

    def test_rejects_infer_flag_with_routing_values(self) -> None:
        with pytest.raises(ValueError, match="cannot be combined"):
            CelShardingSpec(
                cel_expr="region",
                cel_columns={"region": "string"},
                routing_values=["ap", "eu"],
                infer_routing_values_from_data=True,
            )

    def test_to_manifest_dict(self) -> None:
        spec = CelShardingSpec(
            cel_expr="key % 10",
            cel_columns={"key": "int"},
        )
        d = spec.to_manifest_dict()
        assert d["strategy"] == "cel"
        assert d["cel_expr"] == "key % 10"
        assert d["cel_columns"] == {"key": "int"}
        assert "routing_values" not in d

    def test_to_manifest_dict_with_routing_values(self) -> None:
        spec = CelShardingSpec(
            cel_expr="region",
            cel_columns={"region": "string"},
            routing_values=["ap", "eu", "us"],
        )
        d = spec.to_manifest_dict()
        assert d["routing_values"] == ["ap", "eu", "us"]


class TestRoutingValueValidation:
    def _cel_spec(self, routing_values: list[int | str | bytes]) -> CelShardingSpec:
        return CelShardingSpec(
            cel_expr="region",
            cel_columns={"region": "string"},
            routing_values=routing_values,
        )

    def test_rejects_nulls(self) -> None:
        with pytest.raises(ValueError, match="must not contain null values"):
            self._cel_spec(["ap", None])  # type: ignore[list-item]

    def test_rejects_booleans(self) -> None:
        with pytest.raises(ValueError, match="must not be boolean"):
            self._cel_spec([True])  # type: ignore[list-item]

    def test_rejects_mixed_types(self) -> None:
        with pytest.raises(ValueError, match="must all share the same type"):
            self._cel_spec([10, "hello"])  # type: ignore[list-item]

    def test_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            self._cel_spec(["ap", "ap"])

    def test_rejects_floats(self) -> None:
        with pytest.raises(ValueError, match="must be int, str, or bytes"):
            self._cel_spec([1.5, 2.5])  # type: ignore[list-item]

    def test_rejects_bytearrays(self) -> None:
        with pytest.raises(ValueError, match="must be int, str, or bytes"):
            self._cel_spec([bytearray(b"a")])  # type: ignore[list-item]

    def test_accepts_valid_ints(self) -> None:
        spec = self._cel_spec([10, 20, 30])
        assert spec.routing_values == [10, 20, 30]

    def test_accepts_valid_strings(self) -> None:
        spec = self._cel_spec(["a", "b", "c"])
        assert spec.routing_values == ["a", "b", "c"]

    def test_accepts_valid_bytes(self) -> None:
        spec = self._cel_spec([b"\x00", b"\x80", b"\xff"])
        assert spec.routing_values == [b"\x00", b"\x80", b"\xff"]


class TestToManifestDict:
    def test_cel_fields_included(self) -> None:
        spec = CelShardingSpec(
            cel_expr="key % 10",
            cel_columns={"key": "int"},
        )
        d = spec.to_manifest_dict()
        assert d["strategy"] == "cel"
        assert d["hash_algorithm"] == "xxh3_64"
        assert d["cel_expr"] == "key % 10"
        assert d["cel_columns"] == {"key": "int"}
        assert "routing_values" not in d

    def test_categorical_fields_included(self) -> None:
        spec = CelShardingSpec(
            cel_expr="region",
            cel_columns={"region": "string"},
            routing_values=["ap", "eu", "us"],
        )
        d = spec.to_manifest_dict()
        assert d["routing_values"] == ["ap", "eu", "us"]

    def test_hash_no_cel_fields(self) -> None:
        spec = HashShardingSpec()
        d = spec.to_manifest_dict()
        assert "cel_expr" not in d
        assert "cel_columns" not in d
        assert d["hash_algorithm"] == "xxh3_64"


class TestShardHashAlgorithm:
    def test_from_value_accepts_string(self) -> None:
        assert ShardHashAlgorithm.from_value("xxh3_64") == ShardHashAlgorithm.XXH3_64

    def test_from_value_rejects_unknown(self) -> None:
        with pytest.raises(ValueError, match="Unsupported shard hash algorithm"):
            ShardHashAlgorithm.from_value("future_hash")
