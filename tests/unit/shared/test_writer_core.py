from __future__ import annotations

import pytest

from shardyfusion._writer_core import (
    build_categorical_routing_values,
    discover_cel_num_dbs,
    resolve_num_dbs,
)
from shardyfusion.config import CelWriteConfig, HashWriteConfig
from shardyfusion.errors import ConfigValidationError, ShardAssignmentError
from shardyfusion.sharding_types import HashShardingSpec

# ---------------------------------------------------------------------------
# resolve_num_dbs
# ---------------------------------------------------------------------------


class TestResolveNumDbs:
    def test_explicit_num_dbs_returned_directly(self) -> None:
        cfg = HashWriteConfig(num_dbs=4, s3_prefix="s3://b/p")
        result = resolve_num_dbs(cfg, count_fn=lambda: 999)
        assert result == 4

    def test_explicit_num_dbs_does_not_call_count_fn(self) -> None:
        cfg = HashWriteConfig(num_dbs=4, s3_prefix="s3://b/p")
        called = False

        def boom() -> int:
            nonlocal called
            called = True
            return 0

        resolve_num_dbs(cfg, count_fn=boom)
        assert not called

    def test_max_keys_per_shard_basic(self) -> None:
        cfg = HashWriteConfig(
            s3_prefix="s3://b/p",
            max_keys_per_shard=100,
        )
        result = resolve_num_dbs(cfg, count_fn=lambda: 500)
        assert result == 5

    def test_max_keys_per_shard_rounds_up(self) -> None:
        cfg = HashWriteConfig(
            s3_prefix="s3://b/p",
            max_keys_per_shard=3,
        )
        result = resolve_num_dbs(cfg, count_fn=lambda: 10)
        assert result == 4  # ceil(10/3)

    def test_max_keys_per_shard_zero_count_returns_one(self) -> None:
        cfg = HashWriteConfig(
            s3_prefix="s3://b/p",
            max_keys_per_shard=100,
        )
        result = resolve_num_dbs(cfg, count_fn=lambda: 0)
        assert result == 1

    def test_cel_mode_returns_none(self) -> None:
        cfg = CelWriteConfig(
            s3_prefix="s3://b/p",
            cel_expr="key % 4",
            cel_columns={"key": "int"},
        )
        result = resolve_num_dbs(cfg, count_fn=lambda: 1000)
        assert result is None


# ---------------------------------------------------------------------------
# discover_cel_num_dbs
# ---------------------------------------------------------------------------


class TestDiscoverCelNumDbs:
    def test_consecutive_ids(self) -> None:
        assert discover_cel_num_dbs({0, 1, 2}) == 3

    def test_single_id(self) -> None:
        assert discover_cel_num_dbs({0}) == 1

    def test_empty_returns_one(self) -> None:
        assert discover_cel_num_dbs(set()) == 1

    def test_five_consecutive_ids(self) -> None:
        assert discover_cel_num_dbs({0, 1, 2, 3, 4}) == 5

    def test_gap_raises(self) -> None:
        with pytest.raises(ShardAssignmentError, match="non-consecutive"):
            discover_cel_num_dbs({0, 2})

    def test_non_zero_based_raises(self) -> None:
        with pytest.raises(ShardAssignmentError, match="non-consecutive"):
            discover_cel_num_dbs({1, 2})


class TestBuildCategoricalRoutingValues:
    def test_sorts_and_deduplicates_values(self) -> None:
        assert build_categorical_routing_values(["us", "ap", "us", "eu"]) == [
            "ap",
            "eu",
            "us",
        ]

    def test_rejects_mixed_bool_and_int_before_dedup(self) -> None:
        with pytest.raises(ConfigValidationError, match="boolean"):
            build_categorical_routing_values([1, True])  # type: ignore[list-item]

    def test_rejects_float_values(self) -> None:
        with pytest.raises(ConfigValidationError, match="int, str, or bytes"):
            build_categorical_routing_values([1.5, 2.5])  # type: ignore[list-item]

    def test_rejects_bytearray_values(self) -> None:
        with pytest.raises(ConfigValidationError, match="int, str, or bytes"):
            build_categorical_routing_values([bytearray(b"a")])  # type: ignore[list-item]
