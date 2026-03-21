"""Tests for CEL compilation, evaluation, and routing.

These tests require the ``cel`` extra (cel-expr-python).
"""

import json

import pytest

cel_expr_python = pytest.importorskip("cel_expr_python")

from shardyfusion.cel import (
    cel_sharding,
    cel_sharding_by_columns,
    compile_cel,
    evaluate_cel_arrow_batch,
    route_cel,
)
from shardyfusion.errors import ConfigValidationError
from shardyfusion.sharding_types import ShardingStrategy

pytestmark = pytest.mark.cel


class TestCompileCel:
    def test_compile_simple_expr(self) -> None:
        compiled = compile_cel("key % 10", {"key": "int"})
        assert compiled.columns == {"key": "int"}

    def test_compile_string_expr(self) -> None:
        compiled = compile_cel("region", {"region": "string"})
        assert compiled.columns == {"region": "string"}

    def test_compile_invalid_type(self) -> None:
        with pytest.raises(Exception, match="Unsupported CEL column type"):
            compile_cel("key", {"key": "unknown_type"})

    def test_compile_invalid_expr(self) -> None:
        with pytest.raises(Exception, match="Failed to compile"):
            compile_cel("invalid syntax !@#", {"key": "int"})


class TestEvaluateCel:
    def test_evaluate_modulo(self) -> None:
        compiled = compile_cel("key % 10", {"key": "int"})
        assert compiled.evaluate({"key": 42}) == 2
        assert compiled.evaluate({"key": 100}) == 0

    def test_evaluate_string_passthrough(self) -> None:
        compiled = compile_cel("region", {"region": "string"})
        assert compiled.evaluate({"region": "us-east"}) == "us-east"


class TestRouteCel:
    def test_route_with_boundaries(self) -> None:
        compiled = compile_cel("key % 100", {"key": "int"})
        boundaries = [25, 50, 75]

        assert route_cel(compiled, {"key": 10}, boundaries) == 0  # 10 < 25
        assert route_cel(compiled, {"key": 30}, boundaries) == 1  # 25 <= 30 < 50
        assert route_cel(compiled, {"key": 60}, boundaries) == 2  # 50 <= 60 < 75
        assert route_cel(compiled, {"key": 80}, boundaries) == 3  # 80 >= 75

    def test_route_direct_mode(self) -> None:
        compiled = compile_cel("shard_hash(key) % 4u", {"key": "int"})

        shard_id = route_cel(compiled, {"key": 42}, None)
        assert isinstance(shard_id, int)
        assert 0 <= shard_id < 4


class TestShardHash:
    def test_int_key(self) -> None:
        compiled = compile_cel("shard_hash(key) % 100u", {"key": "int"})
        result = compiled.evaluate({"key": 42})
        assert isinstance(result, int)
        assert 0 <= result < 100

    def test_string_key(self) -> None:
        compiled = compile_cel("shard_hash(key) % 100u", {"key": "string"})
        result = compiled.evaluate({"key": "hello"})
        assert isinstance(result, int)
        assert 0 <= result < 100

    def test_bytes_key(self) -> None:
        compiled = compile_cel("shard_hash(key) % 100u", {"key": "bytes"})
        result = compiled.evaluate({"key": b"hello"})
        assert isinstance(result, int)
        assert 0 <= result < 100

    def test_deterministic(self) -> None:
        compiled = compile_cel("shard_hash(key) % 100u", {"key": "int"})
        first = compiled.evaluate({"key": 99})
        second = compiled.evaluate({"key": 99})
        assert first == second


class TestEvaluateCelArrowBatch:
    def test_arrow_batch_evaluation(self) -> None:
        pa = pytest.importorskip("pyarrow")

        compiled = compile_cel("key % 10", {"key": "int"})
        table = pa.table({"key": [10, 25, 30, 42]})
        results = evaluate_cel_arrow_batch(compiled, table)
        assert results == [0, 5, 0, 2]


class TestCelShardingByColumns:
    def test_single_string_column(self) -> None:
        spec = cel_sharding_by_columns("region", num_shards=10)
        assert spec.strategy == ShardingStrategy.CEL
        assert spec.cel_expr == "shard_hash(region) % 10u"
        assert spec.cel_columns == {"region": "string"}
        assert spec.boundaries is None

    def test_single_column_with_cel_column(self) -> None:
        from shardyfusion.cel import CelColumn, CelType

        spec = cel_sharding_by_columns(CelColumn("user_id", CelType.INT), num_shards=4)
        assert spec.cel_expr == "shard_hash(user_id) % 4u"
        assert spec.cel_columns == {"user_id": "int"}

    def test_cel_column_default_type(self) -> None:
        from shardyfusion.cel import CelColumn, CelType

        col = CelColumn("region")
        assert col.type == CelType.STRING
        spec = cel_sharding_by_columns(col, num_shards=5)
        assert spec.cel_columns == {"region": "string"}

    def test_multiple_string_columns(self) -> None:
        spec = cel_sharding_by_columns("region", "env", num_shards=8)
        assert spec.cel_expr == 'shard_hash(region + ":" + env) % 8u'
        assert spec.cel_columns == {"region": "string", "env": "string"}

    def test_mixed_types(self) -> None:
        from shardyfusion.cel import CelColumn, CelType

        spec = cel_sharding_by_columns(
            "region", CelColumn("tier", CelType.INT), num_shards=8
        )
        assert spec.cel_expr == 'shard_hash(region + ":" + string(tier)) % 8u'
        assert spec.cel_columns == {"region": "string", "tier": "int"}

    def test_custom_separator(self) -> None:
        spec = cel_sharding_by_columns("a", "b", num_shards=4, separator="/")
        assert spec.cel_expr == 'shard_hash(a + "/" + b) % 4u'

    @pytest.mark.parametrize(
        ("column", "context", "expected_expr"),
        [
            (
                "flag",
                {"flag": True},
                'shard_hash(string(flag)) % 4u',
            ),
            (
                "score",
                {"score": 1.5},
                'shard_hash(string(score)) % 4u',
            ),
        ],
    )
    def test_single_bool_and_double_columns_are_coerced(
        self,
        column: str,
        context: dict[str, bool | float],
        expected_expr: str,
    ) -> None:
        from shardyfusion.cel import CelColumn, CelType

        col_type = CelType.BOOL if column == "flag" else CelType.DOUBLE
        spec = cel_sharding_by_columns(CelColumn(column, col_type), num_shards=4)

        assert spec.cel_expr == expected_expr
        compiled = compile_cel(spec.cel_expr, spec.cel_columns)  # type: ignore[arg-type]
        shard_id = route_cel(compiled, context, spec.boundaries)
        assert 0 <= shard_id < 4

    @pytest.mark.parametrize("separator", ['"', "\\", "\n"])
    def test_separator_is_escaped_in_generated_expression(
        self, separator: str
    ) -> None:
        spec = cel_sharding_by_columns("a", "b", num_shards=4, separator=separator)

        assert spec.cel_expr == f"shard_hash(a + {json.dumps(separator)} + b) % 4u"
        compiled = compile_cel(spec.cel_expr, spec.cel_columns)  # type: ignore[arg-type]
        shard_id = route_cel(compiled, {"a": "left", "b": "right"}, spec.boundaries)
        assert 0 <= shard_id < 4

    def test_error_no_columns(self) -> None:
        with pytest.raises(ConfigValidationError, match="at least one column"):
            cel_sharding_by_columns(num_shards=4)

    def test_error_num_shards_zero(self) -> None:
        with pytest.raises(ConfigValidationError, match="num_shards must be >= 1"):
            cel_sharding_by_columns("key", num_shards=0)

    def test_error_invalid_column_type(self) -> None:
        with pytest.raises(ConfigValidationError, match="must be a str or CelColumn"):
            cel_sharding_by_columns(42, num_shards=4)  # type: ignore[arg-type]

    def test_cel_type_values(self) -> None:
        from shardyfusion.cel import CelType

        assert CelType.INT.value == "int"
        assert CelType.STRING.value == "string"
        assert CelType.BYTES.value == "bytes"
        assert CelType.DOUBLE.value == "double"
        assert CelType.BOOL.value == "bool"
        assert CelType.UINT.value == "uint"

    def test_roundtrip_compiles_and_routes(self) -> None:
        """Generated spec compiles and produces valid shard IDs."""
        from shardyfusion.cel import CelColumn, CelType

        spec = cel_sharding_by_columns(
            "region", CelColumn("tier", CelType.INT), num_shards=6
        )
        compiled = compile_cel(spec.cel_expr, spec.cel_columns)  # type: ignore[arg-type]
        shard_id = route_cel(
            compiled, {"region": "us-west", "tier": 3}, spec.boundaries
        )
        assert isinstance(shard_id, int)
        assert 0 <= shard_id < 6


class TestCelSharding:
    def test_direct_mode_helper(self) -> None:
        spec = cel_sharding("key % 4", {"key": "int"})

        assert spec.strategy == ShardingStrategy.CEL
        assert spec.cel_expr == "key % 4"
        assert spec.cel_columns == {"key": "int"}
        assert spec.boundaries is None

    def test_boundary_mode_helper_accepts_enum_types(self) -> None:
        from shardyfusion.cel import CelType

        spec = cel_sharding(
            "region",
            {"region": CelType.STRING},
            boundaries=["eu", "us"],
        )

        assert spec.strategy == ShardingStrategy.CEL
        assert spec.cel_expr == "region"
        assert spec.cel_columns == {"region": "string"}
        assert spec.boundaries == ["eu", "us"]

    def test_error_no_columns(self) -> None:
        with pytest.raises(ConfigValidationError, match="at least one column"):
            cel_sharding("key % 4", {})

    def test_error_invalid_column_type(self) -> None:
        with pytest.raises(ConfigValidationError, match="Unsupported CEL column type"):
            cel_sharding("key", {"key": "nope"})
