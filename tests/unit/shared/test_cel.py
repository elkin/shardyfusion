"""Tests for CEL compilation, evaluation, and routing.

These tests require the ``cel`` extra (cel-expr-python).
"""

import pytest

cel_expr_python = pytest.importorskip("cel_expr_python")

from shardyfusion.cel import (
    compile_cel,
    evaluate_cel_arrow_batch,
    route_cel,
)

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
