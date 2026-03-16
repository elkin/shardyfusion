"""Tests for CEL compilation, evaluation, and boundary computation.

These tests require the ``cel`` extra (cel-expr-python, fastdigest).
"""

import pytest

cel_expr_python = pytest.importorskip("cel_expr_python")
fastdigest = pytest.importorskip("fastdigest")

from shardyfusion.cel import (
    compile_cel,
    compute_boundaries_distinct,
    compute_boundaries_tdigest,
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


class TestComputeBoundariesDistinct:
    def test_distinct_boundaries(self) -> None:
        routing_keys = [1, 2, 3, 4, 5, 1, 2, 3]
        num_dbs, boundaries = compute_boundaries_distinct(routing_keys)
        assert num_dbs == 5
        assert boundaries == [2, 3, 4, 5]

    def test_single_value(self) -> None:
        num_dbs, boundaries = compute_boundaries_distinct([42, 42, 42])
        assert num_dbs == 1
        assert boundaries == []

    def test_empty(self) -> None:
        num_dbs, boundaries = compute_boundaries_distinct([])
        assert num_dbs == 1
        assert boundaries == []


class TestComputeBoundariesTdigest:
    def test_basic_boundaries(self) -> None:
        keys = list(range(1000))
        boundaries = compute_boundaries_tdigest(keys, 4)
        assert len(boundaries) == 3
        # Boundaries should be roughly at 250, 500, 750
        assert all(isinstance(b, int) for b in boundaries)

    def test_single_shard(self) -> None:
        boundaries = compute_boundaries_tdigest([1, 2, 3], 1)
        assert boundaries == []

    def test_string_keys_hashed(self) -> None:
        keys = [f"key_{i}" for i in range(100)]
        boundaries = compute_boundaries_tdigest(keys, 4)
        assert len(boundaries) == 3


class TestEvaluateCelArrowBatch:
    def test_arrow_batch_evaluation(self) -> None:
        pa = pytest.importorskip("pyarrow")

        compiled = compile_cel("key % 10", {"key": "int"})
        table = pa.table({"key": [10, 25, 30, 42]})
        results = evaluate_cel_arrow_batch(compiled, table)
        assert results == [0, 5, 0, 2]
