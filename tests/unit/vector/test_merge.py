"""Tests for heap-based result merging."""

from __future__ import annotations

from shardyfusion.vector._merge import merge_results
from shardyfusion.vector.types import DistanceMetric, SearchResult


def _r(id: int, score: float) -> SearchResult:
    return SearchResult(id=id, score=score)


class TestMergeResults:
    def test_empty_input(self):
        assert merge_results([], 10, DistanceMetric.L2) == []

    def test_empty_shards(self):
        assert merge_results([[], []], 10, DistanceMetric.L2) == []

    def test_l2_lower_is_better(self):
        shard1 = [_r(1, 0.1), _r(2, 0.5)]
        shard2 = [_r(3, 0.05), _r(4, 0.3)]
        merged = merge_results([shard1, shard2], 3, DistanceMetric.L2)
        assert len(merged) == 3
        assert merged[0].id == 3  # 0.05
        assert merged[1].id == 1  # 0.1
        assert merged[2].id == 4  # 0.3

    def test_cosine_lower_is_better(self):
        shard1 = [_r(1, 0.1), _r(2, 0.9)]
        shard2 = [_r(3, 0.05)]
        merged = merge_results([shard1, shard2], 2, DistanceMetric.COSINE)
        assert len(merged) == 2
        assert merged[0].id == 3  # 0.05
        assert merged[1].id == 1  # 0.1

    def test_dot_product_lower_distance_is_better(self):
        # Backends return negated dot product as distance: lower = more similar
        shard1 = [_r(1, -10.0), _r(2, -5.0)]
        shard2 = [_r(3, -15.0), _r(4, -1.0)]
        merged = merge_results([shard1, shard2], 3, DistanceMetric.DOT_PRODUCT)
        assert len(merged) == 3
        assert merged[0].id == 3  # -15.0 (most similar)
        assert merged[1].id == 1  # -10.0
        assert merged[2].id == 2  # -5.0

    def test_top_k_limits_output(self):
        shard1 = [_r(i, float(i)) for i in range(20)]
        merged = merge_results([shard1], 5, DistanceMetric.L2)
        assert len(merged) == 5

    def test_single_shard(self):
        shard1 = [_r(1, 0.1), _r(2, 0.2)]
        merged = merge_results([shard1], 10, DistanceMetric.L2)
        assert len(merged) == 2

    def test_fewer_results_than_top_k(self):
        shard1 = [_r(1, 0.1)]
        merged = merge_results([shard1], 10, DistanceMetric.L2)
        assert len(merged) == 1
