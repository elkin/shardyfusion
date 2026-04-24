"""Heap-based top-k merge across shard search results."""

from __future__ import annotations

import heapq

from .types import DistanceMetric, SearchResult

_SUPPORTED_NORMALIZED_METRICS = frozenset(
    {
        DistanceMetric.COSINE,
        DistanceMetric.L2,
        DistanceMetric.DOT_PRODUCT,
    }
)


def merge_results(
    per_shard_results: list[list[SearchResult]],
    top_k: int,
    metric: DistanceMetric,
) -> list[SearchResult]:
    """Merge per-shard results into a single top-k list.

    All backends store distances in ``SearchResult.score`` where lower is
    better, including dot-product (negated so lower = higher similarity).
    """
    if metric not in _SUPPORTED_NORMALIZED_METRICS:
        raise ValueError(f"Unsupported distance metric for result merge: {metric!r}")

    all_results = [r for shard in per_shard_results for r in shard]
    if not all_results:
        return []

    return heapq.nsmallest(top_k, all_results, key=lambda r: r.score)
