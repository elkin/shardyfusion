"""Heap-based top-k merge across shard search results."""

from __future__ import annotations

import heapq

from .types import DistanceMetric, SearchResult


def merge_results(
    per_shard_results: list[list[SearchResult]],
    top_k: int,
    metric: DistanceMetric,
) -> list[SearchResult]:
    """Merge per-shard results into a single top-k list.

    For L2 and COSINE distance: lower score = better (use nsmallest).
    For DOT_PRODUCT similarity: higher score = better (use nlargest).
    """
    all_results = [r for shard in per_shard_results for r in shard]
    if not all_results:
        return []

    if metric == DistanceMetric.DOT_PRODUCT:
        return heapq.nlargest(top_k, all_results, key=lambda r: r.score)
    return heapq.nsmallest(top_k, all_results, key=lambda r: r.score)
