"""Tests for UnifiedShardedReader — merged KV + vector search."""

from __future__ import annotations

import pytest

from shardyfusion.errors import ConfigValidationError
from shardyfusion.reader.unified_reader import (
    UnifiedVectorMeta,
    _merge_top_k,
    _parse_vector_custom,
)
from shardyfusion.vector.types import SearchResult

# ---------------------------------------------------------------------------
# _parse_vector_custom
# ---------------------------------------------------------------------------


class TestParseVectorCustom:
    def test_valid_custom(self) -> None:
        custom = {
            "vector": {
                "dim": 128,
                "metric": "cosine",
                "index_type": "hnsw",
                "quantization": None,
                "index_params": {"M": 16},
                "unified": True,
                "backend": "usearch-sidecar",
            }
        }
        meta = _parse_vector_custom(custom)
        assert meta.dim == 128
        assert meta.metric == "cosine"
        assert meta.index_type == "hnsw"
        assert meta.quantization is None
        assert meta.index_params == {"M": 16}
        assert meta.backend == "usearch-sidecar"

    def test_missing_vector_key(self) -> None:
        with pytest.raises(ConfigValidationError, match="vector metadata"):
            _parse_vector_custom({})

    def test_vector_not_dict(self) -> None:
        with pytest.raises(ConfigValidationError, match="vector metadata"):
            _parse_vector_custom({"vector": "not a dict"})

    def test_defaults(self) -> None:
        custom = {"vector": {"dim": 64, "metric": "l2"}}
        meta = _parse_vector_custom(custom)
        assert meta.dim == 64
        assert meta.metric == "l2"
        assert meta.index_type == "hnsw"
        assert meta.quantization is None
        assert meta.index_params == {}
        assert meta.backend == "usearch-sidecar"  # default

    def test_sqlite_vec_backend(self) -> None:
        custom = {
            "vector": {
                "dim": 32,
                "metric": "cosine",
                "backend": "sqlite-vec",
            }
        }
        meta = _parse_vector_custom(custom)
        assert meta.backend == "sqlite-vec"


# ---------------------------------------------------------------------------
# _merge_top_k
# ---------------------------------------------------------------------------


class TestMergeTopK:
    def _results(self, scores: list[float]) -> list[SearchResult]:
        return [SearchResult(id=i, score=s) for i, s in enumerate(scores)]

    def test_cosine_lower_is_better(self) -> None:
        results = self._results([0.9, 0.1, 0.5, 0.3])
        merged = _merge_top_k(results, top_k=2, metric="cosine")
        assert len(merged) == 2
        assert merged[0].score == 0.1
        assert merged[1].score == 0.3

    def test_l2_lower_is_better(self) -> None:
        results = self._results([10.0, 1.0, 5.0])
        merged = _merge_top_k(results, top_k=2, metric="l2")
        assert merged[0].score == 1.0
        assert merged[1].score == 5.0

    def test_dot_product_higher_is_better(self) -> None:
        results = self._results([0.1, 0.9, 0.5])
        merged = _merge_top_k(results, top_k=2, metric="dot_product")
        assert merged[0].score == 0.9
        assert merged[1].score == 0.5

    def test_empty_results(self) -> None:
        assert _merge_top_k([], top_k=5, metric="cosine") == []

    def test_top_k_larger_than_results(self) -> None:
        results = self._results([0.5, 0.3])
        merged = _merge_top_k(results, top_k=10, metric="cosine")
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# UnifiedVectorMeta
# ---------------------------------------------------------------------------


class TestUnifiedVectorMeta:
    def test_frozen(self) -> None:
        meta = UnifiedVectorMeta(
            dim=128,
            metric="cosine",
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="usearch-sidecar",
            kv_backend="slatedb",
        )
        with pytest.raises(AttributeError):
            meta.dim = 64  # type: ignore[misc]
