"""Tests for UnifiedShardedReader — merged KV + vector search."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from shardyfusion.errors import ConfigValidationError
from shardyfusion.reader.unified_reader import (
    UnifiedShardedReader,
    UnifiedVectorMeta,
    _auto_reader_factory,
    _distance_metric_from_str,
    _parse_vector_custom,
    _search_shard,
)
from shardyfusion.vector.types import DistanceMetric, SearchResult

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
# _distance_metric_from_str
# ---------------------------------------------------------------------------


class TestDistanceMetricFromStr:
    def test_valid_metric(self) -> None:
        assert _distance_metric_from_str("cosine") is DistanceMetric.COSINE

    def test_invalid_metric(self) -> None:
        with pytest.raises(
            ConfigValidationError, match="Invalid vector metric 'manhattan'"
        ):
            _distance_metric_from_str("manhattan")


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


# ---------------------------------------------------------------------------
# _auto_reader_factory
# ---------------------------------------------------------------------------


class TestAutoReaderFactory:
    def test_sqlite_vec_backend(self) -> None:
        meta = UnifiedVectorMeta(
            dim=32,
            metric="cosine",
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="sqlite-vec",
            kv_backend="sqlite-vec",
        )
        with patch(
            "shardyfusion.sqlite_vec_adapter.SqliteVecReaderFactory"
        ) as mock_cls:
            _auto_reader_factory(meta)
            mock_cls.assert_called_once()

    def test_usearch_sidecar_default_slatedb(self) -> None:
        meta = UnifiedVectorMeta(
            dim=32,
            metric="cosine",
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="usearch-sidecar",
            kv_backend="slatedb",
        )
        with (
            patch(
                "shardyfusion.composite_adapter.CompositeReaderFactory"
            ) as mock_composite,
            patch("shardyfusion.reader._types.SlateDbReaderFactory"),
            patch("shardyfusion.vector.adapters.usearch_adapter.USearchReaderFactory"),
            patch("shardyfusion.storage.create_s3_client"),
        ):
            _auto_reader_factory(meta)
            mock_composite.assert_called_once()
            call_kwargs = mock_composite.call_args[1]
            assert call_kwargs["kv_factory"] is not None

    def test_usearch_sidecar_with_sqlite_kv(self) -> None:
        meta = UnifiedVectorMeta(
            dim=32,
            metric="cosine",
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="usearch-sidecar",
            kv_backend="sqlite",
        )
        with (
            patch(
                "shardyfusion.composite_adapter.CompositeReaderFactory"
            ) as mock_composite,
            patch("shardyfusion.sqlite_adapter.SqliteReaderFactory") as mock_sqlite_kv,
            patch("shardyfusion.vector.adapters.usearch_adapter.USearchReaderFactory"),
            patch("shardyfusion.storage.create_s3_client"),
        ):
            _auto_reader_factory(meta)
            mock_sqlite_kv.assert_called_once()
            call_kwargs = mock_composite.call_args[1]
            assert call_kwargs["kv_factory"] is not None


# ---------------------------------------------------------------------------
# _search_shard
# ---------------------------------------------------------------------------


class TestSearchShard:
    def test_null_reader_returns_empty(self) -> None:
        """Readers without search() (e.g. _NullShardReader) return []."""

        class NullReader:
            def get(self, key: bytes) -> None:
                return None

        result = _search_shard(NullReader(), np.zeros(4), top_k=5, ef=50)
        assert result == []

    def test_searchable_reader(self) -> None:
        expected = [SearchResult(id=1, score=0.5)]

        class FakeReader:
            def search(
                self, query: np.ndarray, top_k: int, ef: int = 50
            ) -> list[SearchResult]:
                return expected

        result = _search_shard(FakeReader(), np.zeros(4), top_k=5, ef=50)
        assert result == expected


# ---------------------------------------------------------------------------
# _parse_vector_custom with kv_backend
# ---------------------------------------------------------------------------


class TestParseVectorCustomKvBackend:
    def test_explicit_kv_backend(self) -> None:
        custom = {
            "vector": {
                "dim": 32,
                "metric": "cosine",
                "kv_backend": "sqlite",
            }
        }
        meta = _parse_vector_custom(custom)
        assert meta.kv_backend == "sqlite"

    def test_default_kv_backend(self) -> None:
        custom = {"vector": {"dim": 32, "metric": "cosine"}}
        meta = _parse_vector_custom(custom)
        assert meta.kv_backend == "slatedb"


class TestUnifiedReaderSearchMerge:
    def _build_reader(self, metric: str) -> UnifiedShardedReader:
        reader = cast(Any, object.__new__(UnifiedShardedReader))
        reader._closed = False
        reader._executor = None
        reader._vector_meta = UnifiedVectorMeta(
            dim=2,
            metric=metric,
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="usearch-sidecar",
            kv_backend="slatedb",
        )
        reader._state = SimpleNamespace(
            readers=[object(), object()],
            router=SimpleNamespace(
                shards=[
                    SimpleNamespace(db_url="s3://a"),
                    SimpleNamespace(db_url="s3://b"),
                ]
            ),
        )
        return cast(UnifiedShardedReader, reader)

    def test_search_uses_shared_merge_utility(self) -> None:
        reader = self._build_reader("cosine")
        query = np.zeros(2, dtype=np.float32)
        with patch("shardyfusion.reader.unified_reader._search_shard") as mock_search:
            mock_search.side_effect = [
                [SearchResult(id=1, score=0.9), SearchResult(id=2, score=0.3)],
                [SearchResult(id=3, score=0.1), SearchResult(id=4, score=0.2)],
            ]
            response = reader.search(query, top_k=1)

        assert response.results == [SearchResult(id=3, score=0.1)]

    def test_search_dot_product_uses_higher_scores(self) -> None:
        reader = self._build_reader("dot_product")
        query = np.zeros(2, dtype=np.float32)
        with patch("shardyfusion.reader.unified_reader._search_shard") as mock_search:
            mock_search.side_effect = [
                [SearchResult(id=1, score=0.1), SearchResult(id=2, score=0.3)],
                [SearchResult(id=3, score=0.9), SearchResult(id=4, score=0.2)],
            ]
            response = reader.search(query, top_k=2)

        assert [result.id for result in response.results] == [3, 2]

    def test_search_rejects_invalid_metric_before_merge(self) -> None:
        reader = self._build_reader("manhattan")
        query = np.zeros(2, dtype=np.float32)
        with patch("shardyfusion.reader.unified_reader.merge_results") as mock_merge:
            with pytest.raises(
                ConfigValidationError, match="Invalid vector metric 'manhattan'"
            ):
                reader.search(query, top_k=1)
        mock_merge.assert_not_called()
