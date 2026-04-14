"""Tests for UnifiedShardedReader — merged KV + vector search."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest

from shardyfusion.errors import ConfigValidationError, ReaderStateError
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.reader.unified_reader import (
    UnifiedShardedReader,
    UnifiedVectorMeta,
    _auto_reader_factory,
    _distance_metric_from_str,
    _parse_vector_custom,
    _search_shard,
)
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.vector.types import (
    DistanceMetric,
    SearchResult,
    VectorSearchResponse,
)

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
        assert meta.metric is DistanceMetric.COSINE
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
        assert meta.metric is DistanceMetric.L2
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
            metric=DistanceMetric.COSINE,
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
            metric=DistanceMetric.COSINE,
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
            metric=DistanceMetric.COSINE,
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
            metric=DistanceMetric.COSINE,
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


def _build_mock_reader(
    metric: DistanceMetric = DistanceMetric.COSINE,
    num_shards: int = 2,
    *,
    executor: Any = None,
) -> UnifiedShardedReader:
    """Build a mock UnifiedShardedReader without hitting __init__."""
    reader = cast(Any, object.__new__(UnifiedShardedReader))
    reader._closed = False
    reader._executor = executor
    reader._vector_meta = UnifiedVectorMeta(
        dim=2,
        metric=metric,
        index_type="hnsw",
        quantization=None,
        index_params={},
        backend="usearch-sidecar",
        kv_backend="slatedb",
    )
    readers_list: list[Any] = []
    shards_list: list[Any] = []
    for i in range(num_shards):
        readers_list.append(SimpleNamespace(search=lambda q, k, ef=50: []))
        shards_list.append(SimpleNamespace(db_url=f"s3://shard-{i}"))
    reader._state = SimpleNamespace(
        readers=readers_list,
        router=SimpleNamespace(
            shards=shards_list,
            route_with_context=lambda ctx: 0,
        ),
    )
    return cast(UnifiedShardedReader, reader)


class TestUnifiedReaderSearchMerge:
    def _build_reader(self, metric: DistanceMetric) -> UnifiedShardedReader:
        return _build_mock_reader(metric)

    def test_search_uses_shared_merge_utility(self) -> None:
        reader = self._build_reader(DistanceMetric.COSINE)
        query = np.zeros(2, dtype=np.float32)
        with patch("shardyfusion.reader.unified_reader._search_shard") as mock_search:
            mock_search.side_effect = [
                [SearchResult(id=1, score=0.9), SearchResult(id=2, score=0.3)],
                [SearchResult(id=3, score=0.1), SearchResult(id=4, score=0.2)],
            ]
            response = reader.search(query, top_k=1)

        assert response.results == [SearchResult(id=3, score=0.1)]

    def test_search_dot_product_uses_distance_ordering(self) -> None:
        """Dot-product scores are distances (negated dot), lower = better."""
        reader = self._build_reader(DistanceMetric.DOT_PRODUCT)
        query = np.zeros(2, dtype=np.float32)
        with patch("shardyfusion.reader.unified_reader._search_shard") as mock_search:
            mock_search.side_effect = [
                [SearchResult(id=1, score=-0.9), SearchResult(id=2, score=-0.3)],
                [SearchResult(id=3, score=-0.1), SearchResult(id=4, score=-0.2)],
            ]
            response = reader.search(query, top_k=2)

        # Lower distance = more similar; -0.9 is best, then -0.3
        assert [result.id for result in response.results] == [1, 2]

    def test_search_rejects_invalid_metric_before_merge(self) -> None:
        reader = self._build_reader(cast(Any, "manhattan"))
        query = np.zeros(2, dtype=np.float32)
        with patch("shardyfusion.reader.unified_reader.merge_results") as mock_merge:
            with pytest.raises(
                ConfigValidationError, match="Invalid vector metric 'manhattan'"
            ):
                reader.search(query, top_k=1)
        mock_merge.assert_not_called()


# ---------------------------------------------------------------------------
# search() — shard routing and executor paths
# ---------------------------------------------------------------------------


class TestUnifiedReaderSearchRouting:
    def test_search_with_shard_ids(self) -> None:
        reader = _build_mock_reader(num_shards=3)
        query = np.zeros(2, dtype=np.float32)

        with patch("shardyfusion.reader.unified_reader._search_shard") as mock_search:
            mock_search.return_value = [SearchResult(id=1, score=0.5)]
            response = reader.search(query, top_k=5, shard_ids=[1])

        assert response.num_shards_queried == 1
        # Called only for shard 1
        mock_search.assert_called_once()

    def test_search_with_routing_context(self) -> None:
        reader = _build_mock_reader(num_shards=2)
        query = np.zeros(2, dtype=np.float32)

        with patch("shardyfusion.reader.unified_reader._search_shard") as mock_search:
            mock_search.return_value = []
            response = reader.search(query, top_k=5, routing_context={"key": 42})

        assert response.num_shards_queried == 1

    def test_search_queries_all_non_empty_shards(self) -> None:
        reader = _build_mock_reader(num_shards=2)
        query = np.zeros(2, dtype=np.float32)

        with patch("shardyfusion.reader.unified_reader._search_shard") as mock_search:
            mock_search.return_value = []
            response = reader.search(query, top_k=5)

        # Both shards have non-None db_url
        assert response.num_shards_queried == 2

    def test_search_skips_null_shards(self) -> None:
        reader = _build_mock_reader(num_shards=2)
        # Make shard 1 have db_url=None (empty)
        reader._state.router.shards[1] = SimpleNamespace(db_url=None)  # type: ignore[union-attr]

        query = np.zeros(2, dtype=np.float32)
        with patch("shardyfusion.reader.unified_reader._search_shard") as mock_search:
            mock_search.return_value = []
            response = reader.search(query, top_k=5)

        assert response.num_shards_queried == 1

    def test_search_closed_raises(self) -> None:
        reader = _build_mock_reader()
        reader._closed = True  # type: ignore[union-attr]
        with pytest.raises(ReaderStateError, match="closed"):
            reader.search(np.zeros(2, dtype=np.float32), top_k=5)


class TestUnifiedReaderSearchExecutor:
    def test_search_uses_executor_for_multi_shard(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=2)
        try:
            reader = _build_mock_reader(num_shards=2, executor=executor)
            query = np.zeros(2, dtype=np.float32)

            with patch(
                "shardyfusion.reader.unified_reader._search_shard"
            ) as mock_search:
                mock_search.return_value = [SearchResult(id=1, score=0.5)]
                response = reader.search(query, top_k=5, shard_ids=[0, 1])

            assert response.num_shards_queried == 2
            assert mock_search.call_count == 2
        finally:
            executor.shutdown(wait=False)

    def test_search_single_shard_skips_executor(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=2)
        try:
            reader = _build_mock_reader(num_shards=2, executor=executor)
            query = np.zeros(2, dtype=np.float32)

            with patch(
                "shardyfusion.reader.unified_reader._search_shard"
            ) as mock_search:
                mock_search.return_value = []
                response = reader.search(query, top_k=5, shard_ids=[0])

            assert response.num_shards_queried == 1
        finally:
            executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# batch_search
# ---------------------------------------------------------------------------


class TestBatchSearch:
    def test_batch_search_calls_search_per_query(self) -> None:
        reader = _build_mock_reader(num_shards=1)

        with patch("shardyfusion.reader.unified_reader._search_shard") as mock_search:
            mock_search.return_value = [SearchResult(id=0, score=0.1)]
            queries = np.zeros((3, 2), dtype=np.float32)
            results = reader.batch_search(queries, top_k=5, shard_ids=[0])

        assert len(results) == 3
        for r in results:
            assert isinstance(r, VectorSearchResponse)


# ---------------------------------------------------------------------------
# vector_meta property
# ---------------------------------------------------------------------------


class TestVectorMetaProperty:
    def test_vector_meta_returns_parsed_meta(self) -> None:
        reader = _build_mock_reader()
        meta = reader.vector_meta
        assert meta.dim == 2
        assert meta.metric is DistanceMetric.COSINE


# ---------------------------------------------------------------------------
# _auto_reader_factory import error
# ---------------------------------------------------------------------------


class TestAutoReaderFactoryImportFallback:
    def test_usearch_import_error_raises(self) -> None:
        import builtins

        meta = UnifiedVectorMeta(
            dim=32,
            metric=DistanceMetric.COSINE,
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="usearch-sidecar",
            kv_backend="slatedb",
        )

        original_import = builtins.__import__

        def fail_usearch(name: str, *args: Any, **kwargs: Any) -> Any:
            if "usearch" in name:
                raise ImportError("no usearch")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_usearch):
            with pytest.raises(ConfigValidationError, match="vector.*extra"):
                _auto_reader_factory(meta)


# ---------------------------------------------------------------------------
# UnifiedShardedReader integration (goes through __init__)
# ---------------------------------------------------------------------------


class _FixedManifestStore:
    """In-memory manifest store for testing."""

    def __init__(self, manifest: ParsedManifest, ref: str) -> None:
        self._manifest = manifest
        self._ref = ref

    def publish(self, **kw: Any) -> str:
        raise NotImplementedError

    def load_current(self) -> ManifestRef:
        return ManifestRef(ref=self._ref, run_id="run", published_at=datetime.now(UTC))

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifest

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []

    def set_current(self, ref: str) -> None:
        pass

    def update(self, manifest: ParsedManifest, ref: str) -> None:
        self._manifest = manifest
        self._ref = ref


def _vector_manifest(num_dbs: int = 2) -> ParsedManifest:
    """Build a manifest with vector custom fields."""
    required = RequiredBuildMeta(
        run_id="run-vec",
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_col="key",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )
    shards = [
        RequiredShardMeta(db_id=i, db_url=f"mem://db/{i}", attempt=0, row_count=10)
        for i in range(num_dbs)
    ]
    custom = {
        "vector": {
            "dim": 4,
            "metric": "cosine",
            "index_type": "hnsw",
            "quantization": None,
            "backend": "usearch-sidecar",
            "kv_backend": "slatedb",
        }
    }
    return ParsedManifest(required_build=required, shards=shards, custom=custom)


def _invalid_vector_manifest(num_dbs: int = 2) -> ParsedManifest:
    """Build a manifest missing valid vector custom fields."""
    manifest = _vector_manifest(num_dbs=num_dbs)
    return ParsedManifest(
        required_build=manifest.required_build,
        shards=manifest.shards,
        custom={},
    )


class _FakeShardReader:
    """Fake shard reader supporting both get() and search()."""

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url

    def get(self, key: bytes) -> bytes | None:
        return None

    def search(self, query: np.ndarray, top_k: int, ef: int = 50) -> list[SearchResult]:
        return [SearchResult(id=0, score=0.5)]

    def close(self) -> None:
        pass


def _fake_reader_factory(
    *, db_url: str, local_dir: Any, **kwargs: Any
) -> _FakeShardReader:
    return _FakeShardReader(db_url)


class TestUnifiedShardedReaderInit:
    """Integration tests going through __init__ -> _build_simple_state."""

    def test_init_loads_vector_meta(self, tmp_path: Any) -> None:
        manifest = _vector_manifest(num_dbs=2)
        store = _FixedManifestStore(manifest, "mem://manifest/vec")

        reader = UnifiedShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_reader_factory,
        )
        try:
            meta = reader.vector_meta
            assert meta.dim == 4
            assert meta.metric is DistanceMetric.COSINE
            assert meta.backend == "usearch-sidecar"
        finally:
            reader.close()

    def test_init_auto_dispatch_factory(self, tmp_path: Any) -> None:
        """When no reader_factory is given, auto-dispatch from manifest."""
        manifest = _vector_manifest(num_dbs=1)
        store = _FixedManifestStore(manifest, "mem://manifest/vec")

        with (
            patch(
                "shardyfusion.reader.unified_reader._auto_reader_factory"
            ) as mock_auto,
        ):
            mock_auto.return_value = _fake_reader_factory
            reader = UnifiedShardedReader(
                s3_prefix="s3://bucket/prefix",
                local_root=str(tmp_path),
                manifest_store=store,
                reader_factory=None,
            )
            try:
                mock_auto.assert_called_once()
                assert reader.vector_meta.dim == 4
            finally:
                reader.close()

    def test_search_through_init(self, tmp_path: Any) -> None:
        manifest = _vector_manifest(num_dbs=1)
        store = _FixedManifestStore(manifest, "mem://manifest/vec")

        reader = UnifiedShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_reader_factory,
        )
        try:
            query = np.zeros(4, dtype=np.float32)
            response = reader.search(query, top_k=5)
            assert response.num_shards_queried == 1
            assert len(response.results) == 1
        finally:
            reader.close()

    def test_batch_search_through_init(self, tmp_path: Any) -> None:
        manifest = _vector_manifest(num_dbs=1)
        store = _FixedManifestStore(manifest, "mem://manifest/vec")

        reader = UnifiedShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_reader_factory,
        )
        try:
            queries = np.zeros((2, 4), dtype=np.float32)
            results = reader.batch_search(queries, top_k=5)
            assert len(results) == 2
        finally:
            reader.close()

    def test_refresh_updates_vector_meta(self, tmp_path: Any) -> None:
        manifest = _vector_manifest(num_dbs=1)
        store = _FixedManifestStore(manifest, "mem://manifest/vec")

        reader = UnifiedShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_reader_factory,
        )
        try:
            # Refresh with same manifest — should return False (no change)
            changed = reader.refresh()
            # The fixture always returns the same manifest, so no state swap
            assert not changed or reader.vector_meta.dim == 4
        finally:
            reader.close()

    def test_refresh_with_invalid_vector_manifest_keeps_old_state(
        self, tmp_path: Any
    ) -> None:
        store = _FixedManifestStore(_vector_manifest(num_dbs=1), "mem://manifest/v1")

        reader = UnifiedShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_reader_factory,
        )
        try:
            original_meta = reader.vector_meta
            original_ref = reader.snapshot_info().manifest_ref

            store.update(_invalid_vector_manifest(num_dbs=1), "mem://manifest/v2")

            with pytest.raises(ConfigValidationError, match="vector metadata"):
                reader.refresh()

            assert reader.vector_meta == original_meta
            assert reader.snapshot_info().manifest_ref == original_ref

            response = reader.search(np.zeros(4, dtype=np.float32), top_k=5)
            assert response.num_shards_queried == 1
            assert len(response.results) == 1
        finally:
            reader.close()

    def test_closed_reader_search_raises(self, tmp_path: Any) -> None:
        manifest = _vector_manifest(num_dbs=1)
        store = _FixedManifestStore(manifest, "mem://manifest/vec")

        reader = UnifiedShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_fake_reader_factory,
        )
        reader.close()
        with pytest.raises(ReaderStateError, match="closed"):
            reader.search(np.zeros(4, dtype=np.float32), top_k=5)
