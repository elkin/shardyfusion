"""Tests for AsyncUnifiedShardedReader — async merged KV + vector search."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

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
from shardyfusion.reader.async_reader import _AsyncReaderState, _NullAsyncShardReader
from shardyfusion.reader.async_unified_reader import (
    AsyncUnifiedShardedReader,
    _auto_async_reader_factory,
)
from shardyfusion.reader.unified_reader import (
    UnifiedVectorMeta,
    _parse_vector_custom,
)
from shardyfusion.routing import SnapshotRouter
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.vector.types import DistanceMetric, SearchResult, VectorSearchResponse

# ---------------------------------------------------------------------------
# _auto_async_reader_factory
# ---------------------------------------------------------------------------


class TestAutoAsyncReaderFactory:
    def test_sqlite_vec_backend(self) -> None:
        meta = UnifiedVectorMeta(
            dim=4,
            metric=DistanceMetric.COSINE,
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="sqlite-vec",
            kv_backend="sqlite-vec",
        )
        with patch(
            "shardyfusion.sqlite_vec_adapter.AsyncSqliteVecReaderFactory"
        ) as MockFactory:
            factory = _auto_async_reader_factory(meta)
            MockFactory.assert_called_once()
            assert factory is MockFactory.return_value

    def test_lancedb_backend(self) -> None:
        meta = UnifiedVectorMeta(
            dim=4,
            metric=DistanceMetric.COSINE,
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="lancedb",
            kv_backend="slatedb",
        )
        with patch(
            "shardyfusion.composite_adapter.AsyncCompositeReaderFactory"
        ) as MockComposite:
            with patch(
                "shardyfusion.vector.adapters.lancedb_adapter.AsyncLanceDbReaderFactory"
            ):
                with patch("shardyfusion.storage.create_s3_client"):
                    factory = _auto_async_reader_factory(meta)
                    MockComposite.assert_called_once()
                    assert factory is MockComposite.return_value

    def test_lancedb_backend_with_sqlite_kv(self) -> None:
        meta = UnifiedVectorMeta(
            dim=4,
            metric=DistanceMetric.COSINE,
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="lancedb",
            kv_backend="sqlite",
        )
        with patch(
            "shardyfusion.sqlite_adapter.AsyncSqliteReaderFactory"
        ) as MockSqlite:
            with patch(
                "shardyfusion.composite_adapter.AsyncCompositeReaderFactory"
            ) as MockComposite:
                with patch(
                    "shardyfusion.vector.adapters.lancedb_adapter.AsyncLanceDbReaderFactory"
                ):
                    with patch("shardyfusion.storage.create_s3_client"):
                        factory = _auto_async_reader_factory(meta)
                        MockSqlite.assert_called_once()
                        MockComposite.assert_called_once()
                        assert factory is MockComposite.return_value

    def test_unknown_backend_raises(self) -> None:
        meta = UnifiedVectorMeta(
            dim=4,
            metric=DistanceMetric.COSINE,
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="unknown",
            kv_backend="slatedb",
        )
        with pytest.raises(ConfigValidationError, match="Unknown vector backend"):
            _auto_async_reader_factory(meta)

    def test_lancedb_import_error_raises(self) -> None:
        meta = UnifiedVectorMeta(
            dim=4,
            metric=DistanceMetric.COSINE,
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="lancedb",
            kv_backend="slatedb",
        )

        real_import = __builtins__["__import__"]

        def fail_lancedb(*args: Any, **kwargs: Any) -> Any:
            if args and "lancedb" in str(args[0]):
                raise ImportError("No module named 'lancedb'")
            return real_import(*args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_lancedb):
            with pytest.raises(ConfigValidationError, match="vector.*extra"):
                _auto_async_reader_factory(meta)


# ---------------------------------------------------------------------------
# AsyncUnifiedShardedReader.search helpers
# ---------------------------------------------------------------------------


class _FakeAsyncShardReader:
    """Fake async shard reader supporting get() and search()."""

    def __init__(self, db_url: str) -> None:
        self._db_url = db_url
        self.closed = False

    async def get(self, key: bytes) -> bytes | None:
        return None

    async def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        return [SearchResult(id=0, score=0.5)]

    async def close(self) -> None:
        self.closed = True


class _FakeAsyncShardReaderNoSearch:
    """Fake reader missing search()."""

    async def get(self, key: bytes) -> bytes | None:
        return None

    async def close(self) -> None:
        pass


def _vector_manifest(num_dbs: int = 2, null_db_id: int | None = None) -> ParsedManifest:
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
    shards = []
    for i in range(num_dbs):
        db_url = None if i == null_db_id else f"mem://db/{i}"
        shards.append(
            RequiredShardMeta(db_id=i, db_url=db_url, attempt=0, row_count=10)
        )
    custom = {
        "vector": {
            "dim": 4,
            "metric": "cosine",
            "index_type": "hnsw",
            "quantization": None,
            "unified": True,
            "backend": "lancedb",
            "kv_backend": "slatedb",
        }
    }
    return ParsedManifest(required_build=required, shards=shards, custom=custom)


def _make_async_state(num_dbs: int = 2) -> _AsyncReaderState:
    manifest = _vector_manifest(num_dbs=num_dbs)
    router = SnapshotRouter(manifest.required_build, manifest.shards)
    readers: dict[int, Any] = {
        i: _FakeAsyncShardReader(f"mem://db/{i}") for i in range(num_dbs)
    }
    return _AsyncReaderState(
        manifest_ref="mem://manifest/vec",
        router=router,
        readers=readers,
    )


def _make_async_state_with_null(num_dbs: int = 2) -> _AsyncReaderState:
    manifest = _vector_manifest(num_dbs=num_dbs, null_db_id=1)
    router = SnapshotRouter(manifest.required_build, manifest.shards)
    readers: dict[int, Any] = {}
    for i in range(num_dbs):
        if i == 0:
            readers[i] = _FakeAsyncShardReader(f"mem://db/{i}")
        else:
            readers[i] = _NullAsyncShardReader()
    return _AsyncReaderState(
        manifest_ref="mem://manifest/vec",
        router=router,
        readers=readers,
    )


def _make_reader_with_state(state: _AsyncReaderState) -> AsyncUnifiedShardedReader:
    reader = AsyncUnifiedShardedReader(
        s3_prefix="s3://bucket/prefix",
        local_root="/tmp/async_unified_test",
        reader_factory=_FakeAsyncShardReader,
    )
    reader._state = state
    reader._manifest_custom = _vector_manifest().custom
    reader._vector_meta = _parse_vector_custom(reader._manifest_custom)
    return reader


# ---------------------------------------------------------------------------
# AsyncUnifiedShardedReader.search
# ---------------------------------------------------------------------------


class TestAsyncUnifiedShardedReaderSearch:
    @pytest.mark.asyncio
    async def test_search_with_shard_ids(self) -> None:
        reader = _make_reader_with_state(_make_async_state(num_dbs=2))
        query = np.zeros(4, dtype=np.float32)
        response = await reader.search(query, top_k=5, shard_ids=[0])
        assert response.num_shards_queried == 1
        assert len(response.results) == 1
        assert response.latency_ms > 0

    @pytest.mark.asyncio
    async def test_search_with_routing_context(self) -> None:
        state = _make_async_state(num_dbs=2)
        reader = _make_reader_with_state(state)
        query = np.zeros(4, dtype=np.float32)

        with patch.object(state.router, "route_with_context", return_value=0):
            response = await reader.search(query, top_k=5, routing_context={"key": 42})
            assert response.num_shards_queried == 1
            assert len(response.results) == 1

    @pytest.mark.asyncio
    async def test_search_all_non_empty_shards(self) -> None:
        state = _make_async_state_with_null(num_dbs=2)
        reader = _make_reader_with_state(state)
        query = np.zeros(4, dtype=np.float32)
        response = await reader.search(query, top_k=5)
        assert response.num_shards_queried == 1

    @pytest.mark.asyncio
    async def test_search_rate_limiter_called(self) -> None:
        class RecordingLimiter:
            def __init__(self) -> None:
                self.calls: list[int] = []

            async def acquire_async(self, tokens: int = 1) -> None:
                self.calls.append(tokens)

            def acquire(self, tokens: int = 1) -> None:
                pass

            def try_acquire(self, tokens: int = 1) -> Any:
                return None

        limiter = RecordingLimiter()
        reader = _make_reader_with_state(_make_async_state(num_dbs=1))
        reader._rate_limiter = limiter
        query = np.zeros(4, dtype=np.float32)
        await reader.search(query, top_k=5, shard_ids=[0])
        assert limiter.calls == [1]

    @pytest.mark.asyncio
    async def test_search_null_shard_returns_empty(self) -> None:
        state = _make_async_state_with_null(num_dbs=2)
        reader = _make_reader_with_state(state)
        query = np.zeros(4, dtype=np.float32)
        response = await reader.search(query, top_k=5, shard_ids=[1])
        assert response.num_shards_queried == 1
        assert len(response.results) == 0

    @pytest.mark.asyncio
    async def test_search_shard_without_search_raises(self) -> None:
        state = _make_async_state(num_dbs=1)
        state.readers[0] = _FakeAsyncShardReaderNoSearch()
        reader = _make_reader_with_state(state)
        query = np.zeros(4, dtype=np.float32)
        with pytest.raises(ReaderStateError, match="does not support vector search"):
            await reader.search(query, top_k=5, shard_ids=[0])

    @pytest.mark.asyncio
    async def test_search_max_concurrency_semaphore(self) -> None:
        state = _make_async_state(num_dbs=3)
        reader = _make_reader_with_state(state)
        reader._max_concurrency = 2
        query = np.zeros(4, dtype=np.float32)
        response = await reader.search(query, top_k=5, shard_ids=[0, 1, 2])
        assert response.num_shards_queried == 3

    @pytest.mark.asyncio
    async def test_search_metric_string_coerced(self) -> None:
        state = _make_async_state(num_dbs=1)
        reader = _make_reader_with_state(state)
        reader._vector_meta = UnifiedVectorMeta(
            dim=4,
            metric="l2",
            index_type="hnsw",
            quantization=None,
            index_params={},
            backend="lancedb",
            kv_backend="slatedb",
        )
        query = np.zeros(4, dtype=np.float32)
        response = await reader.search(query, top_k=5, shard_ids=[0])
        assert response.num_shards_queried == 1

    @pytest.mark.asyncio
    async def test_batch_search_iterates_queries(self) -> None:
        reader = _make_reader_with_state(_make_async_state(num_dbs=1))
        queries = np.zeros((3, 4), dtype=np.float32)
        responses = await reader.batch_search(queries, top_k=5, shard_ids=[0])
        assert len(responses) == 3
        for r in responses:
            assert isinstance(r, VectorSearchResponse)


# ---------------------------------------------------------------------------
# AsyncUnifiedShardedReader state / lifecycle
# ---------------------------------------------------------------------------


class TestAsyncUnifiedShardedReaderState:
    @pytest.mark.asyncio
    async def test_vector_meta_before_init_raises(self) -> None:
        reader = AsyncUnifiedShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root="/tmp/async_unified_test",
        )
        with pytest.raises(ReaderStateError, match="not yet initialized"):
            _ = reader.vector_meta

    @pytest.mark.asyncio
    async def test_build_state_restores_factory_on_exception(self) -> None:
        manifest = _vector_manifest(num_dbs=1)
        original_factory = MagicMock()
        reader = AsyncUnifiedShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root="/tmp/async_unified_test",
            reader_factory=original_factory,
        )
        reader._reader_factory = original_factory
        reader._user_reader_factory = None

        # Patch super()._build_state to raise
        with patch(
            "shardyfusion.reader.async_reader.AsyncShardedReader._build_state",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await reader._build_state("ref", manifest)

        assert reader._reader_factory is original_factory

    @pytest.mark.asyncio
    async def test_build_state_sets_manifest_custom(self) -> None:
        manifest = _vector_manifest(num_dbs=1)
        reader = AsyncUnifiedShardedReader(
            s3_prefix="s3://bucket/prefix",
            local_root="/tmp/async_unified_test",
            reader_factory=_FakeAsyncShardReader,
        )

        # Patch super()._build_state to return a dummy state
        dummy_state = _make_async_state(num_dbs=1)
        with patch(
            "shardyfusion.reader.async_reader.AsyncShardedReader._build_state",
            return_value=dummy_state,
        ):
            result = await reader._build_state("ref", manifest)

        assert result is dummy_state
        assert reader._manifest_custom == manifest.custom
        assert reader._vector_meta is not None
        assert reader._vector_meta.dim == 4


# ---------------------------------------------------------------------------
# AsyncUnifiedShardedReader through open()
# ---------------------------------------------------------------------------


class _AsyncFixedManifestStore:
    """In-memory async manifest store for testing."""

    def __init__(self, manifest: ParsedManifest, ref: str) -> None:
        self._manifest = manifest
        self._ref = ref

    async def load_current(self) -> ManifestRef | None:
        return ManifestRef(ref=self._ref, run_id="run", published_at=datetime.now(UTC))

    async def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifest

    async def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []


@pytest.mark.asyncio
async def test_open_initializes_reader(tmp_path: Any) -> None:
    manifest = _vector_manifest(num_dbs=1)
    store = _AsyncFixedManifestStore(manifest, "mem://manifest/vec")

    # Patch super()._build_state so we don't need real adapters
    dummy_state = _make_async_state(num_dbs=1)
    with patch(
        "shardyfusion.reader.async_reader.AsyncShardedReader._build_state",
        return_value=dummy_state,
    ):
        reader = await AsyncUnifiedShardedReader.open(
            s3_prefix="s3://bucket/prefix",
            local_root=str(tmp_path),
            manifest_store=store,
            reader_factory=_FakeAsyncShardReader,
        )
        try:
            meta = reader.vector_meta
            assert meta.dim == 4
            assert meta.backend == "lancedb"
        finally:
            await reader.close()
