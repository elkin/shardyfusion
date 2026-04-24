"""Tests for AsyncShardedVectorReader using mock adapters."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shardyfusion.manifest import ManifestRef, ParsedManifest
from shardyfusion.vector.async_reader import AsyncShardedVectorReader
from shardyfusion.vector.types import SearchResult

from .test_reader import _make_manifest


class MockAsyncShardReader:
    def __init__(self, shard_id: int, num_results: int = 5) -> None:
        self._shard_id = shard_id
        self._num_results = num_results
        self._closed = False

    async def search(self, query: np.ndarray, top_k: int) -> list[SearchResult]:
        results = []
        for i in range(min(top_k, self._num_results)):
            results.append(
                SearchResult(
                    id=self._shard_id * 100 + i,
                    score=float(self._shard_id) + i * 0.1,
                )
            )
        return results

    async def close(self) -> None:
        self._closed = True


class MockAsyncReaderFactory:
    def __init__(self, num_results: int = 5) -> None:
        self._num_results = num_results
        self.created: dict[str, MockAsyncShardReader] = {}

    async def __call__(
        self, *, db_url: str, local_dir: Path, index_config: Any
    ) -> MockAsyncShardReader:
        shard_id = len(self.created)
        reader = MockAsyncShardReader(shard_id, self._num_results)
        self.created[db_url] = reader
        return reader


class MockAsyncManifestStore:
    def __init__(self, manifest: ParsedManifest) -> None:
        self._manifest = manifest
        self._ref = ManifestRef(
            ref="s3://bucket/manifests/test/manifest",
            run_id="test-run",
            published_at=datetime.now(UTC),
        )

    async def load_current(self) -> ManifestRef | None:
        return self._ref

    async def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifest

    async def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return [self._ref]

    def update(self, manifest: ParsedManifest, run_id: str = "run-v2") -> None:
        self._manifest = manifest
        self._ref = ManifestRef(
            ref=f"s3://bucket/manifests/{run_id}/manifest",
            run_id=run_id,
            published_at=datetime.now(UTC),
        )


class MockAsyncRateLimiter:
    def __init__(self) -> None:
        self.calls = 0

    async def acquire_async(self, tokens: int = 1) -> None:
        self.calls += 1


@pytest.fixture
def tmp_path_str(tmp_path: Path) -> str:
    return str(tmp_path)


class TestAsyncShardedVectorReader:
    async def _make_reader(
        self,
        *,
        num_dbs: int = 4,
        sharding_strategy: str = "explicit",
        tmp_path: Path,
        max_cached_shards: int | None = None,
        max_concurrency: int | None = None,
        rate_limiter: MockAsyncRateLimiter | None = None,
        preload_shards: bool = False,
    ) -> tuple[
        AsyncShardedVectorReader, MockAsyncReaderFactory, MockAsyncManifestStore
    ]:
        manifest = _make_manifest(num_dbs=num_dbs, sharding_strategy=sharding_strategy)
        store = MockAsyncManifestStore(manifest)
        factory = MockAsyncReaderFactory()
        reader = await AsyncShardedVectorReader.open(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
            max_cached_shards=max_cached_shards,
            max_concurrency=max_concurrency,
            rate_limiter=rate_limiter,
            preload_shards=preload_shards,
        )
        return reader, factory, store

    @pytest.mark.asyncio
    async def test_search_explicit(self, tmp_path: Path) -> None:
        reader, factory, _ = await self._make_reader(tmp_path=tmp_path)
        query = np.zeros(32, dtype=np.float32)
        response = await reader.search(query, top_k=5, shard_ids=[0, 1])
        assert response.num_shards_queried == 2
        assert len(response.results) <= 5
        assert response.latency_ms > 0
        await reader.close()

    @pytest.mark.asyncio
    async def test_search_empty_shard_ids(self, tmp_path: Path) -> None:
        reader, factory, _ = await self._make_reader(tmp_path=tmp_path)
        query = np.zeros(32, dtype=np.float32)
        response = await reader.search(query, top_k=5, shard_ids=[])
        assert response.num_shards_queried == 0
        assert len(response.results) == 0
        await reader.close()

    @pytest.mark.asyncio
    async def test_shard_details(self, tmp_path: Path) -> None:
        reader, _, _ = await self._make_reader(num_dbs=3, tmp_path=tmp_path)
        details = reader.shard_details()
        assert len(details) == 3
        assert details[0].db_id == 0
        await reader.close()

    @pytest.mark.asyncio
    async def test_snapshot_info(self, tmp_path: Path) -> None:
        reader, _, _ = await self._make_reader(num_dbs=4, tmp_path=tmp_path)
        info = reader.snapshot_info()
        assert info.num_dbs == 4
        assert info.dim == 32
        await reader.close()

    @pytest.mark.asyncio
    async def test_health_healthy_and_closed(self, tmp_path: Path) -> None:
        reader, _, _ = await self._make_reader(tmp_path=tmp_path)
        health = reader.health()
        assert health.status == "healthy"
        assert not health.is_closed

        await reader.close()
        health_closed = reader.health()
        assert health_closed.status == "unhealthy"
        assert health_closed.is_closed

    @pytest.mark.asyncio
    async def test_lazy_loading(self, tmp_path: Path) -> None:
        reader, factory, _ = await self._make_reader(tmp_path=tmp_path)
        assert len(factory.created) == 0

        await reader.search(np.zeros(32, dtype=np.float32), top_k=3, shard_ids=[0])
        assert len(factory.created) == 1
        await reader.close()

    @pytest.mark.asyncio
    async def test_lru_eviction(self, tmp_path: Path) -> None:
        reader, factory, _ = await self._make_reader(
            num_dbs=4,
            sharding_strategy="explicit",
            tmp_path=tmp_path,
            max_cached_shards=2,
        )
        query = np.zeros(32, dtype=np.float32)
        await reader.search(query, top_k=1, shard_ids=[0])
        await reader.search(query, top_k=1, shard_ids=[1])
        assert len(reader._shard_readers) == 2

        await reader.search(query, top_k=1, shard_ids=[2])
        assert len(reader._shard_readers) == 2

        # yield to event loop to let aclose() finish
        await asyncio.sleep(0.01)
        assert factory.created["s3://bucket/shards/db=00000/attempt=00"]._closed is True

        await reader.close()

    @pytest.mark.asyncio
    async def test_batch_search(self, tmp_path: Path) -> None:
        reader, _, _ = await self._make_reader(tmp_path=tmp_path)
        queries = np.zeros((3, 32), dtype=np.float32)
        responses = await reader.batch_search(queries, top_k=5, shard_ids=[0])
        assert len(responses) == 3
        await reader.close()

    @pytest.mark.asyncio
    async def test_refresh_changed_manifest(self, tmp_path: Path) -> None:
        reader, factory, store = await self._make_reader(num_dbs=2, tmp_path=tmp_path)
        await reader.search(np.zeros(32, dtype=np.float32), top_k=1, shard_ids=[0])

        manifest_v2 = _make_manifest(num_dbs=3)
        store.update(manifest_v2, run_id="run-v2")

        assert await reader.refresh() is True
        info = reader.snapshot_info()
        assert info.num_dbs == 3
        assert info.run_id == "run-v2"
        await reader.close()

    @pytest.mark.asyncio
    async def test_preload_shards(self, tmp_path: Path) -> None:
        reader, factory, _ = await self._make_reader(
            num_dbs=3, tmp_path=tmp_path, preload_shards=True
        )
        assert len(factory.created) == 3
        await reader.close()

    @pytest.mark.asyncio
    async def test_rate_limiter_called_on_search(self, tmp_path: Path) -> None:
        limiter = MockAsyncRateLimiter()
        reader, _, _ = await self._make_reader(tmp_path=tmp_path, rate_limiter=limiter)
        query = np.zeros(32, dtype=np.float32)
        await reader.search(query, top_k=3, shard_ids=[0])
        assert limiter.calls == 1
        await reader.close()
