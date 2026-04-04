"""Tests for ShardedVectorReader using mock adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shardyfusion.errors import ConfigValidationError, ReaderStateError
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
    WriterInfo,
)
from shardyfusion.sharding_types import KeyEncoding, ShardingStrategy
from shardyfusion.vector.reader import ShardedVectorReader
from shardyfusion.vector.types import (
    SearchResult,
)

# ---------------------------------------------------------------------------
# Mock infrastructure
# ---------------------------------------------------------------------------


class MockShardReader:
    """Mock per-shard reader that returns deterministic results."""

    def __init__(self, shard_id: int, num_results: int = 5) -> None:
        self._shard_id = shard_id
        self._num_results = num_results
        self._closed = False

    def search(self, query: np.ndarray, top_k: int, ef: int = 50) -> list[SearchResult]:
        results = []
        for i in range(min(top_k, self._num_results)):
            results.append(
                SearchResult(
                    id=self._shard_id * 100 + i,
                    score=float(self._shard_id) + i * 0.1,
                )
            )
        return results

    def close(self) -> None:
        self._closed = True


class MockReaderFactory:
    """Factory that creates MockShardReaders."""

    def __init__(self, num_results: int = 5) -> None:
        self._num_results = num_results
        self.created: dict[str, MockShardReader] = {}

    def __call__(
        self, *, db_url: str, local_dir: Path, index_config: Any
    ) -> MockShardReader:
        # Extract shard_id from db_url pattern
        shard_id = len(self.created)
        reader = MockShardReader(shard_id, self._num_results)
        self.created[db_url] = reader
        return reader


def _make_manifest(
    *,
    num_dbs: int = 4,
    sharding_strategy: str = "cluster",
    dim: int = 32,
    centroids_ref: str | None = None,
    hyperplanes_ref: str | None = None,
) -> ParsedManifest:
    """Build a ParsedManifest with vector metadata in custom fields."""
    shards = []
    for db_id in range(num_dbs):
        shards.append(
            RequiredShardMeta(
                db_id=db_id,
                db_url=f"s3://bucket/shards/db={db_id:05d}/attempt=00",
                attempt=0,
                row_count=100,
                writer_info=WriterInfo(),
            )
        )

    vector_custom: dict[str, Any] = {
        "dim": dim,
        "metric": "cosine",
        "index_type": "hnsw",
        "quantization": None,
        "total_vectors": num_dbs * 100,
        "sharding_strategy": sharding_strategy,
        "num_probes": 2,
    }
    if centroids_ref:
        vector_custom["centroids_ref"] = centroids_ref
    if hyperplanes_ref:
        vector_custom["hyperplanes_ref"] = hyperplanes_ref

    return ParsedManifest(
        required_build=RequiredBuildMeta(
            run_id="test-run",
            created_at=datetime.now(UTC),
            num_dbs=num_dbs,
            s3_prefix="s3://bucket",
            key_col="_vector_id",
            sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
            db_path_template="db={db_id:05d}",
            shard_prefix="shards",
            key_encoding=KeyEncoding.RAW,
        ),
        shards=shards,
        custom={"vector": vector_custom},
    )


class MockManifestStore:
    """Minimal ManifestStore mock for reader tests."""

    def __init__(self, manifest: ParsedManifest) -> None:
        self._manifest = manifest
        self._ref = ManifestRef(
            ref="s3://bucket/manifests/test/manifest",
            run_id="test-run",
            published_at=datetime.now(UTC),
        )

    def load_current(self) -> ManifestRef | None:
        return self._ref

    def load_manifest(self, ref: str) -> ParsedManifest:
        return self._manifest

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return [self._ref]

    def publish(self, **kwargs: Any) -> str:
        return self._ref.ref

    def set_current(self, ref: str) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestShardedVectorReader:
    def _make_reader(
        self,
        *,
        num_dbs: int = 4,
        sharding_strategy: str = "explicit",
        tmp_path: Path,
    ) -> tuple[ShardedVectorReader, MockReaderFactory]:
        manifest = _make_manifest(num_dbs=num_dbs, sharding_strategy=sharding_strategy)
        store = MockManifestStore(manifest)
        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
        )
        return reader, factory

    def test_search_explicit(self, tmp_path: Path):
        reader, factory = self._make_reader(tmp_path=tmp_path)
        query = np.zeros(32, dtype=np.float32)
        response = reader.search(query, top_k=5, shard_ids=[0, 1])
        assert response.num_shards_queried == 2
        assert len(response.results) <= 5
        assert response.latency_ms > 0
        reader.close()

    def test_search_single_shard(self, tmp_path: Path):
        reader, _ = self._make_reader(tmp_path=tmp_path)
        query = np.zeros(32, dtype=np.float32)
        response = reader.search(query, top_k=3, shard_ids=[0])
        assert response.num_shards_queried == 1
        assert len(response.results) <= 3
        reader.close()

    def test_search_empty_shard_ids(self, tmp_path: Path):
        reader, _ = self._make_reader(tmp_path=tmp_path)
        query = np.zeros(32, dtype=np.float32)
        response = reader.search(query, top_k=5, shard_ids=[])
        assert response.num_shards_queried == 0
        assert len(response.results) == 0
        reader.close()

    def test_shard_details(self, tmp_path: Path):
        reader, _ = self._make_reader(num_dbs=3, tmp_path=tmp_path)
        details = reader.shard_details()
        assert len(details) == 3
        assert details[0].db_id == 0
        assert details[0].vector_count == 100
        reader.close()

    def test_snapshot_info(self, tmp_path: Path):
        reader, _ = self._make_reader(num_dbs=4, tmp_path=tmp_path)
        info = reader.snapshot_info()
        assert info.num_dbs == 4
        assert info.dim == 32
        assert info.total_vectors == 400
        reader.close()

    def test_shard_for_id(self, tmp_path: Path):
        reader, _ = self._make_reader(num_dbs=4, tmp_path=tmp_path)
        detail = reader.shard_for_id(0)
        assert detail.db_id == 0
        assert detail.vector_count == 100
        assert detail.db_url is not None

        # Non-existent shard returns empty detail
        missing = reader.shard_for_id(99)
        assert missing.db_url is None
        assert missing.vector_count == 0
        reader.close()

    def test_health_healthy(self, tmp_path: Path):
        reader, _ = self._make_reader(tmp_path=tmp_path)
        health = reader.health()
        assert health.status == "healthy"
        assert health.is_closed is False
        assert health.num_shards == 4
        reader.close()

    def test_health_closed(self, tmp_path: Path):
        reader, _ = self._make_reader(tmp_path=tmp_path)
        reader.close()
        health = reader.health()
        assert health.status == "unhealthy"
        assert health.is_closed is True

    def test_closed_reader_raises(self, tmp_path: Path):
        reader, _ = self._make_reader(tmp_path=tmp_path)
        reader.close()
        with pytest.raises(ReaderStateError):
            reader.search(np.zeros(32, dtype=np.float32), shard_ids=[0])

    def test_context_manager(self, tmp_path: Path):
        manifest = _make_manifest()
        store = MockManifestStore(manifest)
        factory = MockReaderFactory()
        with ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
        ) as reader:
            response = reader.search(
                np.zeros(32, dtype=np.float32), top_k=3, shard_ids=[0]
            )
            assert response.num_shards_queried == 1

    def test_lazy_loading(self, tmp_path: Path):
        """Shards are only loaded when first queried."""
        reader, factory = self._make_reader(tmp_path=tmp_path)
        assert len(factory.created) == 0  # nothing loaded yet

        reader.search(np.zeros(32, dtype=np.float32), top_k=3, shard_ids=[0])
        assert len(factory.created) == 1  # only shard 0 loaded

        reader.search(np.zeros(32, dtype=np.float32), top_k=3, shard_ids=[1])
        assert len(factory.created) == 2  # shard 1 now loaded too
        reader.close()

    def test_lru_eviction(self, tmp_path: Path):
        """max_cached_shards triggers LRU eviction."""
        manifest = _make_manifest(num_dbs=4, sharding_strategy="explicit")
        store = MockManifestStore(manifest)
        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
            max_cached_shards=2,
        )
        query = np.zeros(32, dtype=np.float32)

        reader.search(query, top_k=1, shard_ids=[0])
        reader.search(query, top_k=1, shard_ids=[1])
        assert len(reader._shard_readers) == 2

        reader.search(query, top_k=1, shard_ids=[2])
        assert len(reader._shard_readers) == 2  # evicted oldest
        reader.close()

    def test_batch_search(self, tmp_path: Path):
        reader, _ = self._make_reader(tmp_path=tmp_path)
        queries = np.zeros((3, 32), dtype=np.float32)
        responses = reader.batch_search(queries, top_k=5, shard_ids=[0])
        assert len(responses) == 3
        for r in responses:
            assert r.num_shards_queried == 1
        reader.close()

    def test_route_vector_explicit_raises(self, tmp_path: Path):
        reader, _ = self._make_reader(sharding_strategy="explicit", tmp_path=tmp_path)
        query = np.zeros(32, dtype=np.float32)
        with pytest.raises(ConfigValidationError, match="shard_ids"):
            reader.route_vector(query)
        reader.close()

    def test_refresh(self, tmp_path: Path):
        manifest = _make_manifest(num_dbs=2)
        store = MockManifestStore(manifest)
        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
        )
        # Same ref -> no change
        assert reader.refresh() is False
        reader.close()
