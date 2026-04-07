"""Tests for ShardedVectorReader using mock adapters."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta
from io import BytesIO
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
    url_tag: str = "default",
) -> ParsedManifest:
    """Build a ParsedManifest with vector metadata in custom fields."""
    shards = []
    for db_id in range(num_dbs):
        shards.append(
            RequiredShardMeta(
                db_id=db_id,
                db_url=f"s3://bucket/{url_tag}/shards/db={db_id:05d}/attempt=00",
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

    def update(self, manifest: ParsedManifest, run_id: str = "run-v2") -> None:
        """Swap in a new manifest (for refresh tests)."""
        self._manifest = manifest
        self._ref = ManifestRef(
            ref=f"s3://bucket/manifests/{run_id}/manifest",
            run_id=run_id,
            published_at=datetime.now(UTC),
        )


class MockRateLimiter:
    """Tracks acquire() calls."""

    def __init__(self) -> None:
        self.calls = 0

    def acquire(self, tokens: int = 1) -> None:
        self.calls += 1

    def try_acquire(self, tokens: int = 1) -> object:
        self.calls += 1
        return True

    async def acquire_async(self, tokens: int = 1) -> None:
        self.calls += 1


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

    def test_lru_eviction_cleans_up_shard_locks(self, tmp_path: Path):
        """Evicted shards remove lock entries so lock map does not grow forever."""
        manifest = _make_manifest(num_dbs=8, sharding_strategy="explicit")
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

        for shard_id in [0, 1, 2, 3, 4, 5, 6, 7]:
            reader.search(query, top_k=1, shard_ids=[shard_id])
            assert len(reader._shard_locks) <= 2
            assert len(reader._shard_readers) <= 2

        # Rotate back through earlier shards; lock map should remain bounded.
        for shard_id in [0, 1, 2, 3]:
            reader.search(query, top_k=1, shard_ids=[shard_id])
            assert len(reader._shard_locks) <= 2
            assert len(reader._shard_readers) <= 2

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

    def test_refresh_same_ref(self, tmp_path: Path):
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

    def test_refresh_changed_manifest(self, tmp_path: Path):
        """refresh() swaps state when manifest ref changes."""
        manifest_v1 = _make_manifest(num_dbs=2)
        store = MockManifestStore(manifest_v1)
        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
        )

        # Load a shard under v1
        reader.search(np.zeros(32, dtype=np.float32), top_k=1, shard_ids=[0])
        assert len(factory.created) == 1

        # Update store to a new manifest
        manifest_v2 = _make_manifest(num_dbs=3)
        store.update(manifest_v2, run_id="run-v2")

        assert reader.refresh() is True

        # Old shard readers should be closed and cache cleared
        info = reader.snapshot_info()
        assert info.num_dbs == 3
        assert info.run_id == "run-v2"

        reader.close()

    def test_refresh_closes_old_readers(self, tmp_path: Path):
        """Old shard readers are closed on refresh."""
        manifest = _make_manifest(num_dbs=2)
        store = MockManifestStore(manifest)
        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
        )

        # Load shard 0
        reader.search(np.zeros(32, dtype=np.float32), top_k=1, shard_ids=[0])
        old_shard_reader = list(factory.created.values())[0]
        assert not old_shard_reader._closed

        # Swap manifest
        store.update(_make_manifest(num_dbs=2), run_id="run-v2")
        reader.refresh()

        # Old reader should be closed
        assert old_shard_reader._closed
        reader.close()

    def test_refresh_clears_shard_locks(self, tmp_path: Path):
        manifest = _make_manifest(num_dbs=3)
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
        assert reader._shard_locks

        store.update(_make_manifest(num_dbs=3), run_id="run-v2")
        assert reader.refresh() is True
        assert reader._shard_locks == {}
        reader.close()

    def test_close_clears_shard_locks(self, tmp_path: Path):
        reader, _ = self._make_reader(tmp_path=tmp_path)
        query = np.zeros(32, dtype=np.float32)
        reader.search(query, top_k=1, shard_ids=[0])
        assert reader._shard_locks

        reader.close()
        assert reader._shard_locks == {}

    def test_refresh_does_not_reuse_old_manifest_reader(self, tmp_path: Path):
        """A post-refresh search should load a reader built for the new manifest."""
        manifest_v1 = _make_manifest(
            num_dbs=1,
            sharding_strategy="explicit",
            url_tag="v1",
        )
        store = MockManifestStore(manifest_v1)
        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
        )
        query = np.zeros(32, dtype=np.float32)

        reader.search(query, top_k=1, shard_ids=[0])
        old_reader = list(factory.created.values())[0]
        assert len(factory.created) == 1

        store.update(
            _make_manifest(num_dbs=1, sharding_strategy="explicit", url_tag="v2"),
            run_id="run-v2",
        )
        assert reader.refresh() is True

        reader.search(query, top_k=1, shard_ids=[0])
        assert len(factory.created) == 2
        assert old_reader._closed
        assert "s3://bucket/v2/shards/db=00000/attempt=00" in factory.created
        reader.close()

    def test_thread_pool_fan_out(self, tmp_path: Path):
        """Multi-threaded search with max_workers."""
        manifest = _make_manifest(num_dbs=4, sharding_strategy="explicit")
        store = MockManifestStore(manifest)
        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
            max_workers=3,
        )

        query = np.zeros(32, dtype=np.float32)
        response = reader.search(query, top_k=5, shard_ids=[0, 1, 2])
        assert response.num_shards_queried == 3
        assert len(response.results) == 5
        # All 3 shards should have been loaded
        assert len(factory.created) == 3
        reader.close()

    def test_thread_pool_concurrent_searches(self, tmp_path: Path):
        """Concurrent searches via thread pool don't corrupt state."""
        manifest = _make_manifest(num_dbs=4, sharding_strategy="explicit")
        store = MockManifestStore(manifest)
        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
            max_workers=2,
        )

        errors: list[Exception] = []

        def search_worker() -> None:
            try:
                for _ in range(10):
                    query = np.zeros(32, dtype=np.float32)
                    response = reader.search(query, top_k=3, shard_ids=[0, 1])
                    assert response.num_shards_queried == 2
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=search_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent search errors: {errors}"
        reader.close()

    def test_preload_shards(self, tmp_path: Path):
        """preload_shards=True loads all shards at construction time."""
        manifest = _make_manifest(num_dbs=3, sharding_strategy="explicit")
        store = MockManifestStore(manifest)
        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
            preload_shards=True,
        )

        # All 3 shards should already be loaded
        assert len(factory.created) == 3
        assert len(reader._shard_readers) == 3
        reader.close()

    def test_rate_limiter_called_on_search(self, tmp_path: Path):
        """Rate limiter acquire() is called for each search."""
        manifest = _make_manifest(num_dbs=2, sharding_strategy="explicit")
        store = MockManifestStore(manifest)
        factory = MockReaderFactory()
        limiter = MockRateLimiter()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
            rate_limiter=limiter,
        )

        query = np.zeros(32, dtype=np.float32)
        reader.search(query, top_k=3, shard_ids=[0])
        assert limiter.calls == 1

        reader.search(query, top_k=3, shard_ids=[0, 1])
        assert limiter.calls == 2

        reader.close()

    def test_health_staleness_threshold(self, tmp_path: Path):
        """health() returns degraded when manifest is stale."""
        manifest = _make_manifest(num_dbs=2)
        store = MockManifestStore(manifest)
        # Backdate the manifest ref
        store._ref = ManifestRef(
            ref="s3://bucket/manifests/old/manifest",
            run_id="old-run",
            published_at=datetime(2020, 1, 1, tzinfo=UTC),
        )
        factory = MockReaderFactory()
        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
        )

        health = reader.health(staleness_threshold=timedelta(hours=1))
        assert health.status == "degraded"
        assert health.manifest_age_seconds is not None
        assert health.manifest_age_seconds > 3600
        reader.close()

    def test_refresh_search_consistency(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Concurrent refresh/search sees coherent old or new state snapshots."""

        def _npy_bytes(array: np.ndarray) -> bytes:
            buffer = BytesIO()
            np.save(buffer, array)
            return buffer.getvalue()

        manifest_v1 = _make_manifest(
            num_dbs=1,
            sharding_strategy="cluster",
            centroids_ref="s3://bucket/centroids/v1.npy",
        )
        manifest_v2 = _make_manifest(
            num_dbs=2,
            sharding_strategy="cluster",
            centroids_ref="s3://bucket/centroids/v2.npy",
        )
        store = MockManifestStore(manifest_v1)
        factory = MockReaderFactory()

        centroids_payload = {
            "s3://bucket/centroids/v1.npy": _npy_bytes(
                np.array([[0.0, 0.0]], dtype=np.float32)
            ),
            "s3://bucket/centroids/v2.npy": _npy_bytes(
                np.array([[10.0, 10.0], [1.0, 1.0]], dtype=np.float32)
            ),
        }

        import shardyfusion.vector.reader as reader_module

        monkeypatch.setattr(
            reader_module,
            "get_bytes",
            lambda ref, *, s3_client: centroids_payload[ref],
        )

        reader = ShardedVectorReader(
            s3_prefix="s3://bucket",
            local_root=str(tmp_path),
            reader_factory=factory,
            manifest_store=store,
        )
        query = np.array([1.0, 1.0], dtype=np.float32)

        stop = threading.Event()
        failures: list[str] = []

        def search_worker() -> None:
            while not stop.is_set():
                response = reader.search(query, top_k=1)
                info = reader.snapshot_info()
                if response.num_shards_queried not in (1, 2):
                    failures.append(
                        "expected queried shards to be 1 or 2, got "
                        f"{response.num_shards_queried}"
                    )
                    break
                if info.num_dbs not in (1, 2):
                    failures.append(f"unexpected num_dbs={info.num_dbs}")
                    break

        worker = threading.Thread(target=search_worker)
        worker.start()
        time.sleep(0.02)
        store.update(manifest_v2, run_id="run-v2")
        assert reader.refresh() is True
        time.sleep(0.02)
        stop.set()
        worker.join()

        assert failures == []
        reader.close()
