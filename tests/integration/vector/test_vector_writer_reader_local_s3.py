"""Integration tests for vector writer + reader against moto S3.

These tests verify the full write→manifest→read round-trip using mock
adapters (no usearch dependency) against a real (moto) S3 service.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.manifest_store import S3ManifestStore
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardingSpec,
    VectorWriteConfig,
)
from shardyfusion.vector.reader import ShardedVectorReader
from shardyfusion.vector.types import (
    DistanceMetric,
    SearchResult,
    VectorRecord,
    VectorShardingStrategy,
)
from shardyfusion.vector.writer import write_vector_sharded

# ---------------------------------------------------------------------------
# Mock adapters — no usearch needed
# ---------------------------------------------------------------------------


@dataclass
class FakeVectorWriter:
    """Writes vectors to a local SQLite file for testing."""

    db_url: str
    local_dir: Path
    dim: int
    _ids: list[int] = field(default_factory=list)
    _vectors: list[np.ndarray] = field(default_factory=list)
    _payloads: list[dict[str, Any]] = field(default_factory=list)
    _closed: bool = False

    def add_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        for i in range(len(ids)):
            self._ids.append(int(ids[i]))
            self._vectors.append(vectors[i])
            self._payloads.append(payloads[i] if payloads else {})

    def flush(self) -> None:
        pass

    def checkpoint(self) -> str | None:
        return f"ckpt-{len(self._ids)}"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Write a simple index file (JSON) to S3-like local path
        self.local_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.local_dir / "index.json"
        data = {
            "ids": self._ids,
            "vectors": [v.tolist() for v in self._vectors],
            "payloads": self._payloads,
            "dim": self.dim,
        }
        index_path.write_text(json.dumps(data))

        # Upload to S3
        from shardyfusion.storage import put_bytes

        put_bytes(
            f"{self.db_url}/index.json",
            index_path.read_bytes(),
            "application/json",
        )

    def __enter__(self) -> FakeVectorWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class FakeWriterFactory:
    def __call__(
        self, *, db_url: str, local_dir: Path, index_config: Any
    ) -> FakeVectorWriter:
        return FakeVectorWriter(
            db_url=db_url,
            local_dir=local_dir,
            dim=index_config.dim,
        )


class FakeShardReader:
    """Reads from JSON index file for testing."""

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
    ) -> None:
        from shardyfusion.storage import get_bytes

        local_dir.mkdir(parents=True, exist_ok=True)
        data = json.loads(get_bytes(f"{db_url}/index.json"))
        self._ids = data["ids"]
        self._vectors = [np.array(v, dtype=np.float32) for v in data["vectors"]]
        self._payloads = data["payloads"]
        self._closed = False

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        ef: int = 50,
    ) -> list[SearchResult]:
        # Brute-force L2 search
        dists = []
        for i, v in enumerate(self._vectors):
            d = float(np.sum((query - v) ** 2))
            dists.append((d, i))
        dists.sort()

        results = []
        for score, idx in dists[:top_k]:
            results.append(
                SearchResult(
                    id=self._ids[idx],
                    score=score,
                    payload=self._payloads[idx] if self._payloads else None,
                )
            )
        return results

    def close(self) -> None:
        self._closed = True


class FakeReaderFactory:
    def __call__(
        self, *, db_url: str, local_dir: Path, index_config: Any
    ) -> FakeShardReader:
        return FakeShardReader(
            db_url=db_url,
            local_dir=local_dir,
            index_config=index_config,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def s3_info(local_s3_service: dict[str, Any]) -> dict[str, Any]:
    """Unpack s3 service info."""
    return local_s3_service


@pytest.fixture
def s3_prefix(s3_info: dict[str, Any]) -> str:
    return f"s3://{s3_info['bucket']}/vector-test"


@pytest.fixture
def cred_provider(s3_info: dict[str, Any]) -> StaticCredentialProvider:
    return StaticCredentialProvider(
        access_key_id=s3_info["access_key_id"],
        secret_access_key=s3_info["secret_access_key"],
    )


@pytest.fixture
def s3_conn_opts(s3_info: dict[str, Any]) -> S3ConnectionOptions:
    return S3ConnectionOptions(
        endpoint_url=s3_info["endpoint_url"],
        region_name=s3_info["region_name"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _make_records(
    rng: np.random.Generator, n: int = 100, dim: int = 32
) -> list[VectorRecord]:
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return [
        VectorRecord(
            id=i,
            vector=vectors[i],
            payload={"label": f"item-{i}"},
        )
        for i in range(n)
    ]


class TestVectorWriterReaderRoundTrip:
    """Full write → publish → read round-trip against moto S3."""

    def test_cluster_sharding_round_trip(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        records = _make_records(rng, n=100, dim=32)

        # Train centroids from data
        centroids = np.array(
            [rng.standard_normal(32).astype(np.float32) for _ in range(4)]
        )

        config = VectorWriteConfig(
            num_dbs=4,
            s3_prefix=s3_prefix,
            index_config=VectorIndexConfig(dim=32, metric=DistanceMetric.L2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.CLUSTER,
                centroids=centroids,
                num_probes=2,
            ),
            adapter_factory=FakeWriterFactory(),
            batch_size=50,
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        )

        # Write
        result = write_vector_sharded(records, config)
        assert result.manifest_ref is not None
        assert result.stats.rows_written == 100

        # Read
        reader = ShardedVectorReader(
            s3_prefix=s3_prefix,
            local_root=str(tmp_path / "reader"),
            reader_factory=FakeReaderFactory(),
            manifest_store=S3ManifestStore(
                s3_prefix,
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
        )

        info = reader.snapshot_info()
        assert info.num_dbs == 4
        assert info.dim == 32
        assert info.total_vectors == 100

        # Search all shards explicitly
        query = rng.standard_normal(32).astype(np.float32)
        response = reader.search(query, top_k=5, shard_ids=list(range(4)))
        assert response.num_shards_queried == 4
        assert len(response.results) == 5
        assert response.latency_ms > 0

        # Verify results are sorted by score (L2 = lower is better)
        scores = [r.score for r in response.results]
        assert scores == sorted(scores)

        reader.close()

    def test_explicit_sharding_round_trip(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        prefix = f"{s3_prefix}/explicit"

        # Create records with explicit shard assignments
        records = []
        for i in range(60):
            records.append(
                VectorRecord(
                    id=i,
                    vector=rng.standard_normal(16).astype(np.float32),
                    payload={"idx": i},
                    shard_id=i % 3,
                )
            )

        config = VectorWriteConfig(
            num_dbs=3,
            s3_prefix=prefix,
            index_config=VectorIndexConfig(dim=16, metric=DistanceMetric.COSINE),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.EXPLICIT,
            ),
            adapter_factory=FakeWriterFactory(),
            batch_size=25,
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        )

        result = write_vector_sharded(records, config)
        assert result.stats.rows_written == 60

        # Read back
        reader = ShardedVectorReader(
            s3_prefix=prefix,
            local_root=str(tmp_path / "reader_explicit"),
            reader_factory=FakeReaderFactory(),
            manifest_store=S3ManifestStore(
                prefix,
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
        )

        details = reader.shard_details()
        assert len(details) == 3
        assert all(d.vector_count == 20 for d in details)

        # Query single shard
        query = rng.standard_normal(16).astype(np.float32)
        response = reader.search(query, top_k=3, shard_ids=[0])
        assert response.num_shards_queried == 1
        assert len(response.results) == 3

        reader.close()

    def test_lsh_sharding_round_trip(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        prefix = f"{s3_prefix}/lsh"
        records = _make_records(rng, n=50, dim=16)

        config = VectorWriteConfig(
            num_dbs=4,
            s3_prefix=prefix,
            index_config=VectorIndexConfig(dim=16, metric=DistanceMetric.L2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.LSH,
                num_hash_bits=6,
                num_probes=2,
            ),
            adapter_factory=FakeWriterFactory(),
            batch_size=20,
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        )

        result = write_vector_sharded(records, config)
        assert result.stats.rows_written == 50
        assert result.manifest_ref is not None

        # Read back — for LSH, reader loads hyperplanes from manifest
        reader = ShardedVectorReader(
            s3_prefix=prefix,
            local_root=str(tmp_path / "reader_lsh"),
            reader_factory=FakeReaderFactory(),
            manifest_store=S3ManifestStore(
                prefix,
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
        )

        info = reader.snapshot_info()
        assert info.num_dbs == 4
        assert info.sharding == VectorShardingStrategy.LSH

        # Search via routing
        query = rng.standard_normal(16).astype(np.float32)
        response = reader.search(query, top_k=5)
        assert response.num_shards_queried == 2  # num_probes=2
        assert len(response.results) <= 5

        reader.close()

    def test_health_and_refresh(
        self,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        prefix = f"{s3_prefix}/health"
        records = _make_records(rng, n=20, dim=8)

        config = VectorWriteConfig(
            num_dbs=2,
            s3_prefix=prefix,
            index_config=VectorIndexConfig(dim=8, metric=DistanceMetric.L2),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.EXPLICIT,
            ),
            adapter_factory=FakeWriterFactory(),
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        )

        # All records go to shard 0
        for r in records:
            r.shard_id = 0

        write_vector_sharded(records, config)

        reader = ShardedVectorReader(
            s3_prefix=prefix,
            local_root=str(tmp_path / "reader_health"),
            reader_factory=FakeReaderFactory(),
            manifest_store=S3ManifestStore(
                prefix,
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
        )

        health = reader.health()
        assert health.status == "healthy"
        assert health.is_closed is False
        assert health.num_shards == 2

        # Refresh with same manifest returns False
        assert reader.refresh() is False

        reader.close()
        health = reader.health()
        assert health.status == "unhealthy"
        assert health.is_closed is True


# ---------------------------------------------------------------------------
# End-to-end round-trip across real backends (USearch + sqlite-vec)
# ---------------------------------------------------------------------------


def _usearch_available() -> bool:
    try:
        import usearch.index  # type: ignore[import-untyped]  # noqa: F401
    except ImportError:
        return False
    return True


def _sqlite_vec_available() -> bool:
    import sqlite3

    try:
        import sqlite_vec  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        conn = sqlite3.connect(":memory:")
        if hasattr(conn, "enable_load_extension"):
            conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except Exception:
        return False
    return True


def _make_real_writer_factory(backend: str) -> Any:
    if backend == "usearch":
        from shardyfusion.vector.adapters.usearch_adapter import USearchWriterFactory

        return USearchWriterFactory()
    if backend == "sqlite-vec":
        from shardyfusion.vector.adapters.sqlite_vec_adapter import (
            SqliteVecVectorWriterFactory,
        )

        return SqliteVecVectorWriterFactory()
    raise ValueError(f"unknown backend: {backend}")


class TestRealBackendRoundTrip:
    """End-to-end round-trip against moto S3 with real USearch and sqlite-vec."""

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param(
                "usearch",
                marks=pytest.mark.skipif(
                    not _usearch_available(), reason="usearch not installed"
                ),
            ),
            pytest.param(
                "sqlite-vec",
                marks=[
                    pytest.mark.vector_sqlite,
                    pytest.mark.skipif(
                        not _sqlite_vec_available(),
                        reason="sqlite-vec not installed",
                    ),
                ],
            ),
        ],
    )
    def test_real_backend_round_trip(
        self,
        backend: str,
        s3_prefix: str,
        cred_provider: StaticCredentialProvider,
        s3_conn_opts: S3ConnectionOptions,
        tmp_path: Path,
    ) -> None:
        rng = np.random.default_rng(42)
        prefix = f"{s3_prefix}/real-{backend}"
        records = _make_records(rng, n=40, dim=8)

        from shardyfusion.config import OutputOptions

        config = VectorWriteConfig(
            num_dbs=2,
            s3_prefix=prefix,
            index_config=VectorIndexConfig(dim=8, metric=DistanceMetric.COSINE),
            sharding=VectorShardingSpec(
                strategy=VectorShardingStrategy.EXPLICIT,
            ),
            output=OutputOptions(local_root=str(tmp_path / f"out-{backend}")),
            adapter_factory=_make_real_writer_factory(backend),
            batch_size=20,
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        )
        # Distribute records across the 2 shards explicitly.
        for r in records:
            r.shard_id = int(r.id) % 2

        result = write_vector_sharded(records, config)
        assert result.manifest_ref is not None
        assert result.stats.rows_written == 40

        # No reader_factory passed → reader must auto-dispatch based on
        # the ``vector.backend`` custom field written by the writer.
        reader = ShardedVectorReader(
            s3_prefix=prefix,
            local_root=str(tmp_path / f"reader-{backend}"),
            manifest_store=S3ManifestStore(
                prefix,
                credential_provider=cred_provider,
                s3_connection_options=s3_conn_opts,
            ),
            credential_provider=cred_provider,
            s3_connection_options=s3_conn_opts,
        )

        # The reader_factory type should match the backend.
        if backend == "usearch":
            from shardyfusion.vector.adapters.usearch_adapter import (
                USearchReaderFactory,
            )

            assert isinstance(reader._reader_factory, USearchReaderFactory)
        else:
            from shardyfusion.vector.adapters.sqlite_vec_adapter import (
                SqliteVecVectorReaderFactory,
            )

            assert isinstance(reader._reader_factory, SqliteVecVectorReaderFactory)

        info = reader.snapshot_info()
        assert info.num_dbs == 2
        assert info.total_vectors == 40

        query = rng.standard_normal(8).astype(np.float32)
        response = reader.search(query, top_k=5, shard_ids=[0, 1])
        assert response.num_shards_queried == 2
        assert len(response.results) >= 1
        # Scores sorted ascending (lower distance = better for all backends).
        scores = [r.score for r in response.results]
        assert scores == sorted(scores)

        reader.close()
