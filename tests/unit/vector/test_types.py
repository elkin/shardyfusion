"""Tests for vector type definitions."""

from __future__ import annotations

import numpy as np

from shardyfusion.vector.types import (
    DistanceMetric,
    SearchResult,
    VectorRecord,
    VectorSearchResponse,
    VectorShardDetail,
    VectorShardingStrategy,
    VectorSnapshotInfo,
)


def test_distance_metric_values():
    assert DistanceMetric.COSINE.value == "cosine"
    assert DistanceMetric.L2.value == "l2"
    assert DistanceMetric.DOT_PRODUCT.value == "dot_product"


def test_sharding_strategy_values():
    assert VectorShardingStrategy.CLUSTER.value == "cluster"
    assert VectorShardingStrategy.LSH.value == "lsh"
    assert VectorShardingStrategy.EXPLICIT.value == "explicit"


def test_vector_record():
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    record = VectorRecord(id=42, vector=vec, payload={"tag": "test"})
    assert record.id == 42
    assert np.array_equal(record.vector, vec)
    assert record.payload == {"tag": "test"}
    assert record.shard_id is None


def test_vector_record_explicit_shard():
    vec = np.zeros(3, dtype=np.float32)
    record = VectorRecord(id="abc", vector=vec, shard_id=5)
    assert record.shard_id == 5


def test_search_result_frozen():
    result = SearchResult(id=1, score=0.5)
    assert result.id == 1
    assert result.score == 0.5
    assert result.vector is None
    assert result.payload is None


def test_vector_search_response():
    results = [SearchResult(id=1, score=0.1), SearchResult(id=2, score=0.2)]
    response = VectorSearchResponse(
        results=results, num_shards_queried=3, latency_ms=10.5
    )
    assert len(response.results) == 2
    assert response.num_shards_queried == 3
    assert response.latency_ms == 10.5


def test_vector_shard_detail():
    detail = VectorShardDetail(
        db_id=0, db_url="s3://bucket/shard0", vector_count=1000, checkpoint_id="abc"
    )
    assert detail.db_id == 0
    assert detail.vector_count == 1000


def test_vector_snapshot_info():
    info = VectorSnapshotInfo(
        run_id="run1",
        num_dbs=4,
        dim=128,
        metric=DistanceMetric.COSINE,
        sharding=VectorShardingStrategy.CLUSTER,
        manifest_ref="s3://bucket/manifest",
        total_vectors=10000,
    )
    assert info.num_dbs == 4
    assert info.dim == 128
