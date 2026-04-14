"""Tests for vector sharding strategies."""

from __future__ import annotations

import numpy as np
import pytest

from shardyfusion.errors import ConfigValidationError
from shardyfusion.vector.sharding import (
    cluster_assign,
    cluster_assign_batch,
    cluster_probe_shards,
    lsh_assign,
    lsh_assign_batch,
    lsh_generate_hyperplanes,
    lsh_hash,
    lsh_probe_shards,
    route_vector_to_shards,
    train_centroids_kmeans,
)
from shardyfusion.vector.types import DistanceMetric, VectorShardingStrategy


class TestClusterAssign:
    def test_assigns_to_nearest_centroid(self):
        centroids = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        vec = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        assert cluster_assign(vec, centroids, DistanceMetric.L2) == 0

    def test_assigns_batch(self):
        centroids = np.array([[1, 0], [0, 1]], dtype=np.float32)
        vectors = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]], dtype=np.float32)
        assignments = cluster_assign_batch(vectors, centroids, DistanceMetric.L2)
        assert assignments[0] == 0
        assert assignments[1] == 1
        assert len(assignments) == 3

    @pytest.mark.parametrize(
        "metric",
        [
            DistanceMetric.L2,
            DistanceMetric.COSINE,
            DistanceMetric.DOT_PRODUCT,
        ],
    )
    def test_assign_batch_matches_scalar_assignment(self, metric: DistanceMetric):
        rng = np.random.default_rng(42)
        centroids = rng.standard_normal((4, 6)).astype(np.float32)
        vectors = rng.standard_normal((12, 6)).astype(np.float32)

        expected = np.array(
            [cluster_assign(vector, centroids, metric) for vector in vectors],
            dtype=np.int64,
        )
        assignments = cluster_assign_batch(vectors, centroids, metric)

        np.testing.assert_array_equal(assignments, expected)

    def test_probe_shards_returns_correct_count(self):
        rng = np.random.default_rng(42)
        centroids = rng.standard_normal((10, 32)).astype(np.float32)
        query = rng.standard_normal(32).astype(np.float32)
        probes = cluster_probe_shards(query, centroids, 3, DistanceMetric.COSINE)
        assert len(probes) == 3
        assert len(set(probes)) == 3  # all unique

    def test_probe_all_shards(self):
        centroids = np.eye(4, dtype=np.float32)
        query = np.ones(4, dtype=np.float32)
        probes = cluster_probe_shards(query, centroids, 4, DistanceMetric.L2)
        assert sorted(probes) == [0, 1, 2, 3]


class TestTrainCentroids:
    def test_basic_training(self):
        rng = np.random.default_rng(42)
        vectors = rng.standard_normal((200, 16)).astype(np.float32)
        centroids = train_centroids_kmeans(vectors, 4)
        assert centroids.shape == (4, 16)

    def test_too_few_vectors(self):
        vectors = np.zeros((2, 8), dtype=np.float32)
        with pytest.raises(ConfigValidationError, match="at least 5"):
            train_centroids_kmeans(vectors, 5)

    def test_convergence(self):
        # Two well-separated clusters
        rng = np.random.default_rng(42)
        c1 = rng.standard_normal((50, 8)).astype(np.float32) + 10
        c2 = rng.standard_normal((50, 8)).astype(np.float32) - 10
        vectors = np.vstack([c1, c2])
        centroids = train_centroids_kmeans(vectors, 2)
        assert centroids.shape == (2, 8)
        # Centroids should be near +10 and -10
        centroid_means = sorted(centroids.mean(axis=1))
        assert centroid_means[0] < 0
        assert centroid_means[1] > 0


class TestLSH:
    def test_generate_hyperplanes(self):
        hp = lsh_generate_hyperplanes(8, 32)
        assert hp.shape == (8, 32)
        assert hp.dtype == np.float32

    def test_lsh_hash_deterministic(self):
        hp = lsh_generate_hyperplanes(8, 16)
        vec = np.ones(16, dtype=np.float32)
        h1 = lsh_hash(vec, hp)
        h2 = lsh_hash(vec, hp)
        assert h1 == h2

    def test_lsh_assign_in_range(self):
        hp = lsh_generate_hyperplanes(8, 16)
        vec = np.ones(16, dtype=np.float32)
        shard = lsh_assign(vec, hp, 10)
        assert 0 <= shard < 10

    def test_lsh_assign_batch(self):
        hp = lsh_generate_hyperplanes(8, 16)
        vecs = np.ones((10, 16), dtype=np.float32)
        shards = lsh_assign_batch(vecs, hp, 5)
        assert len(shards) == 10
        assert all(0 <= s < 5 for s in shards)

    def test_lsh_probe_shards(self):
        hp = lsh_generate_hyperplanes(8, 16)
        query = np.ones(16, dtype=np.float32)
        probes = lsh_probe_shards(query, hp, 10, 3)
        assert len(probes) <= 3
        assert len(set(probes)) == len(probes)  # unique
        # First probe should be the primary bucket
        primary = lsh_hash(query, hp) % 10
        assert probes[0] == primary

    def test_similar_vectors_same_bucket(self):
        hp = lsh_generate_hyperplanes(8, 32, seed=42)
        vec1 = np.ones(32, dtype=np.float32)
        vec2 = np.ones(32, dtype=np.float32) * 1.01  # very similar
        assert lsh_assign(vec1, hp, 100) == lsh_assign(vec2, hp, 100)


class TestRouteVectorToShards:
    def test_explicit_shard_ids(self):
        query = np.zeros(8, dtype=np.float32)
        result = route_vector_to_shards(
            query,
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=10,
            num_probes=1,
            shard_ids=[2, 5],
        )
        assert result == [2, 5]

    def test_cluster_routing(self):
        centroids = np.eye(4, dtype=np.float32)
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        result = route_vector_to_shards(
            query,
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=4,
            num_probes=2,
            centroids=centroids,
        )
        assert len(result) == 2
        assert result[0] == 0  # nearest centroid

    def test_lsh_routing(self):
        hp = lsh_generate_hyperplanes(8, 16)
        query = np.ones(16, dtype=np.float32)
        result = route_vector_to_shards(
            query,
            strategy=VectorShardingStrategy.LSH,
            num_dbs=10,
            num_probes=3,
            hyperplanes=hp,
        )
        assert len(result) <= 3

    def test_explicit_without_shard_ids_raises(self):
        query = np.zeros(8, dtype=np.float32)
        with pytest.raises(ConfigValidationError, match="shard_ids"):
            route_vector_to_shards(
                query,
                strategy=VectorShardingStrategy.EXPLICIT,
                num_dbs=10,
                num_probes=1,
            )

    def test_cluster_without_centroids_raises(self):
        query = np.zeros(8, dtype=np.float32)
        with pytest.raises(ConfigValidationError, match="centroids"):
            route_vector_to_shards(
                query,
                strategy=VectorShardingStrategy.CLUSTER,
                num_dbs=4,
                num_probes=1,
            )
