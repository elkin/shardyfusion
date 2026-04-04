"""Vector sharding: CLUSTER, LSH, and EXPLICIT strategies."""

from __future__ import annotations

import numpy as np

from ..errors import ConfigValidationError
from .types import DistanceMetric, VectorShardingStrategy


def _l2_distances(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Squared L2 distance from query to each row of targets."""
    diff = targets - query
    return np.sum(diff * diff, axis=1)


def _cosine_distances(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1 - cosine similarity (distance; lower = more similar)."""
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.ones(len(targets), dtype=np.float64)
    target_norms = np.linalg.norm(targets, axis=1)
    dots = targets @ query
    sims = dots / (target_norms * query_norm + 1e-10)
    return 1.0 - sims


def _dot_product_distances(query: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Negative dot product (lower = higher similarity)."""
    return -(targets @ query)


_DISTANCE_FNS = {
    DistanceMetric.L2: _l2_distances,
    DistanceMetric.COSINE: _cosine_distances,
    DistanceMetric.DOT_PRODUCT: _dot_product_distances,
}


# ---------------------------------------------------------------------------
# CLUSTER sharding
# ---------------------------------------------------------------------------


def cluster_assign(
    vector: np.ndarray,
    centroids: np.ndarray,
    metric: DistanceMetric = DistanceMetric.COSINE,
) -> int:
    """Assign a single vector to its nearest centroid. Returns shard_id."""
    dist_fn = _DISTANCE_FNS[metric]
    dists = dist_fn(vector, centroids)
    return int(np.argmin(dists))


def cluster_assign_batch(
    vectors: np.ndarray,
    centroids: np.ndarray,
    metric: DistanceMetric = DistanceMetric.COSINE,
) -> np.ndarray:
    """Assign a batch of vectors to nearest centroids. Returns array of shard_ids."""
    dist_fn = _DISTANCE_FNS[metric]
    # vectors: (N, dim), centroids: (K, dim)
    # Compute distances from each vector to each centroid
    n = len(vectors)
    assignments = np.empty(n, dtype=np.int64)
    for i in range(n):
        dists = dist_fn(vectors[i], centroids)
        assignments[i] = np.argmin(dists)
    return assignments


def cluster_probe_shards(
    query: np.ndarray,
    centroids: np.ndarray,
    num_probes: int,
    metric: DistanceMetric = DistanceMetric.COSINE,
) -> list[int]:
    """Return the top-N nearest centroid shard IDs for a query vector."""
    dist_fn = _DISTANCE_FNS[metric]
    dists = dist_fn(query, centroids)
    indices = np.argsort(dists)[:num_probes]
    return [int(i) for i in indices]


def train_centroids_kmeans(
    vectors: np.ndarray,
    num_clusters: int,
    max_iter: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """Train k-means centroids from a sample of vectors.

    Pure numpy implementation — no sklearn dependency required.
    Returns centroids array of shape (num_clusters, dim).
    """
    rng = np.random.default_rng(seed)
    n, dim = vectors.shape
    if n < num_clusters:
        raise ConfigValidationError(
            f"Need at least {num_clusters} vectors to train {num_clusters} centroids, "
            f"got {n}"
        )

    # K-means++ initialization
    centroids = np.empty((num_clusters, dim), dtype=vectors.dtype)
    centroids[0] = vectors[rng.integers(n)]
    for i in range(1, num_clusters):
        dists = np.min(
            np.sum((vectors[:, None, :] - centroids[None, :i, :]) ** 2, axis=2),
            axis=1,
        )
        probs = dists / dists.sum()
        centroids[i] = vectors[rng.choice(n, p=probs)]

    # Lloyd's algorithm
    for _ in range(max_iter):
        # Assign each vector to nearest centroid
        # Compute in chunks to avoid huge memory for large N
        assignments = np.empty(n, dtype=np.int64)
        chunk_size = 10_000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = vectors[start:end]
            dists = np.sum((chunk[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            assignments[start:end] = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.empty_like(centroids)
        for j in range(num_clusters):
            members = vectors[assignments == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
            else:
                new_centroids[j] = centroids[j]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return centroids


# ---------------------------------------------------------------------------
# LSH sharding
# ---------------------------------------------------------------------------


def lsh_generate_hyperplanes(
    num_hash_bits: int,
    dim: int,
    seed: int = 42,
) -> np.ndarray:
    """Generate random hyperplanes for LSH. Shape: (num_hash_bits, dim)."""
    rng = np.random.default_rng(seed)
    hyperplanes = rng.standard_normal((num_hash_bits, dim)).astype(np.float32)
    return hyperplanes


def lsh_hash(vector: np.ndarray, hyperplanes: np.ndarray) -> int:
    """Compute LSH hash code for a single vector."""
    projections = hyperplanes @ vector
    bits = (projections > 0).astype(np.uint64)
    code = 0
    for i, b in enumerate(bits):
        code |= int(b) << i
    return code


def lsh_assign(
    vector: np.ndarray,
    hyperplanes: np.ndarray,
    num_dbs: int,
) -> int:
    """Assign a vector to a shard via LSH. Returns shard_id."""
    code = lsh_hash(vector, hyperplanes)
    return code % num_dbs


def lsh_assign_batch(
    vectors: np.ndarray,
    hyperplanes: np.ndarray,
    num_dbs: int,
) -> np.ndarray:
    """Assign a batch of vectors to shards via LSH."""
    # projections: (N, num_hash_bits)
    projections = vectors @ hyperplanes.T
    bits = (projections > 0).astype(np.uint64)
    # Compute hash codes: sum of bit << i for each bit position
    powers = np.uint64(1) << np.arange(bits.shape[1], dtype=np.uint64)
    codes = (bits * powers).sum(axis=1).astype(np.int64)
    return codes % num_dbs


def lsh_probe_shards(
    query: np.ndarray,
    hyperplanes: np.ndarray,
    num_dbs: int,
    num_probes: int,
) -> list[int]:
    """Return shard IDs to probe for a query using multi-probe LSH.

    Primary bucket + neighbors found by flipping individual bits.
    """
    code = lsh_hash(query, hyperplanes)
    num_bits = len(hyperplanes)

    shards: list[int] = [code % num_dbs]
    seen = {shards[0]}

    # Flip each bit to find neighboring buckets
    for bit in range(num_bits):
        if len(shards) >= num_probes:
            break
        flipped = code ^ (1 << bit)
        shard = flipped % num_dbs
        if shard not in seen:
            seen.add(shard)
            shards.append(shard)

    return shards[:num_probes]


# ---------------------------------------------------------------------------
# Routing dispatch
# ---------------------------------------------------------------------------


def route_vector_to_shards(
    query: np.ndarray,
    *,
    strategy: VectorShardingStrategy,
    num_dbs: int,
    num_probes: int,
    metric: DistanceMetric = DistanceMetric.COSINE,
    centroids: np.ndarray | None = None,
    hyperplanes: np.ndarray | None = None,
    shard_ids: list[int] | None = None,
) -> list[int]:
    """Determine which shards to query for a given vector.

    Returns a list of shard IDs.
    """
    if shard_ids is not None:
        return shard_ids

    if strategy == VectorShardingStrategy.CLUSTER:
        if centroids is None:
            raise ConfigValidationError("CLUSTER sharding requires centroids")
        return cluster_probe_shards(query, centroids, num_probes, metric)

    if strategy == VectorShardingStrategy.LSH:
        if hyperplanes is None:
            raise ConfigValidationError("LSH sharding requires hyperplanes")
        return lsh_probe_shards(query, hyperplanes, num_dbs, num_probes)

    if strategy == VectorShardingStrategy.EXPLICIT:
        raise ConfigValidationError(
            "EXPLICIT sharding requires shard_ids to be provided at query time"
        )

    raise ConfigValidationError(f"Unknown vector sharding strategy: {strategy}")
