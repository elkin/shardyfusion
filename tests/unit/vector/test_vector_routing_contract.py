"""Contract tests verifying the writer-reader vector-sharding invariant.

The core invariant, analogous to the scalar/hash routing contract in
``tests/unit/writer/core/test_routing_contract.py``: for any vector ``v``
and resolved routing ``r``, the shard the **writer** assigns
(``assign_vector_shard``) MUST be the shard the **reader** would query
first for that same vector (``route_vector_to_shards(..., num_probes=1)[0]``),
and MUST be contained in the reader's probe set for any ``num_probes >= 1``.
A violation means a written vector is routed to — and silently not found in —
the wrong shard.

These are Python-only property tests (hypothesis), mirroring the structure of
the hash routing contract module. Centroids/hyperplanes are resolved through
the real ``resolve_vector_routing`` so the resolve/manifest determinism
(seed=42) is exercised on the same path the writer and reader use.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from shardyfusion.errors import ConfigValidationError
from shardyfusion.vector._distributed import (
    ResolvedVectorRouting,
    assign_vector_shard,
    resolve_vector_routing,
)
from shardyfusion.vector.config import (
    VectorIndexConfig,
    VectorShardedWriteConfig,
    VectorShardingSpec,
)
from shardyfusion.vector.sharding import (
    lsh_assign,
    lsh_assign_batch,
    route_vector_to_shards,
)
from shardyfusion.vector.types import DistanceMetric, VectorShardingStrategy

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

seeds = st.integers(min_value=0, max_value=2**32 - 1)
dims = st.integers(min_value=2, max_value=64)
num_dbs_st = st.integers(min_value=2, max_value=64)
hash_bits_st = st.integers(min_value=1, max_value=16)
metrics = st.sampled_from(list(DistanceMetric))
dtypes = st.sampled_from([np.float32, np.float64])


def _vec(seed: int, dim: int, dtype: type) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(dim).astype(dtype)


def _cluster_routing(
    num_dbs: int, dim: int, metric: DistanceMetric, seed: int, dtype: type
) -> ResolvedVectorRouting:
    centroids = np.random.default_rng(seed).standard_normal((num_dbs, dim)).astype(dtype)
    cfg = VectorShardedWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        index_config=VectorIndexConfig(dim=dim, metric=metric),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.CLUSTER, centroids=centroids
        ),
    )
    return resolve_vector_routing(cfg)


def _lsh_routing(
    num_dbs: int, dim: int, num_hash_bits: int
) -> ResolvedVectorRouting:
    cfg = VectorShardedWriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        index_config=VectorIndexConfig(dim=dim),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.LSH, num_hash_bits=num_hash_bits
        ),
    )
    return resolve_vector_routing(cfg)


def _reader_probe(
    routing: ResolvedVectorRouting,
    query: np.ndarray,
    num_probes: int,
    *,
    routing_context: dict | None = None,
    cel_columns: dict | None = None,
) -> list[int]:
    """Invoke the exact reader routing call (mirrors ShardedVectorReader.search)."""
    return route_vector_to_shards(
        query,
        strategy=routing.strategy,
        num_dbs=routing.num_dbs,
        num_probes=num_probes,
        metric=routing.metric,
        centroids=routing.centroids,
        hyperplanes=routing.hyperplanes,
        routing_context=routing_context,
        cel_expr=routing.cel_expr,
        cel_columns=cel_columns,
        routing_values=routing.routing_values,
    )


# ---------------------------------------------------------------------------
# CLUSTER: writer assignment == reader probe[0]
# ---------------------------------------------------------------------------


@given(
    seed=seeds, dim=dims, num_dbs=num_dbs_st, metric=metrics, dtype=dtypes
)
@settings(max_examples=400)
def test_cluster_writer_reader_identity(
    seed: int, dim: int, num_dbs: int, metric: DistanceMetric, dtype: type
) -> None:
    routing = _cluster_routing(num_dbs, dim, metric, seed, dtype)
    query = _vec(seed + 1, dim, dtype)

    writer_shard = assign_vector_shard(vector=query, routing=routing)
    reader_shard = _reader_probe(routing, query, num_probes=1)[0]

    assert writer_shard == reader_shard, (
        f"writer={writer_shard} reader={reader_shard} "
        f"metric={metric} num_dbs={num_dbs} dim={dim}"
    )
    assert 0 <= writer_shard < num_dbs


@given(seed=seeds, dim=dims, num_dbs=num_dbs_st, metric=metrics)
@settings(max_examples=200)
def test_cluster_writer_shard_in_probe_set(
    seed: int, dim: int, num_dbs: int, metric: DistanceMetric
) -> None:
    routing = _cluster_routing(num_dbs, dim, metric, seed, np.float32)
    query = _vec(seed + 1, dim, np.float32)
    writer_shard = assign_vector_shard(vector=query, routing=routing)

    for num_probes in (1, 2, num_dbs):
        probes = _reader_probe(routing, query, num_probes=num_probes)
        assert writer_shard in probes
        assert len(probes) == len(set(probes))
        assert len(probes) <= num_probes
        assert all(0 <= p < num_dbs for p in probes)


def test_cluster_tied_centroids_regression() -> None:
    """Tied minimal distances must not split writer assignment from reader
    probe[0]. Before the stable-sort fix in ``cluster_probe_shards`` the
    writer assigned shard 0 (np.argmin) while the reader's first probe was
    192 (non-stable np.argsort) for this exact input."""
    centroids = np.random.default_rng(123).standard_normal((288, 4)).astype(
        np.float32
    )
    winner = centroids[0].copy()
    for i in (0, 96, 192, 287):
        centroids[i] = winner
    routing = ResolvedVectorRouting(
        strategy=VectorShardingStrategy.CLUSTER,
        num_dbs=288,
        metric=DistanceMetric.L2,
        centroids=centroids,
    )
    query = winner.copy()

    writer_shard = assign_vector_shard(vector=query, routing=routing)
    reader_shard = _reader_probe(routing, query, num_probes=1)[0]

    assert writer_shard == reader_shard == 0


@given(num_dbs=num_dbs_st, dim=dims, seed=seeds)
@settings(max_examples=100)
def test_cluster_zero_vector_cosine_identity(
    num_dbs: int, dim: int, seed: int
) -> None:
    """A zero query under COSINE makes every centroid equidistant (all ties).
    Writer and reader must still agree on the chosen shard."""
    routing = _cluster_routing(num_dbs, dim, DistanceMetric.COSINE, seed, np.float32)
    query = np.zeros(dim, dtype=np.float32)

    writer_shard = assign_vector_shard(vector=query, routing=routing)
    reader_shard = _reader_probe(routing, query, num_probes=1)[0]

    assert writer_shard == reader_shard


# ---------------------------------------------------------------------------
# LSH: writer assignment == reader probe[0] (exact, deterministic)
# ---------------------------------------------------------------------------


@given(seed=seeds, dim=dims, num_dbs=num_dbs_st, num_hash_bits=hash_bits_st)
@settings(max_examples=400)
def test_lsh_writer_reader_identity(
    seed: int, dim: int, num_dbs: int, num_hash_bits: int
) -> None:
    routing = _lsh_routing(num_dbs, dim, num_hash_bits)
    query = _vec(seed, dim, np.float32)

    writer_shard = assign_vector_shard(vector=query, routing=routing)
    reader_shard = _reader_probe(routing, query, num_probes=1)[0]

    assert writer_shard == reader_shard
    assert 0 <= writer_shard < num_dbs


@given(seed=seeds, dim=dims, num_dbs=num_dbs_st, num_hash_bits=hash_bits_st)
@settings(max_examples=200)
def test_lsh_writer_shard_in_probe_set(
    seed: int, dim: int, num_dbs: int, num_hash_bits: int
) -> None:
    routing = _lsh_routing(num_dbs, dim, num_hash_bits)
    query = _vec(seed, dim, np.float32)
    writer_shard = assign_vector_shard(vector=query, routing=routing)

    for num_probes in (1, 2, num_dbs):
        probes = _reader_probe(routing, query, num_probes=num_probes)
        assert writer_shard in probes
        assert len(probes) == len(set(probes))
        assert len(probes) <= num_probes
        assert all(0 <= p < num_dbs for p in probes)


@given(seed=seeds, dim=dims, num_dbs=num_dbs_st, num_hash_bits=hash_bits_st)
@settings(max_examples=200)
def test_lsh_scalar_matches_batch(
    seed: int, dim: int, num_dbs: int, num_hash_bits: int
) -> None:
    """Python writer uses scalar ``lsh_assign``; distributed writers use the
    vectorized ``lsh_assign_batch``. They must agree row-for-row."""
    routing = _lsh_routing(num_dbs, dim, num_hash_bits)
    assert routing.hyperplanes is not None
    vectors = np.random.default_rng(seed).standard_normal((16, dim)).astype(
        np.float32
    )
    batch = lsh_assign_batch(vectors, routing.hyperplanes, num_dbs)
    for i, v in enumerate(vectors):
        assert lsh_assign(v, routing.hyperplanes, num_dbs) == int(batch[i])


# ---------------------------------------------------------------------------
# CEL: writer assignment == reader routing for the same context
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("context", "expected"),
    [({"region": "us"}, 0), ({"region": "eu"}, 1), ({"region": "asia"}, 2)],
)
def test_cel_categorical_writer_reader_identity(
    context: dict, expected: int
) -> None:
    cel_columns = {"region": "string"}
    cfg = VectorShardedWriteConfig(
        num_dbs=3,
        s3_prefix="s3://bucket/prefix",
        index_config=VectorIndexConfig(dim=4),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.CEL,
            cel_expr="region",
            cel_columns=cel_columns,
            routing_values=["us", "eu", "asia"],
        ),
    )
    routing = resolve_vector_routing(cfg)
    query = np.zeros(4, dtype=np.float32)

    writer_shard = assign_vector_shard(
        vector=query, routing=routing, routing_context=context
    )
    reader_shard = _reader_probe(
        routing, query, num_probes=1, routing_context=context,
        cel_columns=cel_columns,
    )[0]

    assert writer_shard == reader_shard == expected


@pytest.mark.parametrize("zone", [0, 3, 7])
def test_cel_direct_integer_writer_reader_identity(zone: int) -> None:
    cel_columns = {"zone": "int"}
    cfg = VectorShardedWriteConfig(
        num_dbs=8,
        s3_prefix="s3://bucket/prefix",
        index_config=VectorIndexConfig(dim=4),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.CEL,
            cel_expr="zone",
            cel_columns=cel_columns,
        ),
    )
    routing = resolve_vector_routing(cfg)
    query = np.zeros(4, dtype=np.float32)
    context = {"zone": zone}

    writer_shard = assign_vector_shard(
        vector=query, routing=routing, routing_context=context
    )
    reader_shard = _reader_probe(
        routing, query, num_probes=1, routing_context=context,
        cel_columns=cel_columns,
    )[0]

    assert writer_shard == reader_shard == zone


# ---------------------------------------------------------------------------
# EXPLICIT: reader has no automatic routing — the contract is "caller supplies
# shard_ids; the writer's explicit assignment is echoed back unchanged".
# ---------------------------------------------------------------------------


def test_explicit_reader_requires_shard_ids() -> None:
    routing = ResolvedVectorRouting(
        strategy=VectorShardingStrategy.EXPLICIT,
        num_dbs=4,
        metric=DistanceMetric.COSINE,
    )
    query = np.zeros(8, dtype=np.float32)

    assert assign_vector_shard(vector=query, routing=routing, shard_id=3) == 3
    with pytest.raises(ConfigValidationError, match="shard_ids"):
        _reader_probe(routing, query, num_probes=1)


# ---------------------------------------------------------------------------
# Resolution determinism — trained centroids must be reproducible so a writer
# and a separately-constructed reader resolve the same routing (seed=42).
# ---------------------------------------------------------------------------


def test_trained_centroids_resolution_is_deterministic() -> None:
    sample = np.random.default_rng(0).standard_normal((500, 16)).astype(np.float32)
    cfg = VectorShardedWriteConfig(
        num_dbs=8,
        s3_prefix="s3://bucket/prefix",
        index_config=VectorIndexConfig(dim=16),
        sharding=VectorShardingSpec(
            strategy=VectorShardingStrategy.CLUSTER, train_centroids=True
        ),
    )
    a = resolve_vector_routing(cfg, sample_vectors=sample)
    b = resolve_vector_routing(cfg, sample_vectors=sample)
    assert a.centroids is not None and b.centroids is not None
    assert np.array_equal(a.centroids, b.centroids)
