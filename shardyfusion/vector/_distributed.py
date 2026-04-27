"""Shared vector write core for Python and distributed (Spark/Dask/Ray) writers."""

from __future__ import annotations

import io
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .._rate_limiter import TokenBucket
from ..errors import ConfigValidationError
from ..logging import get_logger, log_event
from ..manifest import (
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
    WriterInfo,
)
from ..manifest_store import S3ManifestStore
from ..metrics._events import MetricEvent
from ..metrics._protocol import MetricsCollector
from ..sharding_types import KeyEncoding, ShardHashAlgorithm, ShardingStrategy
from ..storage import put_bytes
from .config import VectorIndexConfig, VectorWriteConfig
from .sharding import (
    cluster_assign,
    lsh_assign,
    lsh_generate_hyperplanes,
    train_centroids_kmeans,
)
from .types import (
    DistanceMetric,
    VectorIndexWriterFactory,
    VectorShardingStrategy,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class VectorShardState:
    """Per-shard buffer for vector writes."""

    adapter: Any | None = None  # VectorIndexWriter
    db_url: str = ""
    ids: list[int | str] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)
    payloads: list[dict[str, Any] | None] = field(default_factory=list)
    row_count: int = 0
    checkpoint_id: str | None = None
    db_bytes: int = 0


@dataclass(frozen=True, slots=True)
class ResolvedVectorRouting:
    """Resolved routing metadata for vector shard assignment."""

    strategy: VectorShardingStrategy
    num_dbs: int
    metric: DistanceMetric
    centroids: np.ndarray | None = None
    hyperplanes: np.ndarray | None = None
    compiled_cel: Any | None = None
    cel_lookup: dict[int | str | bytes, int] | None = None
    routing_values: list[int | str | bytes] | None = None
    cel_expr: str | None = None


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def validate_vector_config(config: VectorWriteConfig) -> None:
    """Validate VectorWriteConfig before writing."""
    if config.index_config.dim <= 0:
        raise ConfigValidationError(f"dim must be > 0, got {config.index_config.dim}")
    if not config.s3_prefix:
        raise ConfigValidationError("s3_prefix is required")
    if config.batch_size <= 0:
        raise ConfigValidationError(f"batch_size must be > 0, got {config.batch_size}")

    sharding = config.sharding
    if sharding.strategy == VectorShardingStrategy.CLUSTER:
        if sharding.centroids is None and not sharding.train_centroids:
            raise ConfigValidationError(
                "CLUSTER sharding requires either centroids or train_centroids=True"
            )
    if sharding.strategy == VectorShardingStrategy.CEL:
        if not sharding.cel_expr:
            raise ConfigValidationError("CEL sharding requires cel_expr to be set")
        if not sharding.cel_columns:
            raise ConfigValidationError("CEL sharding requires cel_columns to be set")
    if sharding.num_probes < 1:
        raise ConfigValidationError(
            f"num_probes must be >= 1, got {sharding.num_probes}"
        )
    if (
        sharding.strategy
        in {
            VectorShardingStrategy.EXPLICIT,
            VectorShardingStrategy.CEL,
        }
        and sharding.num_probes != 1
    ):
        raise ConfigValidationError(
            f"num_probes is only supported for CLUSTER and LSH sharding, got {sharding.num_probes} for {sharding.strategy.value}"
        )


# ---------------------------------------------------------------------------
# Routing resolution
# ---------------------------------------------------------------------------


def resolve_vector_routing(
    config: VectorWriteConfig,
    *,
    sample_vectors: np.ndarray | None = None,
) -> ResolvedVectorRouting:
    """Resolve routing metadata (centroids, hyperplanes, CEL) from config.

    For CLUSTER with ``train_centroids=True``, ``sample_vectors`` must be
    provided — a ``(N, dim)`` float32 array sampled from the input data.
    The caller is responsible for sampling (framework-specific).

    Returns a frozen ``ResolvedVectorRouting`` carrying everything needed
    for per-record shard assignment.
    """
    sharding = config.sharding
    metric = config.index_config.metric
    centroids: np.ndarray | None = sharding.centroids
    hyperplanes: np.ndarray | None = sharding.hyperplanes
    num_dbs = config.num_dbs
    compiled_cel: Any | None = None
    cel_lookup: dict[int | str | bytes, int] | None = None

    # CLUSTER: train centroids from sample if needed
    if sharding.strategy == VectorShardingStrategy.CLUSTER and sharding.train_centroids:
        if num_dbs is None:
            raise ConfigValidationError(
                "num_dbs must be provided for CLUSTER sharding with train_centroids"
            )
        if sample_vectors is None or len(sample_vectors) == 0:
            raise ConfigValidationError("sample_vectors required for CLUSTER training")
        centroids = train_centroids_kmeans(sample_vectors, num_dbs, seed=42)
        log_event(
            "centroids_trained",
            logger=_logger,
            num_clusters=num_dbs,
            sample_size=len(sample_vectors),
        )

    # LSH: generate hyperplanes
    if sharding.strategy == VectorShardingStrategy.LSH and hyperplanes is None:
        if num_dbs is None:
            raise ConfigValidationError("num_dbs must be provided for LSH sharding")
        hyperplanes = lsh_generate_hyperplanes(
            sharding.num_hash_bits, config.index_config.dim, seed=42
        )

    # CLUSTER: infer or validate num_dbs from centroids
    if sharding.strategy == VectorShardingStrategy.CLUSTER and centroids is not None:
        if num_dbs is None:
            num_dbs = len(centroids)
        elif len(centroids) != num_dbs:
            raise ConfigValidationError(
                f"centroids count ({len(centroids)}) != num_dbs ({num_dbs})"
            )

    # CEL: compile expression
    if sharding.strategy == VectorShardingStrategy.CEL:
        from ..cel import build_categorical_routing_lookup, compile_cel

        assert sharding.cel_expr is not None
        assert sharding.cel_columns is not None
        compiled_cel = compile_cel(sharding.cel_expr, sharding.cel_columns)
        if sharding.routing_values is not None:
            cel_lookup = build_categorical_routing_lookup(sharding.routing_values)
            if num_dbs is None:
                num_dbs = len(sharding.routing_values)

    if num_dbs is None:
        raise ConfigValidationError(
            "num_dbs must be provided (or inferred from centroids for CLUSTER "
            "or routing_values for CEL)"
        )
    if num_dbs <= 0:
        raise ConfigValidationError(f"num_dbs must be > 0, got {num_dbs}")

    return ResolvedVectorRouting(
        strategy=sharding.strategy,
        num_dbs=num_dbs,
        metric=metric,
        centroids=centroids,
        hyperplanes=hyperplanes,
        compiled_cel=compiled_cel,
        cel_lookup=cel_lookup,
        routing_values=sharding.routing_values,
        cel_expr=sharding.cel_expr,
    )


# ---------------------------------------------------------------------------
# Shard assignment
# ---------------------------------------------------------------------------


def _validate_routed_shard_id(db_id: int, *, num_dbs: int, strategy: str) -> int:
    """Reject shard IDs outside the configured shard count."""
    if db_id < 0 or db_id >= num_dbs:
        raise ConfigValidationError(
            f"{strategy} sharding produced shard_id {db_id} outside [0, {num_dbs})"
        )
    return db_id


def assign_vector_shard(
    *,
    vector: np.ndarray,
    routing: ResolvedVectorRouting,
    shard_id: int | None = None,
    routing_context: dict[str, Any] | None = None,
) -> int:
    """Assign a single vector to a shard.

    Args:
        vector: The vector to route (shape ``(dim,)``).
        routing: Resolved routing metadata.
        shard_id: Explicit shard ID (EXPLICIT strategy only).
        routing_context: CEL evaluation context (CEL strategy only).
    """
    strategy = routing.strategy

    if strategy == VectorShardingStrategy.EXPLICIT:
        if shard_id is None:
            raise ConfigValidationError("EXPLICIT sharding requires shard_id")
        if shard_id < 0 or shard_id >= routing.num_dbs:
            raise ConfigValidationError(
                f"shard_id {shard_id} out of range [0, {routing.num_dbs})"
            )
        return shard_id

    if strategy == VectorShardingStrategy.CLUSTER:
        if routing.centroids is None:
            raise ConfigValidationError("CLUSTER sharding requires centroids")
        return _validate_routed_shard_id(
            cluster_assign(
                vector, routing.centroids, routing.metric or DistanceMetric.COSINE
            ),
            num_dbs=routing.num_dbs,
            strategy="CLUSTER",
        )

    if strategy == VectorShardingStrategy.LSH:
        if routing.hyperplanes is None:
            raise ConfigValidationError("LSH sharding requires hyperplanes")
        return _validate_routed_shard_id(
            lsh_assign(vector, routing.hyperplanes, routing.num_dbs),
            num_dbs=routing.num_dbs,
            strategy="LSH",
        )

    if strategy == VectorShardingStrategy.CEL:
        if routing.compiled_cel is None:
            raise ConfigValidationError("CEL sharding requires a compiled expression")
        if routing_context is None:
            raise ConfigValidationError("CEL sharding requires routing_context")
        from ..cel import route_cel

        return _validate_routed_shard_id(
            route_cel(
                routing.compiled_cel,
                routing_context,
                routing_values=routing.routing_values,
                lookup=routing.cel_lookup,
            ),
            num_dbs=routing.num_dbs,
            strategy="CEL",
        )

    raise ConfigValidationError(f"Unknown sharding strategy: {strategy}")


# ---------------------------------------------------------------------------
# Batch flushing
# ---------------------------------------------------------------------------


def flush_vector_shard_batch(state: VectorShardState) -> None:
    """Flush buffered records to the adapter via ``add_batch()``."""
    if not state.ids or state.adapter is None:
        return
    if all(isinstance(i, (int, np.integer)) for i in state.ids):
        ids_arr = np.array(state.ids, dtype=np.int64)
    else:
        ids_arr = np.array(state.ids, dtype=object)
    vectors_arr = np.array(state.vectors, dtype=np.float32)
    payloads_list: list[dict[str, Any]] | None = None
    if any(p is not None for p in state.payloads):
        payloads_list = [p if p is not None else {} for p in state.payloads]

    state.adapter.add_batch(ids_arr, vectors_arr, payloads_list)
    state.ids.clear()
    state.vectors.clear()
    state.payloads.clear()


# ---------------------------------------------------------------------------
# Per-shard write (used by distributed partition writers)
# ---------------------------------------------------------------------------


def coerce_vector_value(value: object) -> np.ndarray:
    """Convert framework-serialized vector values back into float32 arrays.

    Handles string representations (from Dask serialization), objects with
    a ``tolist()`` method (Arrow arrays, numpy arrays), and plain sequences.
    """
    import ast

    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, str):
        value = ast.literal_eval(value)
    elif not isinstance(value, list | tuple):
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            value = tolist()
    return np.asarray(value, dtype=np.float32)


VectorTuple = tuple[int | str, Any, dict[str, Any] | None]
"""(vector_id, vector_data, optional_payload)"""


def write_vector_shard(
    *,
    db_id: int,
    rows: Iterable[VectorTuple],
    run_id: str,
    s3_prefix: str,
    shard_prefix: str,
    db_path_template: str,
    local_root: str,
    index_config: VectorIndexConfig,
    adapter_factory: VectorIndexWriterFactory,
    batch_size: int,
    ops_limiter: TokenBucket | None = None,
    metrics_collector: MetricsCollector | None = None,
) -> RequiredShardMeta:
    """Write vectors for a single shard.

    Creates an adapter, batches rows, flushes, checkpoints, and closes.
    Used by distributed partition writers (Spark/Dask/Ray).
    """
    db_path = db_path_template.format(db_id=db_id)
    db_url = f"{s3_prefix}/{shard_prefix}/run_id={run_id}/{db_path}/attempt=00"
    local_dir = Path(local_root) / f"shard_{db_id:05d}"
    local_dir.mkdir(parents=True, exist_ok=True)

    state = VectorShardState(db_url=db_url)

    with adapter_factory(
        db_url=db_url, local_dir=local_dir, index_config=index_config
    ) as adapter:
        state.adapter = adapter
        for vec_id, vector, payload in rows:
            state.ids.append(vec_id)
            state.vectors.append(vector)
            state.payloads.append(payload)
            state.row_count += 1

            if len(state.ids) >= batch_size:
                if ops_limiter is not None:
                    ops_limiter.acquire()
                flush_vector_shard_batch(state)

        # Flush remaining
        if state.ids:
            if ops_limiter is not None:
                ops_limiter.acquire()
            flush_vector_shard_batch(state)

        state.checkpoint_id = adapter.checkpoint()
        state.db_bytes = adapter.db_bytes()

    if metrics_collector is not None:
        metrics_collector.emit(
            MetricEvent.VECTOR_SHARD_WRITE_COMPLETED,
            {"db_id": db_id, "row_count": state.row_count},
        )

    return RequiredShardMeta(
        db_id=db_id,
        db_url=db_url,
        attempt=0,
        row_count=state.row_count,
        checkpoint_id=state.checkpoint_id,
        writer_info=WriterInfo(),
        db_bytes=state.db_bytes,
    )


# ---------------------------------------------------------------------------
# Routing metadata upload
# ---------------------------------------------------------------------------


def upload_routing_metadata(
    *,
    s3_prefix: str,
    run_id: str,
    centroids: np.ndarray | None,
    hyperplanes: np.ndarray | None,
    s3_client: Any | None,
) -> tuple[str | None, str | None]:
    """Upload centroids/hyperplanes to S3. Returns (centroids_ref, hyperplanes_ref)."""
    centroids_ref: str | None = None
    hyperplanes_ref: str | None = None

    if centroids is not None:
        centroids_ref = f"{s3_prefix}/vector_meta/run_id={run_id}/centroids.npy"
        buf = io.BytesIO()
        np.save(buf, centroids)
        put_bytes(
            centroids_ref,
            buf.getvalue(),
            "application/octet-stream",
            s3_client=s3_client,
        )

    if hyperplanes is not None:
        hyperplanes_ref = f"{s3_prefix}/vector_meta/run_id={run_id}/hyperplanes.npy"
        buf = io.BytesIO()
        np.save(buf, hyperplanes)
        put_bytes(
            hyperplanes_ref,
            buf.getvalue(),
            "application/octet-stream",
            s3_client=s3_client,
        )

    return centroids_ref, hyperplanes_ref


# ---------------------------------------------------------------------------
# Manifest publishing
# ---------------------------------------------------------------------------


def publish_vector_manifest(
    *,
    config: VectorWriteConfig,
    run_id: str,
    num_dbs: int,
    winners: list[RequiredShardMeta],
    total_vectors: int,
    centroids_ref: str | None,
    hyperplanes_ref: str | None,
) -> str:
    """Build and publish manifest with vector metadata in custom fields."""
    sharding = config.sharding

    required_build = RequiredBuildMeta(
        run_id=run_id,
        created_at=datetime.now(UTC),
        num_dbs=num_dbs,
        s3_prefix=config.s3_prefix,
        key_col="_vector_id",
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.HASH,
            hash_algorithm=ShardHashAlgorithm.XXH3_64,
        ),
        db_path_template=config.output.db_path_template,
        shard_prefix=config.output.shard_prefix,
        format_version=4,
        key_encoding=KeyEncoding.RAW,
    )

    vector_custom: dict[str, Any] = {
        "dim": config.index_config.dim,
        "metric": config.index_config.metric.value,
        "index_type": config.index_config.index_type,
        "quantization": config.index_config.quantization,
        "total_vectors": total_vectors,
        "sharding_strategy": sharding.strategy.value,
        "num_probes": sharding.num_probes,
    }
    if centroids_ref is not None:
        vector_custom["centroids_ref"] = centroids_ref
    if hyperplanes_ref is not None:
        vector_custom["hyperplanes_ref"] = hyperplanes_ref
        vector_custom["num_hash_bits"] = sharding.num_hash_bits
    if sharding.strategy == VectorShardingStrategy.CEL:
        vector_custom["cel_expr"] = sharding.cel_expr
        vector_custom["cel_columns"] = sharding.cel_columns
        if sharding.routing_values is not None:
            vector_custom["routing_values"] = [
                v if isinstance(v, (int, str)) else {"__bytes_hex__": v.hex()}
                for v in sharding.routing_values
            ]

    custom_fields = dict(config.manifest.custom_manifest_fields)
    custom_fields["vector"] = vector_custom

    store = config.manifest.store or S3ManifestStore(
        config.s3_prefix,
        credential_provider=config.manifest.credential_provider
        or config.credential_provider,
        s3_connection_options=config.manifest.s3_connection_options
        or config.s3_connection_options,
        metrics_collector=config.metrics_collector,
    )

    manifest_ref = store.publish(
        run_id=run_id,
        required_build=required_build,
        shards=winners,
        custom=custom_fields,
    )

    log_event(
        "vector_manifest_published",
        logger=_logger,
        run_id=run_id,
        manifest_ref=manifest_ref,
    )

    return manifest_ref


# ---------------------------------------------------------------------------
# Adapter factory resolution
# ---------------------------------------------------------------------------


def resolve_adapter_factory(
    config: VectorWriteConfig,
    s3_client: Any | None,
) -> VectorIndexWriterFactory:
    """Return the user-provided factory or the default LanceDB factory."""
    if config.adapter_factory is not None:
        return config.adapter_factory
    from .adapters.lancedb_adapter import LanceDbWriterFactory

    return LanceDbWriterFactory(
        s3_client=s3_client,
        s3_connection_options=config.s3_connection_options,
        credential_provider=config.credential_provider,
    )
