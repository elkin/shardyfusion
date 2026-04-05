"""Sharded vector writer — Python iterator-based."""

from __future__ import annotations

import contextlib
import io
import time
import uuid
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
    BuildDurations,
    BuildResult,
    BuildStats,
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
    WriterInfo,
)
from ..manifest_store import S3ManifestStore
from ..metrics._events import MetricEvent
from ..sharding_types import KeyEncoding, ShardingStrategy
from ..storage import create_s3_client, put_bytes
from .config import VectorWriteConfig
from .sharding import (
    cluster_assign,
    lsh_assign,
    lsh_generate_hyperplanes,
    train_centroids_kmeans,
)
from .types import (
    DistanceMetric,
    VectorIndexWriter,
    VectorIndexWriterFactory,
    VectorRecord,
    VectorShardingStrategy,
)

# CEL support — lazy import to avoid hard dependency on cel extra
_compiled_cel_cache: dict[str, Any] = {}

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _ShardState:
    adapter: VectorIndexWriter | None = None
    db_url: str = ""
    ids: list[int | str] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)
    payloads: list[dict[str, Any] | None] = field(default_factory=list)
    row_count: int = 0
    checkpoint_id: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def write_vector_sharded(
    records: Iterable[VectorRecord],
    config: VectorWriteConfig,
    *,
    parallel: bool = False,
) -> BuildResult:
    """Write vectors into N sharded indices on S3.

    Args:
        records: Iterable of VectorRecord to write.
        config: Write configuration.
        parallel: If True, use multiprocessing (not yet implemented).

    Returns:
        BuildResult with manifest reference and stats.
    """
    if parallel:
        raise ConfigValidationError("Parallel vector writing is not yet implemented")

    started = time.perf_counter()
    run_id = config.output.run_id or str(uuid.uuid4())
    mc = config.metrics_collector

    if mc is not None:
        mc.emit(MetricEvent.VECTOR_WRITE_STARTED, {"run_id": run_id})

    _validate_config(config)

    credentials = (
        config.credential_provider.resolve() if config.credential_provider else None
    )
    s3_client = create_s3_client(credentials, config.s3_connection_options)

    # Resolve sharding metadata
    sharding = config.sharding
    resolved_centroids: np.ndarray | None = sharding.centroids
    resolved_hyperplanes: np.ndarray | None = sharding.hyperplanes
    num_dbs = config.num_dbs

    # For CLUSTER with training, we need to buffer records first
    records_list: list[VectorRecord] | None = None
    if sharding.strategy == VectorShardingStrategy.CLUSTER and sharding.train_centroids:
        records_list = list(records)
        if num_dbs is None:
            raise ConfigValidationError(
                "num_dbs must be provided for CLUSTER sharding with train_centroids"
            )
        sample_size = min(len(records_list), sharding.centroids_training_sample_size)
        sample_vectors = np.array(
            [r.vector for r in records_list[:sample_size]], dtype=np.float32
        )
        resolved_centroids = train_centroids_kmeans(sample_vectors, num_dbs, seed=42)
        log_event(
            "centroids_trained",
            logger=_logger,
            num_clusters=num_dbs,
            sample_size=sample_size,
        )

    if sharding.strategy == VectorShardingStrategy.LSH and resolved_hyperplanes is None:
        if num_dbs is None:
            raise ConfigValidationError("num_dbs must be provided for LSH sharding")
        resolved_hyperplanes = lsh_generate_hyperplanes(
            sharding.num_hash_bits, config.index_config.dim, seed=42
        )

    if (
        sharding.strategy == VectorShardingStrategy.CLUSTER
        and resolved_centroids is not None
    ):
        if num_dbs is None:
            num_dbs = len(resolved_centroids)
        elif len(resolved_centroids) != num_dbs:
            raise ConfigValidationError(
                f"centroids count ({len(resolved_centroids)}) != num_dbs ({num_dbs})"
            )

    # CEL: compile expression and resolve num_dbs from routing_values
    compiled_cel: Any | None = None
    cel_lookup: dict[int | str | bytes, int] | None = None
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

    # Adapter factory
    adapter_factory: VectorIndexWriterFactory
    if config.adapter_factory is not None:
        adapter_factory = config.adapter_factory
    else:
        from .adapters.usearch_adapter import USearchWriterFactory

        adapter_factory = USearchWriterFactory(s3_client=s3_client)

    # Rate limiter
    ops_limiter: TokenBucket | None = None
    if config.max_writes_per_second is not None:
        ops_limiter = TokenBucket(
            config.max_writes_per_second,
            metrics_collector=mc,
            limiter_type="ops",
        )

    # Write records
    shard_start = time.perf_counter()
    records_iter: Iterable[VectorRecord] = (
        records_list if records_list is not None else records
    )

    shard_states = _write_single_process(
        records=records_iter,
        config=config,
        num_dbs=num_dbs,
        run_id=run_id,
        adapter_factory=adapter_factory,
        resolved_centroids=resolved_centroids,
        resolved_hyperplanes=resolved_hyperplanes,
        ops_limiter=ops_limiter,
        s3_client=s3_client,
        compiled_cel=compiled_cel,
        cel_lookup=cel_lookup,
    )
    shard_duration_ms = int((time.perf_counter() - shard_start) * 1000)

    # Build winners list
    winners: list[RequiredShardMeta] = []
    total_vectors = 0
    for db_id in sorted(shard_states.keys()):
        state = shard_states[db_id]
        if state.row_count == 0:
            continue
        total_vectors += state.row_count
        winners.append(
            RequiredShardMeta(
                db_id=db_id,
                db_url=state.db_url,
                attempt=0,
                row_count=state.row_count,
                checkpoint_id=state.checkpoint_id,
                writer_info=WriterInfo(),
            )
        )

        if mc is not None:
            mc.emit(
                MetricEvent.VECTOR_SHARD_WRITE_COMPLETED,
                {"db_id": db_id, "row_count": state.row_count},
            )

    # Upload centroids/hyperplanes to S3 for reader
    centroids_ref: str | None = None
    hyperplanes_ref: str | None = None

    if resolved_centroids is not None:
        centroids_ref = f"{config.s3_prefix}/vector_meta/centroids.npy"
        buf = io.BytesIO()
        np.save(buf, resolved_centroids)
        put_bytes(
            centroids_ref,
            buf.getvalue(),
            "application/octet-stream",
            s3_client=s3_client,
        )

    if resolved_hyperplanes is not None:
        hyperplanes_ref = f"{config.s3_prefix}/vector_meta/hyperplanes.npy"
        buf = io.BytesIO()
        np.save(buf, resolved_hyperplanes)
        put_bytes(
            hyperplanes_ref,
            buf.getvalue(),
            "application/octet-stream",
            s3_client=s3_client,
        )

    # Publish manifest
    manifest_start = time.perf_counter()
    manifest_ref = _publish_vector_manifest(
        config=config,
        run_id=run_id,
        num_dbs=num_dbs,
        winners=winners,
        total_vectors=total_vectors,
        centroids_ref=centroids_ref,
        hyperplanes_ref=hyperplanes_ref,
        s3_client=s3_client,
    )
    manifest_duration_ms = int((time.perf_counter() - manifest_start) * 1000)

    total_duration_ms = int((time.perf_counter() - started) * 1000)
    stats = BuildStats(
        durations=BuildDurations(
            sharding_ms=0,
            write_ms=shard_duration_ms,
            manifest_ms=manifest_duration_ms,
            total_ms=total_duration_ms,
        ),
        num_attempt_results=len(shard_states),
        num_winners=len(winners),
        rows_written=total_vectors,
    )

    if mc is not None:
        mc.emit(
            MetricEvent.VECTOR_WRITE_COMPLETED,
            {
                "run_id": run_id,
                "total_vectors": total_vectors,
                "num_shards": len(winners),
                "elapsed_ms": total_duration_ms,
            },
        )

    log_event(
        "vector_write_completed",
        logger=_logger,
        run_id=run_id,
        total_vectors=total_vectors,
        num_shards=len(winners),
        elapsed_ms=total_duration_ms,
    )

    return BuildResult(
        run_id=run_id,
        winners=winners,
        manifest_ref=manifest_ref,
        stats=stats,
        run_record_ref=None,
    )


# ---------------------------------------------------------------------------
# Single-process writer
# ---------------------------------------------------------------------------


def _write_single_process(
    *,
    records: Iterable[VectorRecord],
    config: VectorWriteConfig,
    num_dbs: int,
    run_id: str,
    adapter_factory: VectorIndexWriterFactory,
    resolved_centroids: np.ndarray | None,
    resolved_hyperplanes: np.ndarray | None,
    ops_limiter: TokenBucket | None,
    s3_client: Any | None,
    compiled_cel: Any | None = None,
    cel_lookup: dict[int | str | bytes, int] | None = None,
) -> dict[int, _ShardState]:
    """Write records to shards in a single process."""
    sharding = config.sharding
    metric = config.index_config.metric
    shard_states: dict[int, _ShardState] = {}

    with contextlib.ExitStack() as stack:
        for record in records:
            # Assign shard
            db_id = _assign_shard(
                record=record,
                strategy=sharding.strategy,
                num_dbs=num_dbs,
                metric=metric,
                centroids=resolved_centroids,
                hyperplanes=resolved_hyperplanes,
                compiled_cel=compiled_cel,
                routing_values=sharding.routing_values,
                cel_lookup=cel_lookup,
            )

            # Ensure shard state
            if db_id not in shard_states:
                state = _ShardState()
                db_path = config.output.db_path_template.format(db_id=db_id)
                shard_prefix = config.output.shard_prefix
                state.db_url = f"{config.s3_prefix}/{shard_prefix}/run_id={run_id}/{db_path}/attempt=00"
                local_dir = Path(config.output.local_root) / f"shard_{db_id:05d}"
                adapter = adapter_factory(
                    db_url=state.db_url,
                    local_dir=local_dir,
                    index_config=config.index_config,
                )
                state.adapter = stack.enter_context(adapter)
                shard_states[db_id] = state

            state = shard_states[db_id]
            state.ids.append(record.id)
            state.vectors.append(record.vector)
            state.payloads.append(record.payload)
            state.row_count += 1

            # Flush batch if threshold reached
            if len(state.ids) >= config.batch_size:
                if ops_limiter is not None:
                    ops_limiter.acquire()
                _flush_shard_batch(state)

        # Flush remaining batches and finalize
        for state in shard_states.values():
            if state.ids:
                if ops_limiter is not None:
                    ops_limiter.acquire()
                _flush_shard_batch(state)
            if state.adapter is not None:
                state.checkpoint_id = state.adapter.checkpoint()

    return shard_states


def _assign_shard(
    *,
    record: VectorRecord,
    strategy: VectorShardingStrategy,
    num_dbs: int,
    metric: DistanceMetric | None,
    centroids: np.ndarray | None,
    hyperplanes: np.ndarray | None,
    compiled_cel: Any | None = None,
    routing_values: list[int | str | bytes] | None = None,
    cel_lookup: dict[int | str | bytes, int] | None = None,
) -> int:
    """Assign a record to a shard based on the sharding strategy."""
    if strategy == VectorShardingStrategy.EXPLICIT:
        if record.shard_id is None:
            raise ConfigValidationError(
                "EXPLICIT sharding requires shard_id on every VectorRecord"
            )
        if record.shard_id < 0 or record.shard_id >= num_dbs:
            raise ConfigValidationError(
                f"shard_id {record.shard_id} out of range [0, {num_dbs})"
            )
        return record.shard_id

    if strategy == VectorShardingStrategy.CLUSTER:
        if centroids is None:
            raise ConfigValidationError("CLUSTER sharding requires centroids")
        return cluster_assign(record.vector, centroids, metric or DistanceMetric.COSINE)

    if strategy == VectorShardingStrategy.LSH:
        if hyperplanes is None:
            raise ConfigValidationError("LSH sharding requires hyperplanes")
        return lsh_assign(record.vector, hyperplanes, num_dbs)

    if strategy == VectorShardingStrategy.CEL:
        if compiled_cel is None:
            raise ConfigValidationError("CEL sharding requires a compiled expression")
        if record.routing_context is None:
            raise ConfigValidationError(
                "CEL sharding requires routing_context on every VectorRecord"
            )
        from ..cel import route_cel

        return route_cel(
            compiled_cel,
            record.routing_context,
            routing_values=routing_values,
            lookup=cel_lookup,
        )

    raise ConfigValidationError(f"Unknown sharding strategy: {strategy}")


def _flush_shard_batch(state: _ShardState) -> None:
    """Flush buffered records to the adapter."""
    if not state.ids or state.adapter is None:
        return
    ids_arr = np.array(state.ids, dtype=np.int64)
    vectors_arr = np.array(state.vectors, dtype=np.float32)
    payloads_list: list[dict[str, Any]] | None = None
    if any(p is not None for p in state.payloads):
        payloads_list = [p if p is not None else {} for p in state.payloads]

    state.adapter.add_batch(ids_arr, vectors_arr, payloads_list)
    state.ids.clear()
    state.vectors.clear()
    state.payloads.clear()


# ---------------------------------------------------------------------------
# Manifest publishing
# ---------------------------------------------------------------------------


def _validate_config(config: VectorWriteConfig) -> None:
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


def _publish_vector_manifest(
    *,
    config: VectorWriteConfig,
    run_id: str,
    num_dbs: int,
    winners: list[RequiredShardMeta],
    total_vectors: int,
    centroids_ref: str | None,
    hyperplanes_ref: str | None,
    s3_client: Any | None,
) -> str:
    """Build and publish manifest with vector metadata in custom fields."""
    sharding = config.sharding

    required_build = RequiredBuildMeta(
        run_id=run_id,
        created_at=datetime.now(UTC),
        num_dbs=num_dbs,
        s3_prefix=config.s3_prefix,
        key_col="_vector_id",
        sharding=ManifestShardingSpec(strategy=ShardingStrategy.HASH),
        db_path_template=config.output.db_path_template,
        shard_prefix=config.output.shard_prefix,
        format_version=2,
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
                v if isinstance(v, (int, str)) else v.hex()
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
