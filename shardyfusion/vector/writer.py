"""Sharded vector writer — Python iterator-based."""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Iterable
from copy import copy
from pathlib import Path

import numpy as np

from .._rate_limiter import TokenBucket
from ..errors import ConfigValidationError
from ..logging import get_logger, log_event
from ..manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    RequiredShardMeta,
    WriterInfo,
)
from ..metrics._events import MetricEvent
from ..run_registry import managed_run_record
from ..storage import create_s3_client
from ._distributed import (
    ResolvedVectorRouting,
    VectorShardState,
    assign_vector_shard,
    flush_vector_shard_batch,
    publish_vector_manifest,
    resolve_adapter_factory,
    resolve_vector_routing,
    upload_routing_metadata,
    validate_vector_config,
)
from .config import VectorWriteConfig
from .types import (
    VectorIndexWriterFactory,
    VectorRecord,
    VectorShardingStrategy,
)

_logger = get_logger(__name__)


_validate_config = validate_vector_config
_flush_shard_batch = flush_vector_shard_batch
_ShardState = VectorShardState


def _write_single_process(
    *,
    records: Iterable[VectorRecord],
    config: VectorWriteConfig,
    routing: ResolvedVectorRouting,
    run_id: str,
    adapter_factory: VectorIndexWriterFactory,
    ops_limiter: TokenBucket | None,
) -> dict[int, VectorShardState]:
    """Write records to shards in a single process (new API)."""
    shard_states: dict[int, VectorShardState] = {}

    with contextlib.ExitStack() as stack:
        for record in records:
            db_id = assign_vector_shard(
                vector=record.vector,
                routing=routing,
                shard_id=record.shard_id,
                routing_context=record.routing_context,
            )

            if db_id not in shard_states:
                state = VectorShardState()
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

            if len(state.ids) >= config.batch_size:
                if ops_limiter is not None:
                    ops_limiter.acquire()
                flush_vector_shard_batch(state)

        for state in shard_states.values():
            if state.ids:
                if ops_limiter is not None:
                    ops_limiter.acquire()
                flush_vector_shard_batch(state)
            if state.adapter is not None:
                state.checkpoint_id = state.adapter.checkpoint()
                state.db_bytes = (
                    state.adapter.db_bytes()
                    if hasattr(state.adapter, "db_bytes")
                    else 0
                )

    return shard_states


def write_vector_sharded(
    records: Iterable[VectorRecord],
    config: VectorWriteConfig,
) -> BuildResult:
    """Write vectors into N sharded indices on S3.

    Args:
        records: Iterable of VectorRecord to write.
        config: Write configuration.

    Returns:
        BuildResult with manifest reference and stats.
    """
    started = time.perf_counter()
    run_id = config.output.run_id or str(uuid.uuid4())
    mc = config.metrics_collector

    if mc is not None:
        mc.emit(MetricEvent.VECTOR_WRITE_STARTED, {"run_id": run_id})

    validate_vector_config(config)

    with managed_run_record(
        config=copy(config),
        run_id=run_id,
        writer_type="vector-python",
    ) as run_record:
        credentials = (
            config.credential_provider.resolve() if config.credential_provider else None
        )
        s3_client = create_s3_client(credentials, config.s3_connection_options)

        records_list: list[VectorRecord] | None = None
        sample_vectors: np.ndarray | None = None
        if (
            config.sharding.strategy == VectorShardingStrategy.CLUSTER
            and config.sharding.train_centroids
        ):
            records_list = list(records)
            if not records_list:
                raise ConfigValidationError(
                    "Cannot train centroids from an empty record set"
                )
            sample_size = min(
                len(records_list), config.sharding.centroids_training_sample_size
            )
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(len(records_list), size=sample_size, replace=False)
            sample_vectors = np.array(
                [records_list[int(idx)].vector for idx in sample_idx],
                dtype=np.float32,
            )

        routing = resolve_vector_routing(config, sample_vectors=sample_vectors)

        adapter_factory = resolve_adapter_factory(config, s3_client)

        ops_limiter: TokenBucket | None = None
        if config.max_writes_per_second is not None:
            ops_limiter = TokenBucket(
                config.max_writes_per_second,
                metrics_collector=mc,
                limiter_type="ops",
            )

        shard_start = time.perf_counter()
        shard_states = _write_single_process(
            records=records_list if records_list is not None else records,
            config=config,
            routing=routing,
            run_id=run_id,
            adapter_factory=adapter_factory,
            ops_limiter=ops_limiter,
        )
        shard_duration_ms = int((time.perf_counter() - shard_start) * 1000)

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
                    db_bytes=state.db_bytes,
                )
            )

            if mc is not None:
                mc.emit(
                    MetricEvent.VECTOR_SHARD_WRITE_COMPLETED,
                    {"db_id": db_id, "row_count": state.row_count},
                )

        centroids_ref, hyperplanes_ref = upload_routing_metadata(
            s3_prefix=config.s3_prefix,
            run_id=run_id,
            centroids=routing.centroids,
            hyperplanes=routing.hyperplanes,
            s3_client=s3_client,
        )

        manifest_start = time.perf_counter()
        manifest_ref = publish_vector_manifest(
            config=config,
            run_id=run_id,
            num_dbs=routing.num_dbs,
            winners=winners,
            total_vectors=total_vectors,
            centroids_ref=centroids_ref,
            hyperplanes_ref=hyperplanes_ref,
        )
        manifest_duration_ms = int((time.perf_counter() - manifest_start) * 1000)
        run_record.set_manifest_ref(manifest_ref)
        run_record.mark_succeeded()

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
            run_record_ref=run_record.run_record_ref,
        )


_publish_vector_manifest = publish_vector_manifest
