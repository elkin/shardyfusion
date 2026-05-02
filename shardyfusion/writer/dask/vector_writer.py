"""Dask-based vector sharded writer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import dask.dataframe as dd
import pandas as pd

from shardyfusion._rate_limiter import TokenBucket
from shardyfusion._writer_core import _normalize_vector_id
from shardyfusion.errors import ShardAssignmentError
from shardyfusion.logging import get_logger
from shardyfusion.manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    RequiredShardMeta,
)
from shardyfusion.run_registry import RunRecordLifecycle
from shardyfusion.vector.config import VectorWriteConfig
from shardyfusion.vector.types import VectorShardingStrategy

from .sharding import VECTOR_DB_ID_COL, _stack_vector_values
from .writer import _RESULT_META

_logger = get_logger(__name__)


@dataclass(slots=True)
class _VectorPartitionWriteRuntime:
    """Picklable runtime config for Dask vector partition writers."""

    run_id: str
    s3_prefix: str
    shard_prefix: str
    db_path_template: str
    local_root: str
    index_config: Any
    adapter_factory: Any
    batch_size: int
    payload_cols: list[str] | None = None
    max_writes_per_second: float | None = None
    metrics_collector: Any | None = None


def _vector_result_row(result: RequiredShardMeta) -> dict[str, object]:
    return {
        "db_id": result.db_id,
        "db_url": result.db_url,
        "attempt": result.attempt,
        "row_count": result.row_count,
        "min_key": result.min_key,
        "max_key": result.max_key,
        "checkpoint_id": result.checkpoint_id,
        "writer_info": result.writer_info,
        "db_bytes": result.db_bytes,
        "all_attempt_urls": (),
    }


def _verify_vector_routing_agreement(
    ddf_with_id: dd.DataFrame,
    *,
    id_col: str,
    vector_col: str,
    routing: Any,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify vector db_id column matches Python routing."""
    from shardyfusion.vector._distributed import (
        assign_vector_shard,
        coerce_vector_value,
    )
    from shardyfusion.vector.types import VectorShardingStrategy

    sample_cols = [id_col, vector_col, VECTOR_DB_ID_COL]
    if (
        routing.strategy == VectorShardingStrategy.EXPLICIT
        and shard_id_col is not None
        and shard_id_col not in sample_cols
    ):
        sample_cols.append(shard_id_col)
    if routing.strategy == VectorShardingStrategy.CEL:
        assert routing_context_cols is not None, (
            "routing_context_cols required for CEL verification"
        )
        sample_cols.extend(c for c in routing_context_cols if c not in sample_cols)

    sampled = ddf_with_id[sample_cols].head(sample_size, npartitions=-1)
    if sampled.empty:
        return

    mismatches: list[tuple[object, int, int]] = []
    for _, row in sampled.iterrows():
        vector_id = row[id_col].item() if hasattr(row[id_col], "item") else row[id_col]
        shard_id: int | None = None
        if routing.strategy == VectorShardingStrategy.EXPLICIT:
            assert shard_id_col is not None, "shard_id_col required for EXPLICIT"
            shard_value = row[shard_id_col]
            shard_id = int(
                shard_value.item() if hasattr(shard_value, "item") else shard_value
            )

        routing_context: dict[str, object] | None = None
        if routing.strategy == VectorShardingStrategy.CEL:
            assert routing_context_cols is not None
            routing_context = {
                col: row[col].item() if hasattr(row[col], "item") else row[col]
                for col in routing_context_cols
            }

        expected_db_id = assign_vector_shard(
            vector=coerce_vector_value(row[vector_col]),
            routing=routing,
            shard_id=shard_id,
            routing_context=routing_context,
        )
        computed_db_id = int(row[VECTOR_DB_ID_COL])
        if expected_db_id != computed_db_id:
            mismatches.append((vector_id, computed_db_id, expected_db_id))

    if mismatches:
        details = "; ".join(
            f"id={vector_id}, dask={dask_db_id}, python={python_db_id}"
            for vector_id, dask_db_id, python_db_id in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Dask/Python vector routing mismatch in {len(mismatches)}/{len(sampled)} "
            f"sampled rows. First mismatches: {details}"
        )


def write_vector_sharded(
    ddf: dd.DataFrame,
    config: VectorWriteConfig,
    *,
    vector_col: str,
    id_col: str,
    payload_cols: list[str] | None = None,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
    max_writes_per_second: float | None = None,
    verify_routing: bool = True,
) -> BuildResult:
    """Write vectors from a Dask DataFrame into N sharded vector indices.

    Args:
        ddf: Dask DataFrame with vector and ID columns.
        config: Vector write configuration.
        vector_col: Name of the vector column.
        id_col: Name of the ID column.
        payload_cols: Optional payload columns.
        shard_id_col: Column with explicit shard IDs (EXPLICIT strategy only).
        routing_context_cols: Columns for CEL expression evaluation.
        max_writes_per_second: Optional rate limit.
        verify_routing: If True (default), spot-check that Dask-assigned vector
            shard IDs match ``assign_vector_shard()``.

    Returns:
        BuildResult with manifest reference and stats.
    """
    import numpy as np

    from shardyfusion.vector._distributed import (
        publish_vector_manifest,
        resolve_adapter_factory,
        resolve_vector_routing,
        upload_routing_metadata,
    )

    from .sharding import add_vector_db_id_column

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    with RunRecordLifecycle.start(
        config=config,
        run_id=run_id,
        writer_type="vector-dask",
    ) as run_record:
        sample_vectors: np.ndarray | None = None
        if (
            config.sharding.strategy == VectorShardingStrategy.CLUSTER
            and config.sharding.train_centroids
        ):
            sample_limit = config.sharding.centroids_training_sample_size
            sample_values = (
                ddf[vector_col]
                .sample(
                    frac=0.1,
                    random_state=42,
                )
                .head(sample_limit, compute=True)
                .tolist()
            )
            if not sample_values:
                sample_values = (
                    ddf[vector_col].head(sample_limit, compute=True).tolist()
                )
            sample_vectors = _stack_vector_values(sample_values)

        routing = resolve_vector_routing(config, sample_vectors=sample_vectors)

        adapter_factory = resolve_adapter_factory(config)

        ddf_with_id, num_dbs = add_vector_db_id_column(
            ddf,
            vector_col=vector_col,
            routing=routing,
            shard_id_col=shard_id_col,
            routing_context_cols=routing_context_cols,
        )

        if verify_routing and num_dbs > 0:
            _verify_vector_routing_agreement(
                ddf_with_id,
                id_col=id_col,
                vector_col=vector_col,
                routing=routing,
                shard_id_col=shard_id_col,
                routing_context_cols=routing_context_cols,
            )

        ddf_shuffled = ddf_with_id.shuffle(on=VECTOR_DB_ID_COL, npartitions=num_dbs)

        _id_col = id_col
        _vector_col = vector_col
        _runtime = _VectorPartitionWriteRuntime(
            run_id=run_id,
            s3_prefix=config.s3_prefix,
            shard_prefix=config.output.shard_prefix,
            db_path_template=config.output.db_path_template,
            local_root=config.output.local_root,
            index_config=config.index_config,
            adapter_factory=adapter_factory,
            batch_size=config.batch_size,
            payload_cols=payload_cols,
            max_writes_per_second=max_writes_per_second,
            metrics_collector=config.metrics_collector,
        )

        def _write_partition_vector(
            pdf: pd.DataFrame, runtime: _VectorPartitionWriteRuntime
        ) -> pd.DataFrame:  # noqa: N803
            from shardyfusion.vector._distributed import (
                VectorTuple,
                coerce_vector_value,
                write_vector_shard,
            )

            if pdf.empty:
                return pd.DataFrame(columns=["db_id", "db_url", "attempt", "row_count"])

            ops_limiter: TokenBucket | None = None
            if runtime.max_writes_per_second is not None:
                ops_limiter = TokenBucket(
                    runtime.max_writes_per_second,
                    metrics_collector=runtime.metrics_collector,
                    limiter_type="ops",
                )

            pdf_copy = pdf.copy()
            pdf_copy["_temp_vec_id"] = pdf_copy[VECTOR_DB_ID_COL].astype(int)
            groups = pdf_copy.groupby("_temp_vec_id")

            results = []
            for db_id, group in groups:
                _pcols = runtime.payload_cols
                rows_iter: list[VectorTuple] = [
                    (
                        _normalize_vector_id(row[_id_col]),
                        coerce_vector_value(row[_vector_col]),
                        {col: row[col] for col in _pcols} if _pcols else None,
                    )
                    for _, row in group.iterrows()
                ]
                result = write_vector_shard(
                    db_id=int(db_id),  # type: ignore[arg-type]
                    rows=rows_iter,
                    run_id=runtime.run_id,
                    s3_prefix=runtime.s3_prefix,
                    shard_prefix=runtime.shard_prefix,
                    db_path_template=runtime.db_path_template,
                    local_root=runtime.local_root,
                    index_config=runtime.index_config,
                    adapter_factory=runtime.adapter_factory,
                    batch_size=runtime.batch_size,
                    ops_limiter=ops_limiter,
                    metrics_collector=runtime.metrics_collector,
                )
                results.append(result)

            return pd.DataFrame.from_records(
                [_vector_result_row(result) for result in results],
                columns=list(_RESULT_META.columns),
            )

        ddf_results = ddf_shuffled.map_partitions(
            _write_partition_vector,
            _runtime,
            meta=_RESULT_META,
        )

        results = ddf_results.compute()

        winners = list(results.itertuples(index=False, name=None))
        winners = [
            RequiredShardMeta(
                db_id=int(db_id),
                db_url=db_url,
                attempt=int(attempt),
                row_count=int(row_count),
                checkpoint_id=checkpoint_id,
                writer_info=writer_info,
                db_bytes=int(db_bytes),
            )
            for (
                db_id,
                db_url,
                attempt,
                row_count,
                _min_key,
                _max_key,
                checkpoint_id,
                writer_info,
                db_bytes,
                _all_attempt_urls,
            ) in winners
        ]
        num_attempts = len(winners)

        total_vectors = sum(w.row_count for w in winners)
        total_duration_ms = int((time.perf_counter() - started) * 1000)

        from shardyfusion.storage import ObstoreBackend, create_s3_store, parse_s3_url

        credentials = (
            config.credential_provider.resolve() if config.credential_provider else None
        )
        bucket, _ = parse_s3_url(config.s3_prefix)
        backend = ObstoreBackend(
            create_s3_store(
                bucket=bucket,
                credentials=credentials,
                connection_options=config.s3_connection_options,
            )
        )
        centroids_ref, hyperplanes_ref = upload_routing_metadata(
            s3_prefix=config.s3_prefix,
            run_id=run_id,
            centroids=routing.centroids,
            hyperplanes=routing.hyperplanes,
            backend=backend,
        )

        manifest_start = time.perf_counter()
        manifest_ref = publish_vector_manifest(
            config=config,
            run_id=run_id,
            num_dbs=num_dbs,
            winners=winners,
            total_vectors=total_vectors,
            centroids_ref=centroids_ref,
            hyperplanes_ref=hyperplanes_ref,
        )
        manifest_duration_ms = int((time.perf_counter() - manifest_start) * 1000)
        run_record.set_manifest_ref(manifest_ref)
        run_record.mark_succeeded()

        stats = BuildStats(
            durations=BuildDurations(
                sharding_ms=0,
                write_ms=total_duration_ms - manifest_duration_ms,
                manifest_ms=manifest_duration_ms,
                total_ms=total_duration_ms,
            ),
            num_attempt_results=num_attempts,
            num_winners=len(winners),
            rows_written=total_vectors,
        )

        return BuildResult(
            run_id=run_id,
            winners=winners,
            manifest_ref=manifest_ref,
            stats=stats,
            run_record_ref=run_record.run_record_ref,
        )
