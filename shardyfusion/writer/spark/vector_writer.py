"""Spark vector writer — write sharded vector indices from PySpark DataFrames."""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any
from uuid import uuid4

from pyspark.sql import DataFrame

from shardyfusion._rate_limiter import TokenBucket
from shardyfusion.errors import ShardAssignmentError
from shardyfusion.logging import get_logger
from shardyfusion.manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    RequiredShardMeta,
)
from shardyfusion.run_registry import RunRecordLifecycle
from shardyfusion.storage import ObstoreBackend, create_s3_store, parse_s3_url
from shardyfusion.vector._distributed import ResolvedVectorRouting
from shardyfusion.vector.config import VectorWriteConfig
from shardyfusion.vector.types import VectorIndexWriterFactory, VectorShardingStrategy

from .sharding import VECTOR_DB_ID_COL, add_vector_db_id_column

_logger = get_logger(__name__)


def _verify_vector_routing_agreement(
    df_with_db_id: DataFrame,
    *,
    id_col: str,
    vector_col: str,
    routing: ResolvedVectorRouting,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
    sample_size: int = 20,
) -> None:
    """Sample rows and verify Spark-computed vector db ids match Python routing."""
    from shardyfusion.vector._distributed import (
        assign_vector_shard,
        coerce_vector_value,
    )

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

    sampled = df_with_db_id.select(*sample_cols).limit(sample_size).collect()
    if not sampled:
        return

    mismatches: list[tuple[object, int, int]] = []
    for row in sampled:
        shard_id: int | None = None
        if routing.strategy == VectorShardingStrategy.EXPLICIT:
            assert shard_id_col is not None, "shard_id_col required for EXPLICIT"
            shard_id = int(row[shard_id_col])

        routing_context: dict[str, Any] | None = None
        if routing.strategy == VectorShardingStrategy.CEL:
            assert routing_context_cols is not None, (
                "routing_context_cols required for CEL"
            )
            routing_context = {col: row[col] for col in routing_context_cols}

        expected_db_id = assign_vector_shard(
            vector=coerce_vector_value(row[vector_col]),
            routing=routing,
            shard_id=shard_id,
            routing_context=routing_context,
        )
        spark_db_id = int(row[VECTOR_DB_ID_COL])
        if expected_db_id != spark_db_id:
            mismatches.append((row[id_col], spark_db_id, expected_db_id))

    if mismatches:
        details = "; ".join(
            f"id={vector_id}, spark={spark_db_id}, python={python_db_id}"
            for vector_id, spark_db_id, python_db_id in mismatches[:5]
        )
        raise ShardAssignmentError(
            f"Spark/Python vector routing mismatch in {len(mismatches)}/{len(sampled)} "
            f"sampled rows. First mismatches: {details}"
        )


def write_vector_sharded(
    df: DataFrame,
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
    """Write vectors from a DataFrame into N sharded vector indices and publish manifest.

    Args:
        df: PySpark DataFrame with vector and ID columns.
        config: Write configuration (num_dbs, s3_prefix, sharding strategy, etc.).
        vector_col: Name of the vector column (list[float] or array[float]).
        id_col: Name of the ID column.
        payload_cols: Optional payload columns to include as metadata.
        shard_id_col: Column with explicit shard IDs (EXPLICIT strategy only).
        routing_context_cols: Columns for CEL expression evaluation (CEL strategy only).
        max_writes_per_second: Optional rate limit.
        verify_routing: If True (default), spot-check that Spark-assigned vector
            shard IDs match ``assign_vector_shard()``.

    Returns:
        BuildResult with manifest reference and stats.
    """
    import numpy as np

    from shardyfusion.vector._distributed import (
        write_vector_shard as write_vector_shard_core,
    )

    started = time.perf_counter()
    mc = config.metrics_collector
    from shardyfusion.vector._distributed import (
        publish_vector_manifest,
        resolve_adapter_factory,
        resolve_vector_routing,
        upload_routing_metadata,
    )

    run_id = config.output.run_id or uuid4().hex

    with RunRecordLifecycle.start(
        config=config,
        run_id=run_id,
        writer_type="vector-spark",
    ) as run_record:
        sample_vectors: np.ndarray | None = None
        if (
            config.sharding.strategy == VectorShardingStrategy.CLUSTER
            and config.sharding.train_centroids
        ):
            sample_limit = config.sharding.centroids_training_sample_size
            sample_df = (
                df.select(vector_col).sample(fraction=0.1, seed=42).limit(sample_limit)
            )
            sample_rows = sample_df.collect()
            sample_vectors = np.array(
                [row[vector_col] for row in sample_rows if row[vector_col] is not None],
                dtype=np.float32,
            )

        routing = resolve_vector_routing(config, sample_vectors=sample_vectors)

        adapter_factory = resolve_adapter_factory(config)

        df_with_id, num_dbs = add_vector_db_id_column(
            df,
            vector_col=vector_col,
            routing=routing,
            shard_id_col=shard_id_col,
            routing_context_cols=routing_context_cols,
        )

        if verify_routing and num_dbs > 0:
            _verify_vector_routing_agreement(
                df_with_id,
                id_col=id_col,
                vector_col=vector_col,
                routing=routing,
                shard_id_col=shard_id_col,
                routing_context_cols=routing_context_cols,
            )

        _id_col = id_col
        _vector_col = vector_col
        _payload_cols = payload_cols
        _max_writes_per_second = max_writes_per_second
        _metrics_collector = mc
        _s3_prefix = config.s3_prefix
        _shard_prefix = config.output.shard_prefix
        _db_path_template = config.output.db_path_template
        _local_root = config.output.local_root
        _index_config = config.index_config
        _batch_size = config.batch_size

        df_shuffled = df_with_id.repartition(num_dbs, VECTOR_DB_ID_COL)

        results_rdd = df_shuffled.rdd.mapPartitionsWithIndex(
            lambda db_id, rows: _write_vector_partition(
                db_id,
                rows,
                run_id,
                adapter_factory,
                _max_writes_per_second,
                _id_col,
                _vector_col,
                _payload_cols,
                write_vector_shard_core,
                _metrics_collector,
                _s3_prefix,
                _shard_prefix,
                _db_path_template,
                _local_root,
                _index_config,
                _batch_size,
            )
        )

        results = list(results_rdd.toLocalIterator())
        winners = results
        num_attempts = len(results)

        total_vectors = sum(w.row_count for w in winners)
        total_duration_ms = int((time.perf_counter() - started) * 1000)

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


def _write_vector_partition(
    db_id: int,
    rows: Iterable,
    run_id: str,
    adapter_factory: VectorIndexWriterFactory,
    max_writes_per_second: float | None,
    id_col: str,
    vector_col: str,
    payload_cols: list[str] | None,
    write_vector_shard_core: Callable[..., RequiredShardMeta],
    metrics_collector: Any | None,
    s3_prefix: str,
    shard_prefix: str,
    db_path_template: str,
    local_root: str,
    index_config: Any,
    batch_size: int,
) -> list:
    """Write vectors for a single partition."""
    from shardyfusion.vector._distributed import coerce_vector_value as _coerce_vec

    rows_list = list(rows)
    if not rows_list:
        return []

    ops_limiter: TokenBucket | None = None
    if max_writes_per_second is not None:
        ops_limiter = TokenBucket(
            max_writes_per_second,
            metrics_collector=metrics_collector,
            limiter_type="ops",
        )

    groups: dict = defaultdict(list)
    for row in rows_list:
        vector_db_id = row[VECTOR_DB_ID_COL]
        groups[vector_db_id].append(row)

    results = []
    for vid, group in groups.items():
        rows_iter = (
            (
                row[id_col],
                _coerce_vec(row[vector_col]),
                {col: row[col] for col in payload_cols} if payload_cols else None,
            )
            for row in group
        )
        result = write_vector_shard_core(
            db_id=vid,
            rows=rows_iter,
            run_id=run_id,
            s3_prefix=s3_prefix,
            shard_prefix=shard_prefix,
            db_path_template=db_path_template,
            local_root=local_root,
            index_config=index_config,
            adapter_factory=adapter_factory,
            batch_size=batch_size,
            ops_limiter=ops_limiter,
            metrics_collector=metrics_collector,
        )
        results.append(result)

    return results
