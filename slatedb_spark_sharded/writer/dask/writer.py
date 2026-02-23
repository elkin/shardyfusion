"""Dask DataFrame-based sharded writer (no Spark/Java dependency).

Uses ``dask.delayed`` to write each shard in parallel.  Each shard task
receives all partitions (as pandas DataFrames) and filters rows by db_id,
so the dataset must fit in aggregate worker memory.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING
from uuid import uuid4

import dask

from slatedb_spark_sharded._rate_limiter import TokenBucket
from slatedb_spark_sharded._writer_core import (
    _assemble_build_result,
    _build_manifest_artifact,
    _publish_manifest_and_current,
    _route_key,
    _select_winners,
    _ShardAttemptResult,
    _update_min_max,
)
from slatedb_spark_sharded.config import WriteConfig
from slatedb_spark_sharded.errors import ConfigValidationError
from slatedb_spark_sharded.manifest import BuildResult
from slatedb_spark_sharded.serde import KeyEncoder, ValueSpec, make_key_encoder
from slatedb_spark_sharded.sharding_types import (
    DB_ID_COL,
    KeyEncoding,
    ShardingSpec,
    ShardingStrategy,
)
from slatedb_spark_sharded.slatedb_adapter import DbAdapterFactory, SlateDbFactory
from slatedb_spark_sharded.storage import _join_s3
from slatedb_spark_sharded.type_defs import JsonObject, KeyLike

if TYPE_CHECKING:
    import dask.dataframe as dd
    import pandas as pd


def write_sharded_dask(
    ddf: dd.DataFrame,
    config: WriteConfig,
    *,
    key_col: str,
    value_spec: ValueSpec,
    max_writes_per_second: float | None = None,
) -> BuildResult:
    """Write a Dask DataFrame into *N* independent sharded databases.

    Parallelism is controlled by the active Dask scheduler (set globally
    via ``dask.config.set(scheduler=...)`` or per-call via the scheduler
    kwarg on ``dask.compute``).

    Parameters
    ----------
    ddf:
        Dask DataFrame to shard and write.
    config:
        Write configuration (num_dbs, s3_prefix, sharding, etc.).
    key_col:
        Name of the integer key column in *ddf*.
    value_spec:
        Serialization strategy for row values.
    max_writes_per_second:
        Optional write-rate limit (batches per second per shard).
    """

    _validate_sharding(config)
    run_id = config.output.run_id or uuid4().hex
    started = time.perf_counter()
    factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory()
    key_encoder = make_key_encoder(config.key_encoding)

    # Step 1: Assign db_id to each row using pandas-level routing.
    meta = ddf._meta.assign(**{DB_ID_COL: 0})
    ddf_with_id = ddf.map_partitions(
        _assign_db_ids,
        key_col=key_col,
        num_dbs=config.num_dbs,
        sharding=config.sharding,
        key_encoding=config.key_encoding,
        meta=meta,
    )

    # Step 2: Get delayed partitions and create per-shard write tasks.
    # Each shard task receives *all* partitions (Dask computes each
    # partition once and shares the result) and filters for its db_id.
    delayed_parts = ddf_with_id.to_delayed()
    shard_tasks = [
        dask.delayed(_write_shard)(
            delayed_parts,
            db_id=db_id,
            config=config,
            run_id=run_id,
            key_col=key_col,
            value_spec=value_spec,
            key_encoder=key_encoder,
            factory=factory,
            max_writes_per_second=max_writes_per_second,
        )
        for db_id in range(config.num_dbs)
    ]

    # Step 3: Compute all shard writes in parallel.
    attempts: list[_ShardAttemptResult] = list(dask.compute(*shard_tasks))
    write_duration_ms = int((time.perf_counter() - started) * 1000)

    # Step 4: Select winners, build manifest, publish.
    winners = _select_winners(attempts, num_dbs=config.num_dbs)

    manifest_started = time.perf_counter()
    artifact = _build_manifest_artifact(
        config=config,
        run_id=run_id,
        resolved_sharding=config.sharding,
        winners=winners,
        key_col=key_col,
    )
    publish_result = _publish_manifest_and_current(
        config=config,
        run_id=run_id,
        artifact=artifact,
    )
    manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

    return _assemble_build_result(
        run_id=run_id,
        winners=winners,
        artifact=artifact,
        manifest_ref=publish_result.manifest_ref,
        current_ref=publish_result.current_ref,
        attempts=attempts,
        shard_duration_ms=0,
        write_duration_ms=write_duration_ms,
        manifest_duration_ms=manifest_duration_ms,
        started=started,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_sharding(config: WriteConfig) -> None:
    if config.sharding.strategy == ShardingStrategy.CUSTOM_EXPR:
        raise ConfigValidationError(
            "Custom expression sharding is not supported in the Dask writer."
        )
    if (
        config.sharding.strategy == ShardingStrategy.RANGE
        and config.sharding.boundaries is None
    ):
        raise ConfigValidationError(
            "Range sharding without explicit boundaries requires Spark."
        )


def _assign_db_ids(
    pdf: pd.DataFrame,
    key_col: str,
    num_dbs: int,
    sharding: ShardingSpec,
    key_encoding: KeyEncoding,
) -> pd.DataFrame:
    """Assign ``_slatedb_db_id`` to each row in a pandas partition."""

    db_ids = pdf[key_col].apply(
        lambda key: _route_key(
            key,
            num_dbs=num_dbs,
            sharding=sharding,
            key_encoding=key_encoding,
        )
    )
    return pdf.assign(**{DB_ID_COL: db_ids})


def _make_db_url(config: WriteConfig, run_id: str, db_id: int, attempt: int) -> str:
    db_rel_path = config.output.db_path_template.format(db_id=db_id)
    return _join_s3(
        config.s3_prefix,
        config.output.tmp_prefix,
        f"run_id={run_id}",
        db_rel_path,
        f"attempt={attempt:02d}",
    )


def _make_local_dir(config: WriteConfig, run_id: str, db_id: int, attempt: int) -> str:
    return os.path.join(
        config.output.local_root,
        f"run_id={run_id}",
        f"db={db_id:05d}",
        f"attempt={attempt:02d}",
    )


def _write_shard(
    partitions: list[pd.DataFrame],
    *,
    db_id: int,
    config: WriteConfig,
    run_id: str,
    key_col: str,
    value_spec: ValueSpec,
    key_encoder: KeyEncoder,
    factory: DbAdapterFactory,
    max_writes_per_second: float | None,
) -> _ShardAttemptResult:
    """Write one shard by filtering rows from all partitions."""

    attempt = 0
    db_url = _make_db_url(config, run_id, db_id, attempt)
    local_dir = _make_local_dir(config, run_id, db_id, attempt)
    os.makedirs(local_dir, exist_ok=True)

    bucket: TokenBucket | None = None
    if max_writes_per_second is not None:
        bucket = TokenBucket(max_writes_per_second)

    row_count = 0
    min_key: KeyLike | None = None
    max_key: KeyLike | None = None
    checkpoint_id: str | None = None
    partition_started = time.perf_counter()

    with factory(db_url=db_url, local_dir=local_dir) as adapter:
        batch: list[tuple[bytes, bytes]] = []

        for pdf in partitions:
            shard_rows = pdf.loc[pdf[DB_ID_COL] == db_id].drop(columns=[DB_ID_COL])
            for row_dict in shard_rows.to_dict("records"):
                key = row_dict[key_col]
                key_bytes = key_encoder(key)
                value_bytes = value_spec.encode(row_dict)

                batch.append((key_bytes, value_bytes))
                row_count += 1
                min_key, max_key = _update_min_max(min_key, max_key, key)

                if len(batch) >= config.batch_size:
                    if bucket is not None:
                        bucket.acquire(1)
                    adapter.write_batch(batch)
                    batch.clear()

        if batch:
            if bucket is not None:
                bucket.acquire(1)
            adapter.write_batch(batch)
            batch.clear()

        adapter.flush()
        checkpoint_id = adapter.checkpoint()

    writer_info: JsonObject = {
        "stage_id": None,
        "task_attempt_id": None,
        "attempt": attempt,
        "duration_ms": int((time.perf_counter() - partition_started) * 1000),
    }

    return _ShardAttemptResult(
        db_id=db_id,
        db_url=db_url,
        attempt=attempt,
        row_count=row_count,
        min_key=min_key,
        max_key=max_key,
        checkpoint_id=checkpoint_id,
        writer_info=writer_info,
    )
