"""Single-database writer using Dask for distributed sorting."""

import math
import os
import time
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from uuid import uuid4

import dask.dataframe as dd
import pandas as pd

from shardyfusion._rate_limiter import TokenBucket
from shardyfusion._writer_core import (
    ShardAttemptResult,
    assemble_build_result,
    build_manifest_artifact,
    publish_manifest_and_current,
    select_winners,
    update_min_max,
)
from shardyfusion.config import WriteConfig
from shardyfusion.errors import ConfigValidationError, SlatedbSparkShardedError
from shardyfusion.logging import FailureSeverity, log_failure
from shardyfusion.manifest import BuildResult
from shardyfusion.serde import KeyEncoder, ValueSpec, make_key_encoder
from shardyfusion.sharding_types import ShardingSpec, ShardingStrategy
from shardyfusion.slatedb_adapter import (
    DbAdapter,
    DbAdapterFactory,
    SlateDbFactory,
)
from shardyfusion.storage import join_s3
from shardyfusion.type_defs import JsonObject, KeyLike


class DaskCacheContext:
    """Persist a Dask DataFrame for the lifetime of the context."""

    def __init__(self, ddf: dd.DataFrame, *, enabled: bool = True) -> None:
        self._ddf = ddf
        self._enabled = enabled
        self._persisted: dd.DataFrame | None = None

    def __enter__(self) -> dd.DataFrame:
        if not self._enabled:
            return self._ddf
        persisted: dd.DataFrame = self._ddf.persist()  # type: ignore[assignment]
        self._persisted = persisted
        return persisted

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        # Release reference to persisted futures; Dask GC handles cleanup
        self._persisted = None


def write_single_db(
    ddf: dd.DataFrame,
    config: WriteConfig,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_keys: bool = True,
    num_partitions: int | None = None,
    prefetch_partitions: bool = True,
    cache_input: bool = True,
    max_writes_per_second: float | None = None,
) -> BuildResult:
    """Write a Dask DataFrame into a single SlateDB database, optionally sorting by key first.

    Uses Dask for the expensive global sort, then streams sorted partitions to the
    local process where they are written into one SlateDB instance.

    ``num_partitions`` controls how many partitions the sorted data is split
    into before streaming.  When *None* (the default) the function calls
    ``len(ddf)`` and derives ``ceil(rows / batch_size)`` partitions.  Pass an
    explicit value to skip that count — useful when the DataFrame is large
    and not cached.
    """

    if config.num_dbs != 1:
        raise ConfigValidationError(
            f"write_single_db requires num_dbs=1, got {config.num_dbs}"
        )

    if key_col not in ddf.columns:
        raise ConfigValidationError(
            f"Key column `{key_col}` not found in DataFrame columns: {list(ddf.columns)}"
        )

    started = time.perf_counter()
    run_id = config.output.run_id or uuid4().hex

    with DaskCacheContext(ddf, enabled=cache_input) as cached_ddf:
        return _write_single_db_impl(
            ddf=cached_ddf,
            config=config,
            run_id=run_id,
            started=started,
            key_col=key_col,
            value_spec=value_spec,
            sort_keys=sort_keys,
            num_partitions=num_partitions,
            prefetch_partitions=prefetch_partitions,
            max_writes_per_second=max_writes_per_second,
        )


def _write_single_db_impl(
    *,
    ddf: dd.DataFrame,
    config: WriteConfig,
    run_id: str,
    started: float,
    key_col: str,
    value_spec: ValueSpec,
    sort_keys: bool,
    num_partitions: int | None,
    prefetch_partitions: bool,
    max_writes_per_second: float | None,
) -> BuildResult:
    """Implementation: sort, stream partitions, write, publish."""

    # Compute partition count (triggers a compute when num_partitions is None).
    if num_partitions is None:
        total_rows = len(ddf)
        num_partitions = max(1, math.ceil(total_rows / config.batch_size))

    # Sort and repartition.
    if sort_keys:
        ddf_prepared = ddf.sort_values(key_col).repartition(npartitions=num_partitions)
    else:
        ddf_prepared = ddf.repartition(npartitions=num_partitions)

    # No sharding happens in single-db mode.  The sort task graph is lazy so
    # its actual execution cost is included in write_duration_ms below.
    shard_duration_ms = 0

    # Phase 3: Stream partitions to single writer
    write_started = time.perf_counter()
    attempt_result = _stream_to_single_db(
        ddf=ddf_prepared,
        config=config,
        run_id=run_id,
        key_col=key_col,
        value_spec=value_spec,
        prefetch_partitions=prefetch_partitions,
        max_writes_per_second=max_writes_per_second,
    )
    write_duration_ms = int((time.perf_counter() - write_started) * 1000)

    # Phase 4: Publish
    attempts = [attempt_result]
    winners = select_winners(attempts, num_dbs=1)

    resolved_sharding = ShardingSpec(strategy=ShardingStrategy.HASH)

    manifest_started = time.perf_counter()
    artifact = build_manifest_artifact(
        config=config,
        run_id=run_id,
        resolved_sharding=resolved_sharding,
        winners=winners,
        key_col=key_col,
    )
    publish_result = publish_manifest_and_current(
        config=config,
        run_id=run_id,
        artifact=artifact,
    )
    manifest_duration_ms = int((time.perf_counter() - manifest_started) * 1000)

    return assemble_build_result(
        run_id=run_id,
        winners=winners,
        artifact=artifact,
        manifest_ref=publish_result.manifest_ref,
        current_ref=publish_result.current_ref,
        attempts=attempts,
        shard_duration_ms=shard_duration_ms,
        write_duration_ms=write_duration_ms,
        manifest_duration_ms=manifest_duration_ms,
        started=started,
    )


def _stream_to_single_db(
    *,
    ddf: dd.DataFrame,
    config: WriteConfig,
    run_id: str,
    key_col: str,
    value_spec: ValueSpec,
    prefetch_partitions: bool,
    max_writes_per_second: float | None,
) -> ShardAttemptResult:
    """Stream Dask partitions to a single SlateDB adapter."""

    db_id = 0
    attempt = 0
    key_encoder = make_key_encoder(config.key_encoding)

    db_rel_path = config.output.db_path_template.format(db_id=db_id)
    db_url = join_s3(
        config.s3_prefix,
        config.output.tmp_prefix,
        f"run_id={run_id}",
        db_rel_path,
        f"attempt={attempt:02d}",
    )
    local_dir = os.path.join(
        config.output.local_root,
        f"run_id={run_id}",
        f"db={db_id:05d}",
        f"attempt={attempt:02d}",
    )
    os.makedirs(local_dir, exist_ok=True)

    factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory()

    bucket: TokenBucket | None = None
    if max_writes_per_second is not None:
        bucket = TokenBucket(max_writes_per_second)

    row_count = 0
    min_key: KeyLike | None = None
    max_key: KeyLike | None = None
    checkpoint_id: str | None = None

    write_started = time.perf_counter()

    try:
        with factory(db_url=db_url, local_dir=local_dir) as adapter:
            partitions = ddf.to_delayed()

            if prefetch_partitions and len(partitions) > 1:
                row_count, min_key, max_key = _write_with_prefetch(
                    partitions=partitions,
                    adapter=adapter,
                    key_col=key_col,
                    key_encoder=key_encoder,
                    value_spec=value_spec,
                    batch_size=config.batch_size,
                    bucket=bucket,
                )
            else:
                row_count, min_key, max_key = _write_sequential(
                    partitions=partitions,
                    adapter=adapter,
                    key_col=key_col,
                    key_encoder=key_encoder,
                    value_spec=value_spec,
                    batch_size=config.batch_size,
                    bucket=bucket,
                )

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
    except SlatedbSparkShardedError:
        raise
    except Exception as exc:
        log_failure(
            "single_db_dask_write_failed",
            severity=FailureSeverity.ERROR,
            error=exc,
            db_id=db_id,
            attempt=attempt,
            db_url=db_url,
            rows_written=row_count,
            include_traceback=True,
        )
        raise SlatedbSparkShardedError(
            f"Single-db write failed for db_id={db_id}, attempt={attempt}: {exc}"
        ) from exc

    writer_info: JsonObject = {
        "stage_id": None,
        "task_attempt_id": None,
        "attempt": attempt,
        "duration_ms": int((time.perf_counter() - write_started) * 1000),
    }

    return ShardAttemptResult(
        db_id=db_id,
        db_url=db_url,
        attempt=attempt,
        row_count=row_count,
        min_key=min_key,
        max_key=max_key,
        checkpoint_id=checkpoint_id,
        writer_info=writer_info,
    )


def _write_pdf_rows(
    pdf: pd.DataFrame,
    adapter: DbAdapter,
    key_col: str,
    key_encoder: KeyEncoder,
    value_spec: ValueSpec,
    batch_size: int,
    bucket: TokenBucket | None,
    row_count: int,
    min_key: KeyLike | None,
    max_key: KeyLike | None,
) -> tuple[int, KeyLike | None, KeyLike | None]:
    """Write rows from a pandas DataFrame to the adapter."""

    batch: list[tuple[bytes, bytes]] = []

    for _, row in pdf.iterrows():
        key_value: Any = row[key_col]
        # Convert numpy scalars to Python types
        if hasattr(key_value, "item"):
            key_value = key_value.item()

        key_bytes = key_encoder(key_value)
        value_bytes = value_spec.encode(row)

        batch.append((key_bytes, value_bytes))
        row_count += 1
        min_key, max_key = update_min_max(min_key, max_key, key_value)

        if len(batch) >= batch_size:
            if bucket is not None:
                bucket.acquire(len(batch))
            adapter.write_batch(batch)
            batch.clear()

    if batch:
        if bucket is not None:
            bucket.acquire(len(batch))
        adapter.write_batch(batch)

    return row_count, min_key, max_key


def _write_with_prefetch(
    *,
    partitions: list[Any],
    adapter: DbAdapter,
    key_col: str,
    key_encoder: KeyEncoder,
    value_spec: ValueSpec,
    batch_size: int,
    bucket: TokenBucket | None,
) -> tuple[int, KeyLike | None, KeyLike | None]:
    """Write partitions with background prefetching of the next partition."""

    row_count = 0
    min_key: KeyLike | None = None
    max_key: KeyLike | None = None

    with ThreadPoolExecutor(max_workers=1) as prefetcher:
        next_future = prefetcher.submit(partitions[0].compute)
        for i in range(len(partitions)):
            current_pdf: pd.DataFrame = next_future.result()
            if i + 1 < len(partitions):
                next_future = prefetcher.submit(partitions[i + 1].compute)

            if not current_pdf.empty:
                row_count, min_key, max_key = _write_pdf_rows(
                    current_pdf,
                    adapter,
                    key_col,
                    key_encoder,
                    value_spec,
                    batch_size,
                    bucket,
                    row_count,
                    min_key,
                    max_key,
                )

    return row_count, min_key, max_key


def _write_sequential(
    *,
    partitions: list[Any],
    adapter: DbAdapter,
    key_col: str,
    key_encoder: KeyEncoder,
    value_spec: ValueSpec,
    batch_size: int,
    bucket: TokenBucket | None,
) -> tuple[int, KeyLike | None, KeyLike | None]:
    """Write partitions sequentially without prefetching."""

    row_count = 0
    min_key: KeyLike | None = None
    max_key: KeyLike | None = None

    for partition in partitions:
        pdf: pd.DataFrame = partition.compute()
        if not pdf.empty:
            row_count, min_key, max_key = _write_pdf_rows(
                pdf,
                adapter,
                key_col,
                key_encoder,
                value_spec,
                batch_size,
                bucket,
                row_count,
                min_key,
                max_key,
            )

    return row_count, min_key, max_key
