"""Pure-Python iterator-based sharded writer (no Spark dependency)."""

import multiprocessing
import os
import time
from collections.abc import Iterable
from typing import Callable, TypeVar
from uuid import uuid4

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
from slatedb_spark_sharded.errors import (
    ConfigValidationError,
    SlatedbSparkShardedError,
)
from slatedb_spark_sharded.logging import FailureSeverity, log_failure
from slatedb_spark_sharded.manifest import BuildResult
from slatedb_spark_sharded.serde import encode_key
from slatedb_spark_sharded.sharding_types import ShardingStrategy
from slatedb_spark_sharded.slatedb_adapter import DbAdapterFactory, SlateDbFactory
from slatedb_spark_sharded.storage import _join_s3
from slatedb_spark_sharded.type_defs import JsonObject, KeyLike

T = TypeVar("T")


def write_sharded(
    records: Iterable[T],
    config: WriteConfig,
    *,
    key_fn: Callable[[T], KeyLike],
    value_fn: Callable[[T], bytes],
    parallel: bool = False,
    max_queue_size: int = 100,
    max_writes_per_second: float | None = None,
) -> BuildResult:
    """Write an iterable of records into N sharded databases."""

    _validate_sharding(config)
    run_id = config.output.run_id or uuid4().hex
    started = time.perf_counter()

    factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory()

    if parallel:
        attempts = _write_parallel(
            records=records,
            config=config,
            run_id=run_id,
            factory=factory,
            key_fn=key_fn,
            value_fn=value_fn,
            max_queue_size=max_queue_size,
            max_writes_per_second=max_writes_per_second,
        )
    else:
        attempts = _write_single_process(
            records=records,
            config=config,
            run_id=run_id,
            factory=factory,
            key_fn=key_fn,
            value_fn=value_fn,
            max_writes_per_second=max_writes_per_second,
        )

    write_duration_ms = int((time.perf_counter() - started) * 1000)
    winners = _select_winners(attempts, num_dbs=config.num_dbs)

    manifest_started = time.perf_counter()
    artifact = _build_manifest_artifact(
        config=config,
        run_id=run_id,
        resolved_sharding=config.sharding,
        winners=winners,
        key_col="_key",
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


def _validate_sharding(config: WriteConfig) -> None:
    if config.sharding.strategy == ShardingStrategy.CUSTOM_EXPR:
        raise ConfigValidationError(
            "Custom expression sharding is not supported in the Python writer."
        )
    if (
        config.sharding.strategy == ShardingStrategy.RANGE
        and config.sharding.boundaries is None
    ):
        raise ConfigValidationError(
            "Range sharding without explicit boundaries requires Spark."
        )


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


def _write_single_process(
    *,
    records: Iterable[T],
    config: WriteConfig,
    run_id: str,
    factory: DbAdapterFactory,
    key_fn: Callable[[T], KeyLike],
    value_fn: Callable[[T], bytes],
    max_writes_per_second: float | None,
) -> list[_ShardAttemptResult]:
    """Single-process mode: all adapters open simultaneously, single pass."""

    attempt = 0
    num_dbs = config.num_dbs
    bucket: TokenBucket | None = None
    if max_writes_per_second is not None:
        bucket = TokenBucket(max_writes_per_second)

    # Open all shard adapters
    adapters = []
    db_urls: list[str] = []
    try:
        for db_id in range(num_dbs):
            db_url = _make_db_url(config, run_id, db_id, attempt)
            local_dir = _make_local_dir(config, run_id, db_id, attempt)
            os.makedirs(local_dir, exist_ok=True)
            adapter = factory(db_url=db_url, local_dir=local_dir)
            adapter.__enter__()
            adapters.append(adapter)
            db_urls.append(db_url)

        # Per-shard tracking
        batches: list[list[tuple[bytes, bytes]]] = [[] for _ in range(num_dbs)]
        row_counts = [0] * num_dbs
        min_keys: list[KeyLike | None] = [None] * num_dbs
        max_keys: list[KeyLike | None] = [None] * num_dbs

        for record in records:
            key = key_fn(record)
            db_id = _route_key(
                key,
                num_dbs=num_dbs,
                sharding=config.sharding,
                key_encoding=config.key_encoding,
            )
            key_bytes = encode_key(key, encoding=config.key_encoding)
            value_bytes = value_fn(record)

            batches[db_id].append((key_bytes, value_bytes))
            row_counts[db_id] += 1
            min_keys[db_id], max_keys[db_id] = _update_min_max(
                min_keys[db_id], max_keys[db_id], key
            )

            if len(batches[db_id]) >= config.batch_size:
                if bucket is not None:
                    bucket.acquire(1)
                adapters[db_id].write_batch(batches[db_id])
                batches[db_id].clear()

        # Flush remaining
        for db_id in range(num_dbs):
            if batches[db_id]:
                if bucket is not None:
                    bucket.acquire(1)
                adapters[db_id].write_batch(batches[db_id])
                batches[db_id].clear()

        # Finalize
        checkpoint_ids: list[str | None] = []
        for db_id in range(num_dbs):
            adapters[db_id].flush()
            checkpoint_ids.append(adapters[db_id].checkpoint())

    finally:
        for adapter in adapters:
            try:
                adapter.__exit__(None, None, None)
            except Exception:
                pass

    results: list[_ShardAttemptResult] = []
    for db_id in range(num_dbs):
        writer_info: JsonObject = {
            "stage_id": None,
            "task_attempt_id": None,
            "attempt": attempt,
            "duration_ms": 0,
        }
        results.append(
            _ShardAttemptResult(
                db_id=db_id,
                db_url=db_urls[db_id],
                attempt=attempt,
                row_count=row_counts[db_id],
                min_key=min_keys[db_id],
                max_key=max_keys[db_id],
                checkpoint_id=checkpoint_ids[db_id]
                if db_id < len(checkpoint_ids)
                else None,
                writer_info=writer_info,
            )
        )

    return results


def _shard_worker(
    db_id: int,
    queue: "multiprocessing.Queue[list[tuple[bytes, bytes]] | None]",
    result_queue: "multiprocessing.Queue[_ShardAttemptResult]",
    config: WriteConfig,
    run_id: str,
    factory: DbAdapterFactory,
    max_writes_per_second: float | None,
) -> None:
    """Worker process: consume chunks from queue, write to one shard."""

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

    try:
        with factory(db_url=db_url, local_dir=local_dir) as adapter:
            batch: list[tuple[bytes, bytes]] = []
            while True:
                chunk = queue.get()
                if chunk is None:
                    break
                batch.extend(chunk)
                row_count += len(chunk)

                while len(batch) >= config.batch_size:
                    if bucket is not None:
                        bucket.acquire(1)
                    adapter.write_batch(batch[: config.batch_size])
                    batch = batch[config.batch_size :]

            if batch:
                if bucket is not None:
                    bucket.acquire(1)
                adapter.write_batch(batch)
                batch.clear()

            adapter.flush()
            checkpoint_id = adapter.checkpoint()
    except Exception as exc:
        log_failure(
            "python_shard_worker_failed",
            severity=FailureSeverity.ERROR,
            error=exc,
            db_id=db_id,
        )
        raise SlatedbSparkShardedError(
            f"Shard write failed for db_id={db_id}: {exc}"
        ) from exc

    writer_info: JsonObject = {
        "stage_id": None,
        "task_attempt_id": None,
        "attempt": attempt,
        "duration_ms": 0,
    }

    result_queue.put(
        _ShardAttemptResult(
            db_id=db_id,
            db_url=db_url,
            attempt=attempt,
            row_count=row_count,
            min_key=min_key,
            max_key=max_key,
            checkpoint_id=checkpoint_id,
            writer_info=writer_info,
        )
    )


def _write_parallel(
    *,
    records: Iterable[T],
    config: WriteConfig,
    run_id: str,
    factory: DbAdapterFactory,
    key_fn: Callable[[T], KeyLike],
    value_fn: Callable[[T], bytes],
    max_queue_size: int,
    max_writes_per_second: float | None,
) -> list[_ShardAttemptResult]:
    """Multi-process mode: one worker per shard, main routes to queues."""

    num_dbs = config.num_dbs
    chunk_size = max(1, config.batch_size // 10)
    ctx = multiprocessing.get_context("spawn")

    queues: list[multiprocessing.Queue[list[tuple[bytes, bytes]] | None]] = [
        ctx.Queue(maxsize=max_queue_size) for _ in range(num_dbs)
    ]
    result_queue: multiprocessing.Queue[_ShardAttemptResult] = ctx.Queue()

    workers: list[multiprocessing.Process] = []
    for db_id in range(num_dbs):
        p = ctx.Process(
            target=_shard_worker,
            args=(
                db_id,
                queues[db_id],
                result_queue,
                config,
                run_id,
                factory,
                max_writes_per_second,
            ),
        )
        p.start()
        workers.append(p)  # type: ignore[arg-type]  # SpawnProcess is a Process subclass

    chunk_bufs: list[list[tuple[bytes, bytes]]] = [[] for _ in range(num_dbs)]
    # Track min/max in main process for parallel mode
    min_keys: list[KeyLike | None] = [None] * num_dbs
    max_keys: list[KeyLike | None] = [None] * num_dbs

    try:
        for record in records:
            key = key_fn(record)
            db_id = _route_key(
                key,
                num_dbs=num_dbs,
                sharding=config.sharding,
                key_encoding=config.key_encoding,
            )
            key_bytes = encode_key(key, encoding=config.key_encoding)
            value_bytes = value_fn(record)
            chunk_bufs[db_id].append((key_bytes, value_bytes))
            min_keys[db_id], max_keys[db_id] = _update_min_max(
                min_keys[db_id], max_keys[db_id], key
            )

            if len(chunk_bufs[db_id]) >= chunk_size:
                queues[db_id].put(chunk_bufs[db_id])
                chunk_bufs[db_id] = []

        # Flush remaining chunks
        for db_id in range(num_dbs):
            if chunk_bufs[db_id]:
                queues[db_id].put(chunk_bufs[db_id])
                chunk_bufs[db_id] = []
            queues[db_id].put(None)  # sentinel

    except Exception:
        # On error, send sentinels to all workers so they exit
        for db_id in range(num_dbs):
            try:
                queues[db_id].put(None)
            except Exception:
                pass
        raise
    finally:
        for p in workers:
            p.join(timeout=60)

    # Collect results
    results: list[_ShardAttemptResult] = []
    for _ in range(num_dbs):
        result = result_queue.get(timeout=10)
        # Patch in min/max from main process tracking
        result.min_key = min_keys[result.db_id]
        result.max_key = max_keys[result.db_id]
        results.append(result)

    return results
