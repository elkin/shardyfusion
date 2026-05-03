"""Test adapters for exercising the new writer API from legacy-style tests."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from shardyfusion.config import (
    BufferingOptions,
    ColumnWriteInput,
    DaskWriteOptions,
    KvWriteRateLimitConfig,
    PythonRecordInput,
    PythonWriteOptions,
    RayWriteOptions,
    SharedMemoryOptions,
    SingleDbWriteConfig,
    SingleDbWriteOptions,
    SparkWriteOptions,
    VectorColumnInput,
)
from shardyfusion.errors import ConfigValidationError
from shardyfusion.manifest import BuildResult
from shardyfusion.serde import ValueSpec
from shardyfusion.vector.config import VectorWriteOptions

T = TypeVar("T")


def _set_kv_rate_limits(
    config: Any,
    *,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
) -> None:
    if max_writes_per_second is not None or max_write_bytes_per_second is not None:
        config.rate_limits = KvWriteRateLimitConfig(
            max_writes_per_second=max_writes_per_second,
            max_write_bytes_per_second=max_write_bytes_per_second,
        )


def write_python_hash_sharded(
    records: Iterable[T],
    config: Any,
    *,
    key_fn: Callable[[T], Any],
    value_fn: Callable[[T], bytes],
    columns_fn: Callable[[T], dict[str, Any]] | None = None,
    vector_fn: Callable[[T], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    parallel: bool = False,
    max_queue_size: int = 100,
    max_parallel_shared_memory_bytes: int | None = 256 * 1024 * 1024,
    max_parallel_shared_memory_bytes_per_worker: int | None = 32 * 1024 * 1024,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    max_total_batched_items: int | None = None,
    max_total_batched_bytes: int | None = None,
) -> BuildResult:
    from shardyfusion.writer.python import write_hash_sharded

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_hash_sharded(
        records,
        config,
        PythonRecordInput(
            key_fn=key_fn,
            value_fn=value_fn,
            columns_fn=columns_fn,
            vector_fn=vector_fn,
        ),
        PythonWriteOptions(
            parallel=parallel,
            max_queue_size=max_queue_size,
            shared_memory=SharedMemoryOptions(
                max_total_bytes=max_parallel_shared_memory_bytes,
                max_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
            ),
            buffering=BufferingOptions(
                max_total_batched_items=max_total_batched_items,
                max_total_batched_bytes=max_total_batched_bytes,
            ),
        ),
    )


def write_python_cel_sharded(
    records: Iterable[T],
    config: Any,
    *,
    key_fn: Callable[[T], Any],
    value_fn: Callable[[T], bytes],
    columns_fn: Callable[[T], dict[str, Any]] | None = None,
    vector_fn: Callable[[T], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    parallel: bool = False,
    max_queue_size: int = 100,
    max_parallel_shared_memory_bytes: int | None = 256 * 1024 * 1024,
    max_parallel_shared_memory_bytes_per_worker: int | None = 32 * 1024 * 1024,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    max_total_batched_items: int | None = None,
    max_total_batched_bytes: int | None = None,
) -> BuildResult:
    from shardyfusion.writer.python import write_cel_sharded

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_cel_sharded(
        records,
        config,
        PythonRecordInput(
            key_fn=key_fn,
            value_fn=value_fn,
            columns_fn=columns_fn,
            vector_fn=vector_fn,
        ),
        PythonWriteOptions(
            parallel=parallel,
            max_queue_size=max_queue_size,
            shared_memory=SharedMemoryOptions(
                max_total_bytes=max_parallel_shared_memory_bytes,
                max_bytes_per_worker=max_parallel_shared_memory_bytes_per_worker,
            ),
            buffering=BufferingOptions(
                max_total_batched_items=max_total_batched_items,
                max_total_batched_bytes=max_total_batched_bytes,
            ),
        ),
    )


def _column_input(
    *,
    key_col: str,
    value_spec: ValueSpec,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    vector_col: str | None = None,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
) -> ColumnWriteInput:
    vector = (
        VectorColumnInput(
            vector_col=vector_col,
            id_col=id_col,
            payload_cols=payload_cols,
        )
        if vector_col is not None
        else None
    )
    return ColumnWriteInput(
        key_col=key_col,
        value_spec=value_spec,
        vector=vector,
        vector_fn=vector_fn,
    )


def write_dask_hash_sharded(
    ddf: Any,
    config: Any,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool = False,
    verify_routing: bool = True,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    vector_col: str | None = None,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
) -> BuildResult:
    from shardyfusion.writer.dask import write_hash_sharded

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_hash_sharded(
        ddf,
        config,
        _column_input(
            key_col=key_col,
            value_spec=value_spec,
            vector_fn=vector_fn,
            vector_col=vector_col,
            id_col=id_col,
            payload_cols=payload_cols,
        ),
        DaskWriteOptions(
            sort_within_partitions=sort_within_partitions,
            verify_routing=verify_routing,
        ),
    )


def write_dask_cel_sharded(
    ddf: Any,
    config: Any,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool = False,
    verify_routing: bool = True,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    vector_col: str | None = None,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
) -> BuildResult:
    from shardyfusion.writer.dask import write_cel_sharded

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_cel_sharded(
        ddf,
        config,
        _column_input(
            key_col=key_col,
            value_spec=value_spec,
            vector_fn=vector_fn,
            vector_col=vector_col,
            id_col=id_col,
            payload_cols=payload_cols,
        ),
        DaskWriteOptions(
            sort_within_partitions=sort_within_partitions,
            verify_routing=verify_routing,
        ),
    )


def write_ray_hash_sharded(
    ds: Any,
    config: Any,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool = False,
    verify_routing: bool = True,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    vector_col: str | None = None,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
) -> BuildResult:
    from shardyfusion.writer.ray import write_hash_sharded

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_hash_sharded(
        ds,
        config,
        _column_input(
            key_col=key_col,
            value_spec=value_spec,
            vector_fn=vector_fn,
            vector_col=vector_col,
            id_col=id_col,
            payload_cols=payload_cols,
        ),
        RayWriteOptions(
            sort_within_partitions=sort_within_partitions,
            verify_routing=verify_routing,
        ),
    )


def write_ray_cel_sharded(
    ds: Any,
    config: Any,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool = False,
    verify_routing: bool = True,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    vector_col: str | None = None,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
) -> BuildResult:
    from shardyfusion.writer.ray import write_cel_sharded

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_cel_sharded(
        ds,
        config,
        _column_input(
            key_col=key_col,
            value_spec=value_spec,
            vector_fn=vector_fn,
            vector_col=vector_col,
            id_col=id_col,
            payload_cols=payload_cols,
        ),
        RayWriteOptions(
            sort_within_partitions=sort_within_partitions,
            verify_routing=verify_routing,
        ),
    )


def write_spark_hash_sharded(
    df: Any,
    config: Any,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool = False,
    verify_routing: bool = True,
    cache_input: bool = False,
    storage_level: Any | None = None,
    spark_conf_overrides: dict[str, str] | None = None,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    vector_col: str | None = None,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
) -> BuildResult:
    from shardyfusion.writer.spark import write_hash_sharded

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_hash_sharded(
        df,
        config,
        _column_input(
            key_col=key_col,
            value_spec=value_spec,
            vector_fn=vector_fn,
            vector_col=vector_col,
            id_col=id_col,
            payload_cols=payload_cols,
        ),
        SparkWriteOptions(
            sort_within_partitions=sort_within_partitions,
            verify_routing=verify_routing,
            cache_input=cache_input,
            storage_level=storage_level,
            spark_conf_overrides=spark_conf_overrides,
        ),
    )


def write_spark_cel_sharded(
    df: Any,
    config: Any,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_within_partitions: bool = False,
    verify_routing: bool = True,
    cache_input: bool = False,
    storage_level: Any | None = None,
    spark_conf_overrides: dict[str, str] | None = None,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]]
    | None = None,
    vector_col: str | None = None,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
) -> BuildResult:
    from shardyfusion.writer.spark import write_cel_sharded

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_cel_sharded(
        df,
        config,
        _column_input(
            key_col=key_col,
            value_spec=value_spec,
            vector_fn=vector_fn,
            vector_col=vector_col,
            id_col=id_col,
            payload_cols=payload_cols,
        ),
        SparkWriteOptions(
            sort_within_partitions=sort_within_partitions,
            verify_routing=verify_routing,
            cache_input=cache_input,
            storage_level=storage_level,
            spark_conf_overrides=spark_conf_overrides,
        ),
    )


def _single_db_config(config: Any) -> SingleDbWriteConfig:
    if isinstance(config, SingleDbWriteConfig):
        return config
    if getattr(config, "num_dbs", 1) != 1:
        raise ConfigValidationError("single-db writer requires num_dbs=1")
    return SingleDbWriteConfig(
        storage=config.storage,
        output=config.output,
        manifest=config.manifest,
        kv=config.kv,
        retry=config.retry,
        rate_limits=config.rate_limits,
        observability=config.observability,
        lifecycle=config.lifecycle,
        vector=config.vector_spec,
    )


def write_spark_single_db(
    df: Any,
    config: Any,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_keys: bool = True,
    num_partitions: int | None = None,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    prefetch_partitions: bool = True,
    cache_input: bool = True,
    storage_level: Any | None = None,
    spark_conf_overrides: dict[str, str] | None = None,
) -> BuildResult:
    from shardyfusion.writer.spark.single_db_writer import write_single_db

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_single_db(
        df,
        _single_db_config(config),
        ColumnWriteInput(key_col=key_col, value_spec=value_spec),
        SingleDbWriteOptions(
            sort_keys=sort_keys,
            num_partitions=num_partitions,
            prefetch_partitions=prefetch_partitions,
            cache_input=cache_input,
            storage_level=storage_level,
            spark_conf_overrides=spark_conf_overrides,
        ),
    )


def write_dask_single_db(
    ddf: Any,
    config: Any,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_keys: bool = True,
    num_partitions: int | None = None,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    prefetch_partitions: bool = True,
    cache_input: bool = True,
) -> BuildResult:
    from shardyfusion.writer.dask.single_db_writer import write_single_db

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_single_db(
        ddf,
        _single_db_config(config),
        ColumnWriteInput(key_col=key_col, value_spec=value_spec),
        SingleDbWriteOptions(
            sort_keys=sort_keys,
            num_partitions=num_partitions,
            prefetch_partitions=prefetch_partitions,
            cache_input=cache_input,
        ),
    )


def write_ray_single_db(
    ds: Any,
    config: Any,
    *,
    key_col: str,
    value_spec: ValueSpec,
    sort_keys: bool = True,
    num_partitions: int | None = None,
    max_writes_per_second: float | None = None,
    max_write_bytes_per_second: float | None = None,
    prefetch_partitions: bool = True,
    cache_input: bool = True,
) -> BuildResult:
    from shardyfusion.writer.ray.single_db_writer import write_single_db

    _set_kv_rate_limits(
        config,
        max_writes_per_second=max_writes_per_second,
        max_write_bytes_per_second=max_write_bytes_per_second,
    )
    return write_single_db(
        ds,
        _single_db_config(config),
        ColumnWriteInput(key_col=key_col, value_spec=value_spec),
        SingleDbWriteOptions(
            sort_keys=sort_keys,
            num_partitions=num_partitions,
            prefetch_partitions=prefetch_partitions,
            cache_input=cache_input,
        ),
    )


def _vector_input(
    *,
    vector_col: str,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
) -> VectorColumnInput:
    return VectorColumnInput(
        vector_col=vector_col,
        id_col=id_col,
        payload_cols=payload_cols,
        shard_id_col=shard_id_col,
        routing_context_cols=routing_context_cols,
    )


def write_spark_vector_sharded(
    df: Any,
    config: Any,
    *,
    vector_col: str,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
    verify_routing: bool = True,
) -> BuildResult:
    from shardyfusion.writer.spark.vector_writer import write_sharded

    return write_sharded(
        df,
        config,
        _vector_input(
            vector_col=vector_col,
            id_col=id_col,
            payload_cols=payload_cols,
            shard_id_col=shard_id_col,
            routing_context_cols=routing_context_cols,
        ),
        VectorWriteOptions(verify_routing=verify_routing),
    )


def write_dask_vector_sharded(
    ddf: Any,
    config: Any,
    *,
    vector_col: str,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
    verify_routing: bool = True,
) -> BuildResult:
    from shardyfusion.writer.dask.vector_writer import write_sharded

    return write_sharded(
        ddf,
        config,
        _vector_input(
            vector_col=vector_col,
            id_col=id_col,
            payload_cols=payload_cols,
            shard_id_col=shard_id_col,
            routing_context_cols=routing_context_cols,
        ),
        VectorWriteOptions(verify_routing=verify_routing),
    )


def write_ray_vector_sharded(
    ds: Any,
    config: Any,
    *,
    vector_col: str,
    id_col: str | None = None,
    payload_cols: list[str] | None = None,
    shard_id_col: str | None = None,
    routing_context_cols: dict[str, str] | None = None,
    verify_routing: bool = True,
) -> BuildResult:
    from shardyfusion.writer.ray.vector_writer import write_sharded

    return write_sharded(
        ds,
        config,
        _vector_input(
            vector_col=vector_col,
            id_col=id_col,
            payload_cols=payload_cols,
            shard_id_col=shard_id_col,
            routing_context_cols=routing_context_cols,
        ),
        VectorWriteOptions(verify_routing=verify_routing),
    )
