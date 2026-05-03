"""Shared runtime config for distributed partition writers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shardyfusion._writer_core import wrap_factory_for_vector
from shardyfusion.config import BaseShardedWriteConfig, ColumnWriteInput
from shardyfusion.metrics import MetricsCollector
from shardyfusion.serde import KeyEncoder, ValueSpec, make_key_encoder
from shardyfusion.slatedb_adapter import DbAdapterFactory, SlateDbFactory
from shardyfusion.type_defs import RetryConfig

_VectorFn = Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]]


@dataclass(slots=True)
class PartitionWriteRuntime:
    """Common picklable runtime config for distributed partition writers."""

    run_id: str
    s3_prefix: str
    shard_prefix: str
    db_path_template: str
    local_root: str
    key_col: str
    key_encoder: KeyEncoder
    value_spec: ValueSpec
    batch_size: int
    adapter_factory: DbAdapterFactory
    max_writes_per_second: float | None
    max_write_bytes_per_second: float | None = None
    metrics_collector: MetricsCollector | None = None
    started: float = 0.0
    vector_fn: _VectorFn | None = None

    def __post_init__(self) -> None:
        assert self.run_id
        assert self.s3_prefix.startswith("s3://")
        assert self.shard_prefix
        assert "{db_id" in self.db_path_template
        assert self.local_root
        assert self.key_col
        assert self.batch_size > 0
        assert self.adapter_factory is not None
        assert self.max_writes_per_second is None or self.max_writes_per_second > 0
        assert (
            self.max_write_bytes_per_second is None
            or self.max_write_bytes_per_second > 0
        )

    @staticmethod
    def from_public_config(
        *,
        config: BaseShardedWriteConfig,
        input: ColumnWriteInput,
        run_id: str,
        started: float = 0.0,
        vector_fn: _VectorFn | None = None,
    ) -> PartitionWriteRuntime:
        """Create worker runtime config from validated public inputs."""

        return PartitionWriteRuntime(
            **_common_runtime_kwargs(
                config=config,
                input=input,
                run_id=run_id,
                started=started,
                vector_fn=vector_fn,
            )
        )


@dataclass(slots=True)
class RetryingPartitionWriteRuntime(PartitionWriteRuntime):
    """Partition runtime for Dask and Ray writers with worker-side retries."""

    sort_within_partitions: bool = False
    shard_retry: RetryConfig | None = None

    @staticmethod
    def from_public_config(
        *,
        config: BaseShardedWriteConfig,
        input: ColumnWriteInput,
        run_id: str,
        started: float = 0.0,
        sort_within_partitions: bool = False,
        vector_fn: _VectorFn | None = None,
    ) -> RetryingPartitionWriteRuntime:
        """Create Dask/Ray worker runtime config from validated public inputs."""

        return RetryingPartitionWriteRuntime(
            **_common_runtime_kwargs(
                config=config,
                input=input,
                run_id=run_id,
                started=started,
                vector_fn=vector_fn,
            ),
            sort_within_partitions=sort_within_partitions,
            shard_retry=config.shard_retry,
        )


def _common_runtime_kwargs(
    *,
    config: BaseShardedWriteConfig,
    input: ColumnWriteInput,
    run_id: str,
    started: float,
    vector_fn: _VectorFn | None,
) -> dict[str, Any]:
    """Resolve common runtime constructor kwargs from public config/input."""

    config.validate()
    input.validate()
    factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory(
        credential_provider=config.credential_provider
    )
    if config.vector_spec is not None:
        factory = wrap_factory_for_vector(factory, config)

    return {
        "run_id": run_id,
        "s3_prefix": config.s3_prefix,
        "shard_prefix": config.output.shard_prefix,
        "db_path_template": config.output.db_path_template,
        "local_root": config.output.local_root,
        "key_col": input.key_col,
        "key_encoder": make_key_encoder(config.key_encoding),
        "value_spec": input.value_spec,
        "batch_size": config.batch_size,
        "adapter_factory": factory,
        "max_writes_per_second": config.rate_limits.max_writes_per_second,
        "max_write_bytes_per_second": config.rate_limits.max_write_bytes_per_second,
        "metrics_collector": config.metrics_collector,
        "started": started,
        "vector_fn": vector_fn,
    }
