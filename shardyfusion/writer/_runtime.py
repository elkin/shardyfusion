"""Shared runtime config for distributed partition writers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shardyfusion._writer_core import wrap_factory_for_vector
from shardyfusion.config import BaseShardedWriteConfig, ColumnWriteInput
from shardyfusion.credentials import CredentialProvider
from shardyfusion.metrics import MetricsCollector
from shardyfusion.serde import KeyEncoder, ValueSpec, make_key_encoder
from shardyfusion.sharding_types import KeyEncoding
from shardyfusion.slatedb_adapter import DbAdapterFactory, SlateDbFactory
from shardyfusion.type_defs import RetryConfig


@dataclass(slots=True)
class PartitionWriteRuntime:
    """Picklable runtime config for Spark, Dask, and Ray partition writers."""

    run_id: str
    s3_prefix: str
    shard_prefix: str
    db_path_template: str
    local_root: str
    key_col: str
    key_encoding: KeyEncoding
    key_encoder: KeyEncoder
    value_spec: ValueSpec
    batch_size: int
    adapter_factory: DbAdapterFactory
    credential_provider: CredentialProvider | None
    max_writes_per_second: float | None
    max_write_bytes_per_second: float | None = None
    sort_within_partitions: bool = False
    metrics_collector: MetricsCollector | None = None
    started: float = 0.0
    shard_retry: RetryConfig | None = None
    vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]] | None = (
        None
    )

    def __post_init__(self) -> None:
        assert self.run_id
        assert self.s3_prefix.startswith("s3://")
        assert self.shard_prefix
        assert "{db_id" in self.db_path_template
        assert self.local_root
        assert self.key_col
        assert isinstance(self.key_encoding, KeyEncoding)
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
        sort_within_partitions: bool = False,
        vector_fn: Callable[[Any], tuple[int | str, Any, dict[str, Any] | None]]
        | None = None,
    ) -> PartitionWriteRuntime:
        """Create worker runtime config from validated public inputs."""

        config.validate()
        input.validate()
        factory: DbAdapterFactory = config.adapter_factory or SlateDbFactory(
            credential_provider=config.credential_provider
        )
        if config.vector_spec is not None:
            factory = wrap_factory_for_vector(factory, config)

        return PartitionWriteRuntime(
            run_id=run_id,
            s3_prefix=config.s3_prefix,
            shard_prefix=config.output.shard_prefix,
            db_path_template=config.output.db_path_template,
            local_root=config.output.local_root,
            key_col=input.key_col,
            key_encoding=config.key_encoding,
            key_encoder=make_key_encoder(config.key_encoding),
            value_spec=input.value_spec,
            batch_size=config.batch_size,
            adapter_factory=factory,
            credential_provider=config.credential_provider,
            max_writes_per_second=config.rate_limits.max_writes_per_second,
            max_write_bytes_per_second=config.rate_limits.max_write_bytes_per_second,
            sort_within_partitions=sort_within_partitions,
            metrics_collector=config.metrics_collector,
            started=started,
            shard_retry=config.shard_retry,
            vector_fn=vector_fn,
        )
