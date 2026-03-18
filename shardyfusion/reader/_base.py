"""Shared base class for ShardedReader and ConcurrentShardedReader."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Self

from shardyfusion._rate_limiter import RateLimiter
from shardyfusion.credentials import CredentialProvider
from shardyfusion.errors import ReaderStateError
from shardyfusion.logging import (
    FailureSeverity,
    get_logger,
    log_failure,
)
from shardyfusion.manifest import ManifestRef, RequiredShardMeta
from shardyfusion.manifest_store import ManifestStore, S3ManifestStore
from shardyfusion.metrics import MetricEvent, MetricsCollector
from shardyfusion.type_defs import (
    S3ConnectionOptions,
    ShardReader,
    ShardReaderFactory,
)

from ._types import SlateDbReaderFactory

_logger = get_logger(__name__)


class _BaseShardedReader:
    """Shared constructor and utilities for ShardedReader and ConcurrentShardedReader."""

    def __init__(
        self,
        *,
        s3_prefix: str,
        local_root: str,
        manifest_store: ManifestStore | None = None,
        current_name: str = "_CURRENT",
        reader_factory: ShardReaderFactory | None = None,
        slate_env_file: str | None = None,
        credential_provider: CredentialProvider | None = None,
        s3_connection_options: S3ConnectionOptions | None = None,
        max_workers: int | None = None,
        max_fallback_attempts: int = 3,
        metrics_collector: MetricsCollector | None = None,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.s3_prefix = s3_prefix
        self.local_root = local_root
        self.max_workers = max_workers
        self._max_fallback_attempts = max_fallback_attempts
        if max_workers is not None and (
            not isinstance(max_workers, int) or max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer (>= 1)")
        self._metrics = metrics_collector
        self._rate_limiter = rate_limiter

        if reader_factory is not None:
            self._reader_factory: ShardReaderFactory = reader_factory
        else:
            self._reader_factory = SlateDbReaderFactory(
                env_file=slate_env_file,
                credential_provider=credential_provider,
            )

        if manifest_store is not None:
            self._manifest_store = manifest_store
        else:
            self._manifest_store = S3ManifestStore(
                s3_prefix,
                current_name=current_name,
                credential_provider=credential_provider,
                s3_connection_options=s3_connection_options,
                metrics_collector=metrics_collector,
            )

        self._closed = False
        self._init_time = time.monotonic()

        # Shared executor for multi_get parallelism (created once, reused)
        if max_workers is not None and max_workers > 1:
            self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
                max_workers=max_workers
            )
        else:
            self._executor = None

    def _load_current(self) -> ManifestRef:
        """Load current manifest pointer, raising ReaderStateError if missing."""
        current = self._manifest_store.load_current()
        if current is None:
            log_failure(
                "reader_current_not_found",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                s3_prefix=self.s3_prefix,
            )
            raise ReaderStateError("CURRENT pointer not found")
        return current

    def _open_one_reader(self, shard: RequiredShardMeta) -> ShardReader:
        """Create local dir and open a single shard reader.

        Callers must only pass shards with ``db_url is not None``
        (empty shards use ``_NullShardReader`` instead).
        """
        assert shard.db_url is not None, f"shard {shard.db_id} has no db_url"
        local_path = Path(self.local_root) / f"shard={shard.db_id:05d}"
        local_path.mkdir(parents=True, exist_ok=True)

        return self._reader_factory(
            db_url=shard.db_url,
            local_dir=local_path,
            checkpoint_id=shard.checkpoint_id,
        )

    def _emit(self, event: MetricEvent, payload: dict[str, Any]) -> None:
        if self._metrics is not None:
            self._metrics.emit(event, payload)

    def _shutdown_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        raise NotImplementedError
