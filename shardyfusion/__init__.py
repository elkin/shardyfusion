"""Public API for shardyfusion."""

from ._rate_limiter import (
    AcquireResult,
    RateLimiter,
    ThreadSafeTokenBucket,
    TokenBucket,
)
from .async_manifest_store import (
    AsyncManifestStore,
    AsyncS3ManifestStore,
)
from .cel import CelColumn, CelType, cel_sharding, cel_sharding_by_columns
from .config import (
    ManifestOptions,
    OutputOptions,
    VectorSpec,
    WriteConfig,
)
from .credentials import (
    CredentialProvider,
    EnvCredentialProvider,
    S3Credentials,
    StaticCredentialProvider,
)
from .errors import (
    ConfigValidationError,
    DbAdapterError,
    ManifestBuildError,
    ManifestParseError,
    ManifestStoreError,
    PoolExhaustedError,
    PublishCurrentError,
    PublishManifestError,
    ReaderStateError,
    S3TransientError,
    ShardAssignmentError,
    ShardCoverageError,
    ShardWriteError,
    ShardyfusionError,
    SlateDbApiError,
)
from .logging import (
    FailureSeverity,
    JsonFormatter,
    LogContext,
    configure_logging,
    get_logger,
)
from .manifest import (
    SQLITE_MANIFEST_CONTENT_TYPE,
    BuildDurations,
    BuildResult,
    BuildStats,
    CurrentPointer,
    ManifestArtifact,
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
    SqliteManifestBuilder,
    WriterInfo,
)
from .manifest_store import (
    InMemoryManifestStore,
    ManifestStore,
    S3ManifestStore,
    SqliteShardLookup,
    load_sqlite_build_meta,
    parse_manifest_payload,
    parse_sqlite_manifest,
)
from .metrics import MetricEvent, MetricsCollector
from .reader import (
    AsyncShardedReader,
    AsyncShardReaderHandle,
    AsyncSlateDbReaderFactory,
    ConcurrentShardedReader,
    ReaderHealth,
    ShardDetail,
    ShardedReader,
    ShardReaderHandle,
    SlateDbReaderFactory,
    SnapshotInfo,
)
from .routing import ShardLookup, SnapshotRouter
from .run_registry import (
    InMemoryRunRegistry,
    RunRecord,
    RunRegistry,
    RunStatus,
    S3RunRegistry,
)
from .serde import ValueSpec
from .sharding_types import KeyEncoding, ShardingSpec, ShardingStrategy
from .slatedb_adapter import (
    DbAdapter,
    DbAdapterFactory,
    SlateDbFactory,
)
from .type_defs import (
    AsyncShardReader,
    AsyncShardReaderFactory,
    RetryConfig,
    S3ConnectionOptions,
    ShardReader,
    ShardReaderFactory,
)

__all__ = [
    "AcquireResult",
    "AsyncManifestStore",
    "AsyncS3ManifestStore",
    "AsyncShardReader",
    "AsyncShardReaderFactory",
    "AsyncShardReaderHandle",
    "AsyncShardedReader",
    "AsyncSlateDbReaderFactory",
    "BuildDurations",
    "BuildResult",
    "BuildStats",
    "CelColumn",
    "CelType",
    "ConcurrentShardedReader",
    "ConfigValidationError",
    "CredentialProvider",
    "DbAdapterError",
    "CurrentPointer",
    "DbAdapter",
    "DbAdapterFactory",
    "EnvCredentialProvider",
    "FailureSeverity",
    "InMemoryManifestStore",
    "InMemoryRunRegistry",
    "JsonFormatter",
    "KeyEncoding",
    "LogContext",
    "ManifestArtifact",
    "ManifestBuildError",
    "ManifestOptions",
    "ManifestParseError",
    "ManifestRef",
    "ManifestShardingSpec",
    "ManifestStore",
    "ManifestStoreError",
    "MetricEvent",
    "MetricsCollector",
    "OutputOptions",
    "ParsedManifest",
    "PoolExhaustedError",
    "PublishCurrentError",
    "PublishManifestError",
    "RateLimiter",
    "ReaderHealth",
    "ReaderStateError",
    "RequiredBuildMeta",
    "RequiredShardMeta",
    "RetryConfig",
    "RunRecord",
    "RunRegistry",
    "RunStatus",
    "S3ConnectionOptions",
    "S3Credentials",
    "SQLITE_MANIFEST_CONTENT_TYPE",
    "S3ManifestStore",
    "S3RunRegistry",
    "S3TransientError",
    "ShardAssignmentError",
    "ShardCoverageError",
    "ShardWriteError",
    "ShardDetail",
    "ShardReader",
    "ShardReaderFactory",
    "ShardReaderHandle",
    "ShardedReader",
    "ShardingSpec",
    "ShardLookup",
    "ShardingStrategy",
    "ShardyfusionError",
    "SlateDbApiError",
    "SlateDbFactory",
    "SlateDbReaderFactory",
    "SnapshotInfo",
    "SnapshotRouter",
    "SqliteManifestBuilder",
    "SqliteShardLookup",
    "StaticCredentialProvider",
    "ThreadSafeTokenBucket",
    "TokenBucket",
    "UnifiedShardedReader",
    "ValueSpec",
    "VectorSpec",
    "WriteConfig",
    "WriterInfo",
    "cel_sharding",
    "cel_sharding_by_columns",
    "configure_logging",
    "get_logger",
    "load_sqlite_build_meta",
    "parse_manifest_payload",
    "parse_sqlite_manifest",
]


def __getattr__(name: str):  # noqa: ANN202
    if name == "UnifiedShardedReader":
        from .reader.unified_reader import UnifiedShardedReader

        return UnifiedShardedReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
