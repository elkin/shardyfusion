"""Public API for shardyfusion."""

from .config import (
    ManifestOptions,
    OutputOptions,
    WriteConfig,
)
from .errors import (
    ConfigValidationError,
    ManifestBuildError,
    ManifestParseError,
    ManifestStoreError,
    PublishCurrentError,
    PublishManifestError,
    ReaderStateError,
    S3TransientError,
    ShardAssignmentError,
    ShardCoverageError,
    ShardyfusionError,
    SlateDbApiError,
)
from .logging import FailureSeverity, get_logger
from .manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    CurrentPointer,
    JsonManifestBuilder,
    ManifestArtifact,
    ManifestBuilder,
    ManifestShardingSpec,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from .manifest_store import (
    InMemoryManifestStore,
    ManifestStore,
    S3ManifestStore,
    parse_json_manifest,
)
from .metrics import MetricEvent, MetricsCollector
from .reader import (
    ConcurrentShardedReader,
    ShardedReader,
    SlateDbReaderFactory,
    SnapshotInfo,
)
from .routing import SnapshotRouter
from .serde import ValueSpec
from .sharding_types import KeyEncoding, ShardingSpec, ShardingStrategy
from .slatedb_adapter import (
    DbAdapter,
    DbAdapterFactory,
    SlateDbFactory,
)
from .type_defs import S3ClientConfig, ShardReader, ShardReaderFactory

__all__ = [
    "ConfigValidationError",
    "FailureSeverity",
    "get_logger",
    "ManifestBuildError",
    "ManifestParseError",
    "ManifestStoreError",
    "MetricEvent",
    "MetricsCollector",
    "PublishCurrentError",
    "PublishManifestError",
    "ReaderStateError",
    "S3ClientConfig",
    "S3TransientError",
    "ShardAssignmentError",
    "ShardCoverageError",
    "ShardReader",
    "ShardyfusionError",
    "SlateDbApiError",
    "BuildResult",
    "BuildStats",
    "BuildDurations",
    "CurrentPointer",
    "DbAdapter",
    "DbAdapterFactory",
    "InMemoryManifestStore",
    "JsonManifestBuilder",
    "KeyEncoding",
    "ManifestArtifact",
    "ManifestBuilder",
    "ManifestShardingSpec",
    "ManifestStore",
    "ManifestOptions",
    "OutputOptions",
    "RequiredBuildMeta",
    "RequiredShardMeta",
    "S3ManifestStore",
    "ShardReaderFactory",
    "ConcurrentShardedReader",
    "ShardedReader",
    "SlateDbReaderFactory",
    "SlateDbFactory",
    "SnapshotInfo",
    "SnapshotRouter",
    "ShardingSpec",
    "ShardingStrategy",
    "WriteConfig",
    "ValueSpec",
    "parse_json_manifest",
]
