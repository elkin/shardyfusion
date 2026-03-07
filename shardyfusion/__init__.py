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
from .manifest_readers import (
    DefaultS3ManifestReader,
    FunctionManifestReader,
    ManifestReader,
    parse_json_manifest,
)
from .metrics import MetricEvent, MetricsCollector
from .publish import DefaultS3Publisher, ManifestPublisher
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
    "DefaultS3Publisher",
    "DefaultS3ManifestReader",
    "DbAdapter",
    "DbAdapterFactory",
    "FunctionManifestReader",
    "JsonManifestBuilder",
    "KeyEncoding",
    "ManifestArtifact",
    "ManifestBuilder",
    "ManifestShardingSpec",
    "ManifestReader",
    "ManifestPublisher",
    "ManifestOptions",
    "OutputOptions",
    "RequiredBuildMeta",
    "RequiredShardMeta",
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
