"""Public API for slatedb_spark_sharded."""

from .config import (
    ManifestOptions,
    OutputOptions,
    WriteConfig,
)
from .errors import (
    ManifestParseError,
    ReaderStateError,
)
from .logging import FailureSeverity
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
from .publish import DefaultS3Publisher, ManifestPublisher
from .reader import SlateDbReaderFactory, SlateShardedReader
from .routing import SnapshotRouter
from .serde import ValueSpec
from .sharding_types import KeyEncoding, ShardingSpec, ShardingStrategy
from .slatedb_adapter import (
    DbAdapter,
    DbAdapterFactory,
    SlateDbFactory,
)
from .type_defs import ShardReaderFactory

__all__ = [
    "FailureSeverity",
    "ManifestParseError",
    "ReaderStateError",
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
    "SlateShardedReader",
    "SlateDbReaderFactory",
    "SlateDbFactory",
    "SnapshotRouter",
    "ShardingSpec",
    "ShardingStrategy",
    "WriteConfig",
    "ValueSpec",
    "parse_json_manifest",
]
