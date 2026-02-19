"""Public API for slatedb_spark_sharded."""

from .config import (
    EngineOptions,
    ManifestOptions,
    OutputOptions,
    ShardingOptions,
    SlateDbConfig,
)
from .errors import (
    ManifestParseError,
    ReaderStateError,
)
from .manifest import (
    BuildDurations,
    BuildResult,
    BuildStats,
    CurrentPointer,
    JsonManifestBuilder,
    ManifestArtifact,
    ManifestBuilder,
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
from .reader import SlateShardedReader
from .routing import SnapshotRouter
from .serde import ValueSpec
from .sharding import ShardingSpec, ShardingStrategy

_writer_exports: list[str] = []
try:
    from .writer import (
        DataFrameCacheContext,
        SparkConfOverrideContext,
        write_sharded_slatedb,
    )

    _writer_exports = [
        "DataFrameCacheContext",
        "SparkConfOverrideContext",
        "write_sharded_slatedb",
    ]
except ImportError:
    # Writer APIs are unavailable when optional writer dependencies
    # (notably pyspark) are not installed.
    pass

__all__ = [
    "ManifestParseError",
    "ReaderStateError",
    "BuildResult",
    "BuildStats",
    "BuildDurations",
    "CurrentPointer",
    "DefaultS3Publisher",
    "DefaultS3ManifestReader",
    "FunctionManifestReader",
    "JsonManifestBuilder",
    "ManifestArtifact",
    "ManifestBuilder",
    "ManifestReader",
    "ManifestPublisher",
    "ManifestOptions",
    "OutputOptions",
    "RequiredBuildMeta",
    "RequiredShardMeta",
    "SlateShardedReader",
    "SnapshotRouter",
    "ShardingSpec",
    "ShardingOptions",
    "ShardingStrategy",
    "SlateDbConfig",
    "EngineOptions",
    "ValueSpec",
    "parse_json_manifest",
]
__all__.extend(_writer_exports)
