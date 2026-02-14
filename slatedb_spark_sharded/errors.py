"""Custom exception hierarchy for slatedb_spark_sharded."""


class SlatedbSparkShardedError(Exception):
    """Base exception for all package-specific errors."""


class ConfigValidationError(SlatedbSparkShardedError):
    """Configuration failed validation."""


class ShardAssignmentError(SlatedbSparkShardedError):
    """Rows could not be assigned to valid shard ids."""


class ShardCoverageError(SlatedbSparkShardedError):
    """Partition results did not cover all expected shard ids."""


class SlateDbApiError(SlatedbSparkShardedError):
    """SlateDB binding was unavailable or incompatible."""


class ManifestBuildError(SlatedbSparkShardedError):
    """Manifest artifact creation failed."""


class PublishManifestError(SlatedbSparkShardedError):
    """Manifest publish operation failed."""


class PublishCurrentError(SlatedbSparkShardedError):
    """CURRENT pointer publish operation failed."""
