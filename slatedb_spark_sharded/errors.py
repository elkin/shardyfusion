"""Custom exception hierarchy for slatedb_spark_sharded.

Failure classification
----------------------
Each exception carries a ``retryable`` flag so callers can decide whether
retrying the operation may help:

* Retryable errors (e.g. ``PublishManifestError``, ``PublishCurrentError``)
  may succeed on retry — typically transient infrastructure failures.
* Non-retryable errors (e.g. ``ConfigValidationError``, ``ManifestParseError``)
  indicate programmer/data errors that require a fix, not a retry.
"""


class SlatedbSparkShardedError(Exception):
    """Base exception for all package-specific errors."""

    retryable: bool = False


# ---------------------------------------------------------------------------
# Configuration / programmer errors  (never retryable)
# ---------------------------------------------------------------------------

class ConfigValidationError(SlatedbSparkShardedError):
    """Configuration failed validation."""

    retryable = False


class ShardAssignmentError(SlatedbSparkShardedError):
    """Rows could not be assigned to valid shard ids."""

    retryable = False


class ShardCoverageError(SlatedbSparkShardedError):
    """Partition results did not cover all expected shard ids."""

    retryable = False


# ---------------------------------------------------------------------------
# Runtime / infrastructure errors  (retryable varies)
# ---------------------------------------------------------------------------

class SlateDbApiError(SlatedbSparkShardedError):
    """SlateDB binding was unavailable or incompatible."""

    retryable = False


class ManifestBuildError(SlatedbSparkShardedError):
    """Manifest artifact creation failed."""

    retryable = False


class PublishManifestError(SlatedbSparkShardedError):
    """Manifest publish operation failed."""

    retryable = True


class PublishCurrentError(SlatedbSparkShardedError):
    """CURRENT pointer publish operation failed."""

    retryable = True


class ManifestParseError(SlatedbSparkShardedError):
    """Manifest or CURRENT pointer payload could not be parsed or is structurally invalid."""

    retryable = False


class ReaderStateError(SlatedbSparkShardedError):
    """Reader operation attempted in an invalid lifecycle state (closed, missing CURRENT)."""

    retryable = False


class S3TransientError(SlatedbSparkShardedError):
    """Transient S3 error that may succeed on retry (throttle, 500, 503, timeout)."""

    retryable = True
