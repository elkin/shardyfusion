"""Custom exception hierarchy for shardyfusion.

Failure classification
----------------------
Each exception carries a ``retryable`` flag so callers can decide whether
retrying the operation may help:

* Retryable errors (e.g. ``PublishManifestError``, ``PublishCurrentError``)
  may succeed on retry — typically transient infrastructure failures.
* Non-retryable errors (e.g. ``ConfigValidationError``, ``ManifestParseError``)
  indicate programmer/data errors that require a fix, not a retry.
"""


class ShardyfusionError(Exception):
    """Base exception for all package-specific errors."""

    retryable: bool = False


# ---------------------------------------------------------------------------
# Configuration / programmer errors  (never retryable)
# ---------------------------------------------------------------------------


class ConfigValidationError(ShardyfusionError):
    """Configuration failed validation.

    Raised during ``WriteConfig.__post_init__`` or writer startup when
    parameters are invalid (e.g. ``num_dbs <= 0``, malformed ``s3_prefix``,
    unsupported sharding strategy for a given writer backend).
    """

    retryable = False


class ShardAssignmentError(ShardyfusionError):
    """Rows could not be assigned to valid shard IDs.

    Raised when routing verification detects a mismatch between
    framework-assigned shard IDs and Python routing.
    """

    retryable = False


class ShardCoverageError(ShardyfusionError):
    """Partition results did not cover all expected shard IDs.

    Raised after shard writes complete if the set of shard IDs in
    results doesn't match ``range(num_dbs)``.
    """

    retryable = False


# ---------------------------------------------------------------------------
# Runtime / infrastructure errors  (retryable varies)
# ---------------------------------------------------------------------------


class SlateDbApiError(ShardyfusionError):
    """SlateDB binding was unavailable or incompatible.

    Also raised when shard reader operations fail (e.g. during close).
    """

    retryable = False


class ManifestBuildError(ShardyfusionError):
    """Manifest artifact creation failed.

    Raised when the ``ManifestBuilder`` raises during ``build()``.
    """

    retryable = False


class PublishManifestError(ShardyfusionError):
    """Manifest publish operation failed.

    The manifest bytes could not be uploaded to S3. This is retryable
    because the failure is typically a transient S3 error.
    """

    retryable = True


class PublishCurrentError(ShardyfusionError):
    """CURRENT pointer publish operation failed.

    The manifest has already been published when this error is raised.
    Use ``manifest_ref`` to locate the published manifest for recovery
    (e.g. manually updating the CURRENT pointer).
    """

    retryable = True

    def __init__(self, message: str, *, manifest_ref: str | None = None) -> None:
        super().__init__(message)
        self.manifest_ref = manifest_ref


class ManifestParseError(ShardyfusionError):
    """Manifest or CURRENT pointer payload could not be parsed or is structurally invalid.

    Raised when JSON is malformed, required fields are missing, or
    structural invariants are violated (e.g. shard count mismatch).
    """

    retryable = False


class ReaderStateError(ShardyfusionError):
    """Reader operation attempted in an invalid lifecycle state (closed, missing CURRENT).

    Raised when calling ``get()``, ``multi_get()``, or ``refresh()``
    on a closed reader, or when the CURRENT pointer is not found.
    """

    retryable = False


class S3TransientError(ShardyfusionError):
    """Transient S3 error that may succeed on retry (throttle, 500, 503, timeout).

    Used internally by the storage layer's exponential-backoff retry logic.
    """

    retryable = True
