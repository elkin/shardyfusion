"""Helpers for picking SQLite ``page_size`` based on value-size hints.

Pure, dependency-free utilities used by the writer adapters and by
distributed-writer pre-aggregation paths.  The recommendation is the
smallest supported ``page_size`` whose ``WITHOUT ROWID`` inline-payload
threshold accommodates the supplied value/key bytes plus a small safety
margin (the per-cell header is a few bytes).  Anything that does not
fit inline spills into an overflow chain — one extra S3 GET per page
in object-storage backed readers.
"""

from __future__ import annotations

from typing import Final

from .errors import ConfigValidationError

# SQLite supported page sizes, ascending. 512 is technically valid but
# is too small to be useful and we omit it.
SUPPORTED_PAGE_SIZES: Final[tuple[int, ...]] = (4096, 8192, 16384, 32768, 65536)

# Per-cell overhead inside a leaf: the cell header (varint payload size +
# varint rowid/key length when applicable) plus a small slack for the
# index-cell metadata. Empirically ~12 bytes covers a typical kv row.
_CELL_OVERHEAD_BYTES: Final[int] = 12


def inline_payload_threshold(page_size: int) -> int:
    """Maximum inline payload (key + value) for a ``WITHOUT ROWID`` leaf.

    Derived from SQLite's ``X = ((U - 12) * 64 / 255) - 23`` formula with
    ``U == page_size`` (no reserved bytes).  Anything larger spills to
    the overflow chain.
    """
    if page_size not in SUPPORTED_PAGE_SIZES:
        raise ConfigValidationError(
            f"unsupported page_size {page_size!r}; expected one of {SUPPORTED_PAGE_SIZES}"
        )
    # X = ((U-12) * 64 / 255) - 23, integer-truncated to match SQLite's C math.
    return ((page_size - 12) * 64 // 255) - 23


def recommend_page_size(
    *,
    p95_value_bytes: int,
    max_key_bytes: int = 64,
) -> int:
    """Smallest supported ``page_size`` whose inline threshold fits the value.

    Args:
        p95_value_bytes: A representative value size (typically the 95th
            percentile from the workload).  Non-negative; 0 returns the
            smallest supported size (4096).
        max_key_bytes: Conservative upper bound on key bytes — the inline
            threshold covers key + value, so we subtract this from the
            available budget.  Default 64 matches the U64BE-encoded keys
            plus headroom.

    Returns:
        One of :data:`SUPPORTED_PAGE_SIZES`.  Returns the largest
        supported size when the value is too big to fit inline at any
        size (so an overflow chain is unavoidable but is at least
        as short as possible).
    """
    if p95_value_bytes < 0:
        raise ConfigValidationError(
            f"p95_value_bytes must be >= 0, got {p95_value_bytes}"
        )
    if max_key_bytes < 0:
        raise ConfigValidationError(f"max_key_bytes must be >= 0, got {max_key_bytes}")

    required = int(p95_value_bytes) + int(max_key_bytes) + _CELL_OVERHEAD_BYTES
    for size in SUPPORTED_PAGE_SIZES:
        if inline_payload_threshold(size) >= required:
            return size
    return SUPPORTED_PAGE_SIZES[-1]
