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

from typing import Final, NamedTuple

from .errors import ConfigValidationError

# SQLite supported page sizes, ascending. 512 is technically valid but
# is too small to be useful and we omit it.
SUPPORTED_PAGE_SIZES: Final[tuple[int, ...]] = (4096, 8192, 16384, 32768, 65536)

# Per-cell overhead inside a leaf: the cell header (varint payload size +
# varint rowid/key length when applicable) plus a small slack for the
# index-cell metadata. Empirically ~12 bytes covers a typical kv row.
_CELL_OVERHEAD_BYTES: Final[int] = 12

# Max bytes for a SQLite int64 rowid varint — used as the conservative
# "key" width for vec_index leaf cells, which key on rowid rather than a
# user-supplied composite key.
VEC_INDEX_ROWID_MAX_BYTES: Final[int] = 9


class CellShape(NamedTuple):
    """One b-tree cell shape (payload + key budget) for page-size sizing.

    A SQLite file's inline-payload threshold is per-cell, not per-table,
    and different tables in the same file can have very different cell
    shapes (e.g. a wide-key ``WITHOUT ROWID`` kv table next to a narrow
    rowid-keyed ``vec_index`` leaf).  Compute one shape per table you
    care about, then pass the list to
    :func:`recommend_page_size_for_cells`.
    """

    payload_bytes: int
    max_key_bytes: int


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


def recommend_page_size_for_cells(cells: list[CellShape]) -> int:
    """Smallest supported page_size whose inline threshold fits every cell.

    Computes :func:`recommend_page_size` independently for each cell shape
    and returns the maximum.  This is the correct way to size a database
    that contains multiple tables with materially different cell shapes —
    e.g. a unified kv + vec_index file where the kv cell carries a wide
    user key and the vec_index leaf carries a small rowid varint.

    Raises ``ConfigValidationError`` if ``cells`` is empty: every caller
    has at least one shape to fit, and silently returning the default
    page size would mask the bug.
    """
    if not cells:
        raise ConfigValidationError(
            "recommend_page_size_for_cells requires at least one cell shape"
        )
    return max(
        recommend_page_size(
            p95_value_bytes=cell.payload_bytes,
            max_key_bytes=cell.max_key_bytes,
        )
        for cell in cells
    )
