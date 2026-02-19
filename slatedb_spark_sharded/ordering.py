"""Shared helpers for deterministic ordered comparisons."""

from __future__ import annotations

from typing import Any, cast


def compare_ordered(
    left: object,
    right: object,
    *,
    mismatch_message: str,
) -> int:
    """Return -1, 0, 1 for ordered values, raising on non-comparable pairs."""

    lhs = cast(Any, left)
    rhs = cast(Any, right)
    try:
        if lhs < rhs:
            return -1
        if lhs > rhs:
            return 1
        return 0
    except TypeError as exc:
        raise ValueError(mismatch_message) from exc
