"""Unit tests for the ``recommend_page_size`` helper and inline threshold.

The helper picks the smallest supported SQLite ``page_size`` whose
WITHOUT-ROWID inline-payload threshold accommodates a given p95 value
size plus a key-bytes allowance.  These tests cover the boundary cases
around each supported page size and a few representative inputs.
"""

from __future__ import annotations

import pytest

from shardyfusion.errors import ConfigValidationError
from shardyfusion.sqlite_page_size import (
    SUPPORTED_PAGE_SIZES,
    inline_payload_threshold,
    recommend_page_size,
)


class TestInlineThreshold:
    def test_known_values(self) -> None:
        # SQLite's `X = ((U-12)*64/255) - 23` with U = page_size.
        # Verified against the SQLite source for the canonical sizes.
        assert inline_payload_threshold(4096) == 1002
        assert inline_payload_threshold(8192) == 2030
        assert inline_payload_threshold(16384) == 4086
        assert inline_payload_threshold(32768) == 8198
        assert inline_payload_threshold(65536) == 16422

    def test_rejects_unsupported(self) -> None:
        with pytest.raises(ConfigValidationError):
            inline_payload_threshold(3000)


class TestRecommendPageSize:
    def test_small_value_picks_smallest(self) -> None:
        assert recommend_page_size(p95_value_bytes=100) == 4096

    def test_value_just_under_4k_threshold_stays_at_4k(self) -> None:
        # 1002 inline - 64 key - 12 overhead = 926 byte ceiling at 4k.
        assert recommend_page_size(p95_value_bytes=900) == 4096

    def test_value_above_4k_threshold_picks_8k(self) -> None:
        assert recommend_page_size(p95_value_bytes=1500) == 8192

    def test_value_above_8k_threshold_picks_16k(self) -> None:
        assert recommend_page_size(p95_value_bytes=4000) == 16384

    def test_value_above_16k_threshold_picks_32k(self) -> None:
        assert recommend_page_size(p95_value_bytes=8000) == 32768

    def test_value_above_32k_threshold_picks_64k(self) -> None:
        assert recommend_page_size(p95_value_bytes=12000) == 65536

    def test_value_too_big_for_any_page_clamps_to_largest(self) -> None:
        # 1 MB cannot fit inline at any supported page size; we pick the
        # largest to at least shorten the overflow chain.
        assert (
            recommend_page_size(p95_value_bytes=1_000_000) == SUPPORTED_PAGE_SIZES[-1]
        )

    def test_zero_value_returns_smallest(self) -> None:
        assert recommend_page_size(p95_value_bytes=0) == 4096

    def test_large_key_shifts_recommendation_up(self) -> None:
        # 800-byte value fits at 4 KB with a 64-byte key, but a 400-byte
        # key tips the needed inline budget over the 4 KB threshold.
        assert recommend_page_size(p95_value_bytes=800, max_key_bytes=64) == 4096
        assert recommend_page_size(p95_value_bytes=800, max_key_bytes=400) == 8192

    def test_rejects_negative_inputs(self) -> None:
        with pytest.raises(ConfigValidationError):
            recommend_page_size(p95_value_bytes=-1)
        with pytest.raises(ConfigValidationError):
            recommend_page_size(p95_value_bytes=100, max_key_bytes=-1)
