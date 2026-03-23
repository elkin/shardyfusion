"""Direct unit tests for compare_ordered() in ordering.py."""

from __future__ import annotations

import pytest

from shardyfusion.ordering import compare_ordered


class TestCompareOrdered:
    def test_less_than(self) -> None:
        assert compare_ordered(1, 2, mismatch_message="bad") == -1

    def test_equal(self) -> None:
        assert compare_ordered(5, 5, mismatch_message="bad") == 0

    def test_greater_than(self) -> None:
        assert compare_ordered(10, 3, mismatch_message="bad") == 1

    def test_strings_less(self) -> None:
        assert compare_ordered("abc", "xyz", mismatch_message="bad") == -1

    def test_strings_equal(self) -> None:
        assert compare_ordered("hello", "hello", mismatch_message="bad") == 0

    def test_strings_greater(self) -> None:
        assert compare_ordered("z", "a", mismatch_message="bad") == 1

    def test_incomparable_types_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="custom msg"):
            compare_ordered("a", 1, mismatch_message="custom msg")

    def test_none_vs_int_raises(self) -> None:
        with pytest.raises(ValueError, match="no compare"):
            compare_ordered(None, 1, mismatch_message="no compare")

    def test_bytes_comparison(self) -> None:
        assert compare_ordered(b"abc", b"abd", mismatch_message="bad") == -1

    def test_float_comparison(self) -> None:
        assert compare_ordered(1.5, 2.5, mismatch_message="bad") == -1
