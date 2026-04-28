"""Shared fixtures for Spark writer unit tests."""

from __future__ import annotations

import pytest

pytest.importorskip("pyspark", reason="requires writer-spark-slatedb extra")
