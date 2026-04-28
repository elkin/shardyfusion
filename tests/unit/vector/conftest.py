"""Shared fixtures for vector unit tests."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def _patch_obstore_backend():
    """Patch obstore put/get to prevent real S3 calls in tests that need it."""
    from unittest.mock import MagicMock

    with (
        patch("obstore.put", MagicMock()),
        patch("obstore.put_async", MagicMock()),
        patch("obstore.get", MagicMock()),
        patch("obstore.get_async", MagicMock()),
        patch("obstore.list", MagicMock(return_value=iter([]))),
        patch(
            "obstore.list_with_delimiter",
            MagicMock(return_value={"common_prefixes": [], "objects": []}),
        ),
        patch("obstore.delete", MagicMock()),
    ):
        yield


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def sample_vectors(rng: np.random.Generator) -> np.ndarray:
    """100 vectors of dim 32."""
    return rng.standard_normal((100, 32)).astype(np.float32)


@pytest.fixture
def sample_query(rng: np.random.Generator) -> np.ndarray:
    """Single query vector of dim 32."""
    return rng.standard_normal(32).astype(np.float32)
