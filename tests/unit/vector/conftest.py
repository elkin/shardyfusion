"""Shared fixtures for vector unit tests."""

from __future__ import annotations

import numpy as np
import pytest


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
