"""Shared fixtures for vector unit tests."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _patch_obstore_backend():
    """Patch ObstoreBackend in vector._distributed to use MemoryBackend."""
    from shardyfusion.storage import MemoryBackend

    with patch(
        "shardyfusion.vector._distributed.ObstoreBackend",
        side_effect=lambda store: MemoryBackend(),
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
