"""Shared fixtures for Ray writer unit tests."""

from __future__ import annotations

import pytest
import ray


@pytest.fixture(scope="session", autouse=True)
def _ray_init():
    """Initialize Ray once per test session."""
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()
