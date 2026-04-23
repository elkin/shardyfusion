"""Ray test fixtures shared across unit, integration, and e2e test suites."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _ray_init():
    """Initialize Ray once per test session."""
    import ray

    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()
