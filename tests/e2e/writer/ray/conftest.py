"""Shared fixtures for Ray writer e2e tests."""

from __future__ import annotations

import pytest

ray = pytest.importorskip("ray", reason="requires writer-ray extra")


@pytest.fixture(scope="session", autouse=True)
def _ray_init():
    """Initialize Ray once per test session."""
    ray.init(num_cpus=2, ignore_reinit_error=True)
    yield
    ray.shutdown()
