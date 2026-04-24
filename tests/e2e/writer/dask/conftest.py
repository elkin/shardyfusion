"""Shared fixtures for Dask writer e2e tests."""

from __future__ import annotations

import dask
import pytest

pytest.importorskip("dask", reason="requires writer-dask extra")


@pytest.fixture(autouse=True)
def _synchronous_scheduler():
    """Force synchronous Dask scheduler for deterministic test behavior."""
    with dask.config.set(scheduler="synchronous"):
        yield
