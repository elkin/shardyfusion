"""Shared fixtures for Dask writer integration tests."""

from __future__ import annotations

import pytest

pytest.importorskip("dask", reason="requires writer-dask extra")
