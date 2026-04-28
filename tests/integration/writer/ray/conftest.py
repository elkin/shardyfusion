"""Shared fixtures for Ray writer integration tests."""

from __future__ import annotations

import pytest

pytest.importorskip("ray", reason="requires writer-ray-slatedb extra")

from tests.helpers.ray_fixtures import _ray_init  # noqa: F401
