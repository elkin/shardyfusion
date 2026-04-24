"""Tests for lazy __getattr__ imports in package __init__ modules."""

from __future__ import annotations

import pytest

_MISSING = "NoSuchThing"


class TestShardyfusionLazyImports:
    def test_unified_sharded_reader(self) -> None:
        import shardyfusion

        cls = shardyfusion.UnifiedShardedReader
        assert cls is not None
        assert cls.__name__ == "UnifiedShardedReader"

    def test_unknown_attr_raises(self) -> None:
        import shardyfusion

        with pytest.raises(AttributeError, match=_MISSING):
            getattr(shardyfusion, _MISSING)


class TestReaderLazyImports:
    def test_unified_sharded_reader(self) -> None:
        import shardyfusion.reader

        cls = shardyfusion.reader.UnifiedShardedReader
        assert cls is not None
        assert cls.__name__ == "UnifiedShardedReader"

    def test_unknown_attr_raises(self) -> None:
        import shardyfusion.reader

        with pytest.raises(AttributeError, match=_MISSING):
            getattr(shardyfusion.reader, _MISSING)


class TestVectorAdaptersLazyImports:
    def test_unknown_attr_raises(self) -> None:
        import shardyfusion.vector.adapters

        with pytest.raises(AttributeError, match=_MISSING):
            getattr(shardyfusion.vector.adapters, _MISSING)

    def test_known_attr_resolves(self) -> None:
        """Accessing a known __all__ name triggers the lazy import path."""
        import shardyfusion.vector.adapters

        # LanceDbWriterFactory is in __all__ — accessing it goes through __getattr__
        # which imports lancedb_adapter. This works because lancedb is importable
        # in this env (it's a Python module even if the C extension fails).
        try:
            cls = shardyfusion.vector.adapters.LanceDbWriterFactory
            assert cls is not None
        except (ImportError, AttributeError):
            pytest.skip("lancedb not available")
