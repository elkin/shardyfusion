from __future__ import annotations

import pytest

from shardyfusion.async_manifest_store import AsyncS3ManifestStore
from shardyfusion.storage import AsyncMemoryBackend


@pytest.mark.asyncio
async def test_async_manifest_store_load_current() -> None:
    backend = AsyncMemoryBackend()
    store = AsyncS3ManifestStore(backend, "s3://bucket/prefix")
    result = await store.load_current()
    assert result is None
