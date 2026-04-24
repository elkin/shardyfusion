from __future__ import annotations

import builtins
from typing import Any
from unittest.mock import patch

import pytest

from shardyfusion.async_manifest_store import AsyncS3ManifestStore


@pytest.mark.asyncio
async def test_async_manifest_store_defers_aiobotocore_import() -> None:
    original_import = builtins.__import__

    def fail_aiobotocore(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("aiobotocore"):
            raise ImportError("No module named 'aiobotocore'")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fail_aiobotocore):
        store = AsyncS3ManifestStore("s3://bucket/prefix")

        with pytest.raises(ImportError, match="aiobotocore"):
            await store.load_current()
