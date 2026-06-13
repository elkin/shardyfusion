"""Tests for recording each shard's decompressed page-cache sidecar size.

The writer captures the exact ``len(body)`` of the v8 sidecar at ``seal()`` and
stores it in the sidecar header (``body_size`` at offset 5).  The adapter still
exposes the size via ``sidecar_decompressed_bytes()`` so a reader can size a
download before fetching + decompressing the sidecar, but the size is no longer
round-tripped through the manifest.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def _mock_backend() -> MagicMock:
    with patch("shardyfusion.sqlite_adapter.ObstoreBackend") as m:
        inst = m.return_value
        inst.put = MagicMock()
        inst.head = MagicMock(return_value='"etag"')
        yield inst


class TestAdapterRecordsSize:
    def test_seal_records_exact_decompressed_size(
        self, tmp_path: Path, _mock_backend: MagicMock
    ) -> None:
        pytest.importorskip("apsw")
        zstandard = pytest.importorskip("zstandard")
        from shardyfusion.sqlite_adapter import (
            _SIDECAR_FIXED_PREFIX_SIZE,
            _SIDECAR_TAG_LEN_OFFSET,
            SqliteAdapter,
        )

        local_dir = tmp_path / "shard"
        with SqliteAdapter(db_url="s3://bucket/shard", local_dir=local_dir) as adapter:
            # Enough rows that the kv btree has at least one interior page.
            adapter.write_batch(
                [(i.to_bytes(8, "big"), b"v" * 32) for i in range(5000)]
            )
            adapter.seal()
            size = adapter.sidecar_decompressed_bytes()

        assert isinstance(size, int)
        assert size > 0
        # The recorded size equals the decompressed body of the uploaded sidecar.
        keys = [c.args[0] for c in _mock_backend.put.call_args_list]
        idx = next(i for i, k in enumerate(keys) if k.endswith("/shard.sidecar"))
        blob = _mock_backend.put.call_args_list[idx].args[1]
        tag_len = blob[_SIDECAR_TAG_LEN_OFFSET]
        body = zstandard.ZstdDecompressor().decompress(
            blob[_SIDECAR_FIXED_PREFIX_SIZE + tag_len :]
        )
        assert size == len(body)

    def test_no_size_when_emit_sidecar_false(
        self, tmp_path: Path, _mock_backend: MagicMock
    ) -> None:
        from shardyfusion.sqlite_adapter import SqliteAdapter

        local_dir = tmp_path / "shard"
        with SqliteAdapter(
            db_url="s3://bucket/shard", local_dir=local_dir, emit_sidecar=False
        ) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.seal()
            assert adapter.sidecar_decompressed_bytes() is None
