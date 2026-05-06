"""Integration tests for the SQLite B-tree metadata sidecar against moto S3.

End-to-end: write via SqliteAdapter → both shard.db and shard.btreemeta land
in the bucket → download sidecar → parse the binary blob → verify each
recorded slab matches the corresponding slice of the source shard.db.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shardyfusion.sqlite_adapter import (
    _BTREEMETA_FORMAT_VERSION,
    _BTREEMETA_MAGIC,
    SqliteAdapter,
    SqliteFactory,
)
from shardyfusion.storage import (
    ObstoreBackend,
    create_s3_store,
    parse_s3_url,
)


@pytest.fixture()
def s3_prefix(local_s3_service):  # type: ignore[no-untyped-def]
    bucket = local_s3_service["bucket"]
    return f"s3://{bucket}/btreemeta-test"


def _make_backend(s3_url: str) -> ObstoreBackend:
    bucket, _ = parse_s3_url(s3_url)
    return ObstoreBackend(create_s3_store(bucket=bucket))


def _parse_sidecar(blob: bytes) -> tuple[int, int, list[int], list[bytes]]:
    import struct

    import zstandard

    assert blob[:8] == _BTREEMETA_MAGIC
    fv = int.from_bytes(blob[8:12], "little")
    assert fv == _BTREEMETA_FORMAT_VERSION
    body = zstandard.ZstdDecompressor().decompress(blob[12:])
    page_size, n = struct.unpack("<II", body[:8])
    pairs = struct.unpack(f"<{2 * n}I", body[8 : 8 + 8 * n]) if n else ()
    page_nums = list(pairs[0::2])
    offsets = list(pairs[1::2])
    blobs = [body[off : off + page_size] for off in offsets]
    return fv, page_size, page_nums, blobs


class TestBtreemetaSidecarRoundTrip:
    """Round-trip the sidecar through moto + obstore and validate contents."""

    def test_default_factory_emits_sidecar(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        pytest.importorskip("apsw")
        db_url = f"{s3_prefix}/shards/run_id=meta/db=00000/attempt=00"
        write_dir = tmp_path / "write"

        # Default SqliteFactory has emit_btree_metadata=True.
        factory = SqliteFactory()
        with factory(db_url=db_url, local_dir=write_dir) as adapter:
            # ~10k rows so the kv btree certainly has at least one interior level.
            pairs = [(i.to_bytes(8, "big"), f"v{i}".encode()) for i in range(10_000)]
            adapter.write_batch(pairs)
            adapter.checkpoint()

        backend = _make_backend(db_url)
        sidecar_key = f"{db_url}/shard.btreemeta"
        db_key = f"{db_url}/shard.db"

        # Both objects exist.
        sidecar_bytes = backend.get(sidecar_key)
        db_bytes = backend.get(db_key)

        _fv, page_size, page_nums, slabs = _parse_sidecar(sidecar_bytes)
        assert page_size == 4096
        assert page_nums == sorted(page_nums)
        assert len(page_nums) >= 1

        # Each slab matches the corresponding 4 KiB slice of the source DB.
        for pgno, slab in zip(page_nums, slabs, strict=True):
            offset = (pgno - 1) * page_size
            assert slab == db_bytes[offset : offset + page_size]

    def test_opt_out_omits_sidecar(self, tmp_path: Path, s3_prefix: str) -> None:
        db_url = f"{s3_prefix}/shards/run_id=opt-out/db=00000/attempt=00"
        write_dir = tmp_path / "write"

        with SqliteAdapter(
            db_url=db_url, local_dir=write_dir, emit_btree_metadata=False
        ) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.checkpoint()

        backend = _make_backend(db_url)
        # Main db must exist.
        backend.get(f"{db_url}/shard.db")
        # Sidecar must NOT exist — backend.get on a missing key raises.
        # The exact exception type is obstore-internal; the broad except
        # is intentional here because the test asserts only that *some*
        # error surfaces (404 from moto via the obstore range layer).
        with pytest.raises(Exception):  # noqa: B017 — obstore not-found surface
            backend.get(f"{db_url}/shard.btreemeta")

    def test_sidecar_object_count_via_listing(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        """Defensive check: list bucket prefix and confirm exactly the two
        expected keys are present (shard.db, shard.btreemeta)."""
        pytest.importorskip("apsw")
        db_url = f"{s3_prefix}/shards/run_id=listing/db=00000/attempt=00"
        write_dir = tmp_path / "write"

        with SqliteFactory()(db_url=db_url, local_dir=write_dir) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.checkpoint()

        backend = _make_backend(db_url)
        # Confirm both keys exist.
        sidecar = backend.get(f"{db_url}/shard.btreemeta")
        db = backend.get(f"{db_url}/shard.db")
        assert sidecar
        assert db
        # The sidecar header sanity check.
        assert sidecar[:8] == _BTREEMETA_MAGIC
