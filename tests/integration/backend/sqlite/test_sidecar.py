"""Integration tests for the SQLite page-cache sidecar against moto S3.

End-to-end: write via SqliteAdapter → both shard.db and shard.sidecar land in
the bucket → download the sidecar → parse the v8 blob → reconstruct each page
and verify it is byte-consistent with the source shard.db, that overflow chains
are recorded, and that the sidecar is bound to the live shard.db's object ETag.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shardyfusion.sqlite_adapter import (
    _SIDECAR_FORMAT_VERSION,
    _SIDECAR_MAGIC,
    SqliteAdapter,
    SqliteFactory,
)
from shardyfusion.storage import (
    ObstoreBackend,
    create_s3_store,
    parse_s3_url,
)
from tests.helpers.sidecar import parse_sidecar, reconstruct_page


@pytest.fixture()
def s3_prefix(local_s3_service):  # type: ignore[no-untyped-def]
    bucket = local_s3_service["bucket"]
    return f"s3://{bucket}/sidecar-test"


def _make_backend(s3_url: str) -> ObstoreBackend:
    bucket, _ = parse_s3_url(s3_url)
    return ObstoreBackend(create_s3_store(bucket=bucket))


def _parse_sidecar(
    blob: bytes,
) -> tuple[int, str | None, int, list[int], list[bytes], list[list[int]]]:
    """v8 sidecar as a positional tuple, via the shared helper."""
    p = parse_sidecar(blob)
    return p.version, p.db_tag, p.page_size, p.pagenos, p.stored_pages, p.chains


class TestSidecarRoundTrip:
    """Round-trip the sidecar through moto + obstore and validate contents."""

    def test_default_factory_emits_sidecar(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        pytest.importorskip("apsw")
        db_url = f"{s3_prefix}/shards/run_id=meta/db=00000/attempt=00"
        write_dir = tmp_path / "write"

        # Default SqliteFactory has emit_sidecar=True.
        with SqliteFactory()(db_url=db_url, local_dir=write_dir) as adapter:
            # ~10k rows so the kv btree certainly has at least one interior level.
            pairs = [(i.to_bytes(8, "big"), f"v{i}".encode()) for i in range(10_000)]
            adapter.write_batch(pairs)
            adapter.seal()

        backend = _make_backend(db_url)
        sidecar_bytes = backend.get(f"{db_url}/shard.sidecar")
        db_bytes = backend.get(f"{db_url}/shard.db")

        version, _tag, page_size, pagenos, stored, _ = _parse_sidecar(sidecar_bytes)
        assert version == _SIDECAR_FORMAT_VERSION == 8
        assert page_size == 4096
        assert pagenos == sorted(pagenos)
        assert len(pagenos) >= 1

        # Each stored page reconstructs to a full page byte-consistent with the
        # source DB (reconstruction only zeroes the gap SQLite ignores).
        for pgno, s in zip(pagenos, stored, strict=True):
            recon = reconstruct_page(s, pgno, page_size)
            assert len(recon) == page_size
            original = db_bytes[(pgno - 1) * page_size : pgno * page_size]
            for a, b in zip(original, recon, strict=True):
                assert b == a or b == 0

    def test_opt_out_omits_sidecar(self, tmp_path: Path, s3_prefix: str) -> None:
        db_url = f"{s3_prefix}/shards/run_id=opt-out/db=00000/attempt=00"
        write_dir = tmp_path / "write"

        with SqliteAdapter(
            db_url=db_url, local_dir=write_dir, emit_sidecar=False
        ) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.seal()

        backend = _make_backend(db_url)
        backend.get(f"{db_url}/shard.db")
        with pytest.raises(Exception):  # noqa: B017 — obstore not-found surface
            backend.get(f"{db_url}/shard.sidecar")

    def test_overflow_chains_recorded_in_sidecar(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        """Values bigger than the inline-payload threshold spill into overflow
        chains; the sidecar records every chain in traversal order so a reader
        can prefetch them in one parallel multi-range request."""
        pytest.importorskip("apsw")
        db_url = f"{s3_prefix}/shards/run_id=overflow/db=00000/attempt=00"
        write_dir = tmp_path / "write"

        with SqliteFactory()(db_url=db_url, local_dir=write_dir) as adapter:
            # 20 KB values at 4 KB pages → ~1 KB inline + 5 overflow pages.
            adapter.write_batch(
                [(i.to_bytes(8, "big"), b"x" * 20_000) for i in range(20)]
            )
            adapter.seal()

        backend = _make_backend(db_url)
        sidecar = backend.get(f"{db_url}/shard.sidecar")
        db_bytes = backend.get(f"{db_url}/shard.db")

        _, _, _, _, _, chains = _parse_sidecar(sidecar)
        assert len(chains) == 20
        assert {len(c) for c in chains} == {5}

        for chain in chains:
            for src, dst in zip(chain[:-1], chain[1:], strict=True):
                off = (src - 1) * 4096
                assert int.from_bytes(db_bytes[off : off + 4], "big") == dst

    def test_default_factory_writes_both_objects(
        self, tmp_path: Path, s3_prefix: str
    ) -> None:
        """A default ``SqliteFactory`` write produces both ``shard.db`` and
        ``shard.sidecar`` at the expected S3 keys, and the sidecar starts with
        the vendor-neutral ``SQPC`` magic header."""
        pytest.importorskip("apsw")
        db_url = f"{s3_prefix}/shards/run_id=both/db=00000/attempt=00"
        write_dir = tmp_path / "write"

        with SqliteFactory()(db_url=db_url, local_dir=write_dir) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.seal()

        backend = _make_backend(db_url)
        sidecar = backend.get(f"{db_url}/shard.sidecar")
        db = backend.get(f"{db_url}/shard.db")
        assert sidecar
        assert db
        assert sidecar[:4] == _SIDECAR_MAGIC

    def test_sidecar_binds_live_db_etag(self, tmp_path: Path, s3_prefix: str) -> None:
        """The sidecar's embedded object tag matches the live ``shard.db``
        ETag, so a reader can bind the two — the writer uploads the ``.db``
        first, then stamps its ETag into the sidecar."""
        pytest.importorskip("apsw")
        db_url = f"{s3_prefix}/shards/run_id=bind/db=00000/attempt=00"
        write_dir = tmp_path / "write"

        with SqliteFactory()(db_url=db_url, local_dir=write_dir) as adapter:
            adapter.write_batch([(b"k", b"v")])
            adapter.seal()

        backend = _make_backend(db_url)
        sidecar = backend.get(f"{db_url}/shard.sidecar")
        _, tag, *_ = _parse_sidecar(sidecar)
        live_tag = backend.head(f"{db_url}/shard.db")
        assert tag is not None
        assert tag == live_tag
