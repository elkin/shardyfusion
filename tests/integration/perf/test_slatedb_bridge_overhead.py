"""Microbenchmarks for the sync→async slatedb.uniffi bridge overhead.

These tests are opt-in (``-m perf``) and excluded from the default CI run.
They verify that the cost of crossing the Python sync→uniffi-async
boundary in the hot read/write paths stays within an acceptable budget,
and that batching iterator reads (``iterator_chunk_size``) recovers most
of the per-call overhead vs. a naive per-row loop.

Failure thresholds are deliberately loose: they exist to catch order-of-
magnitude regressions, not microsecond drift. Tune them only when a real
slatedb upstream change moves the floor.

Run with::

    just perf
    just perf tests/integration/perf/test_slatedb_bridge_overhead.py::test_write_batch_throughput
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

import pytest

pytestmark = pytest.mark.perf

from shardyfusion._checkpoint_id import generate_checkpoint_id  # noqa: E402
from shardyfusion._slatedb_runtime import run_coro  # noqa: E402
from shardyfusion._slatedb_symbols import (  # noqa: E402
    get_flush_options_classes,
    get_writer_symbols,
)
from shardyfusion.reader._types import SlateDbReaderFactory  # noqa: E402
from shardyfusion.testing import open_slatedb_db  # noqa: E402
from shardyfusion.type_defs import Manifest  # noqa: E402

# ---------------------------------------------------------------------------
# Budgets (per-op wall-clock ceilings; measured locally, padded ~5x)
# ---------------------------------------------------------------------------

# Round-trip cost of a single sync write call going through the bridge.
# Local measurements sit around 30–50 µs; budget at 500 µs to absorb CI noise.
WRITE_OP_BUDGET_S = 5e-4

# Round-trip cost of a single sync get() call. Budget at 1 ms.
GET_OP_BUDGET_S = 1e-3

# Per-row cost when scanning with iterator chunking. Budget at 200 µs/row;
# without chunking we would expect ~1 ms+ per row.
SCAN_ROW_BUDGET_S = 2e-4


def _seed_db(db_url: str, n: int) -> str:
    """Populate ``db_url`` with ``n`` u64-be keys → b"v{i}" values."""
    _, _, write_batch_cls, _ = get_writer_symbols()
    flush_options_cls, flush_type_cls = get_flush_options_classes()

    db = open_slatedb_db(db_url)
    try:
        wb = write_batch_cls()
        for i in range(n):
            wb.put(i.to_bytes(8, "big", signed=False), f"v{i}".encode())
        run_coro(db.write(wb))
        run_coro(
            db.flush_with_options(flush_options_cls(flush_type=flush_type_cls.WAL))
        )
    finally:
        run_coro(db.shutdown())
    return generate_checkpoint_id()


def test_write_batch_throughput(tmp_path: Path) -> None:
    """A 1 000-row WriteBatch must complete well under the per-op budget × N."""
    db_url = f"file://{(tmp_path / 'db').as_posix()}"
    n = 1_000

    start = time.perf_counter()
    _seed_db(db_url, n)
    elapsed = time.perf_counter() - start

    per_op = elapsed / n
    assert per_op < WRITE_OP_BUDGET_S, (
        f"WriteBatch per-op cost {per_op * 1e6:.1f} µs exceeds budget "
        f"{WRITE_OP_BUDGET_S * 1e6:.0f} µs (total {elapsed * 1000:.1f} ms for {n} rows)"
    )


def test_point_get_overhead(tmp_path: Path) -> None:
    """Sync ``reader.get()`` round-trip must stay under the per-op budget."""
    db_url = f"file://{(tmp_path / 'db').as_posix()}"
    n = 200
    _seed_db(db_url, n)

    factory = SlateDbReaderFactory()
    handle = factory(
        db_url=db_url,
        local_dir=tmp_path / "cache",
        checkpoint_id=None,
        manifest=cast(Manifest, None),
    )
    try:
        # Warm one call so any first-touch JIT/import cost does not skew us.
        handle.get((0).to_bytes(8, "big", signed=False))

        iters = 100
        start = time.perf_counter()
        for i in range(iters):
            value = handle.get((i % n).to_bytes(8, "big", signed=False))
            assert value is not None
        elapsed = time.perf_counter() - start
    finally:
        handle.close()

    per_op = elapsed / iters
    assert per_op < GET_OP_BUDGET_S, (
        f"Point-get per-op cost {per_op * 1e6:.1f} µs exceeds budget "
        f"{GET_OP_BUDGET_S * 1e6:.0f} µs ({iters} ops in {elapsed * 1000:.1f} ms)"
    )


def test_chunked_scan_amortizes_bridge_cost(tmp_path: Path) -> None:
    """Iterator chunking (default 1024) must keep per-row cost below the budget.

    Without ``iterator_chunk_size`` the per-row sync→async bridge cost would
    dominate (~1 ms/row). With chunking we amortize one bridge crossing
    across ``chunk_size`` rows.
    """
    db_url = f"file://{(tmp_path / 'db').as_posix()}"
    n = 5_000
    _seed_db(db_url, n)

    factory = SlateDbReaderFactory(iterator_chunk_size=1024)
    handle = factory(
        db_url=db_url,
        local_dir=tmp_path / "cache",
        checkpoint_id=None,
        manifest=cast(Manifest, None),
    )
    try:
        start = time.perf_counter()
        rows = 0
        for _ in handle.scan_iter(
            start=(0).to_bytes(8, "big", signed=False),
            end=(n).to_bytes(8, "big", signed=False),
        ):
            rows += 1
        elapsed = time.perf_counter() - start
    finally:
        handle.close()

    assert rows == n
    per_row = elapsed / n
    assert per_row < SCAN_ROW_BUDGET_S, (
        f"Chunked-scan per-row cost {per_row * 1e6:.1f} µs exceeds budget "
        f"{SCAN_ROW_BUDGET_S * 1e6:.0f} µs ({n} rows in {elapsed * 1000:.1f} ms)"
    )
