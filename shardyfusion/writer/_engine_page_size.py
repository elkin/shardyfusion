"""Engine-percentile page_size substitution for distributed writers.

When ``KeyValueWriteConfig.profile_value_sizes_for_page_size`` is set,
the distributed writer samples value-bytes from the input, computes a
95th-percentile, picks a SQLite ``page_size`` via
:func:`shardyfusion.sqlite_page_size.recommend_page_size_for_cells`, and
rebuilds the adapter factory with that size before partition dispatch.

The helper takes an already-collected list of encoded value-byte
samples; the engine writer is responsible for the engine-native
sampling step (``DataFrame.limit(N).collect()`` /
``ddf.head(N)`` / ``ds.take(N)``).  Sampling rather than full
aggregation keeps the cost negligible — page_size has only five
discrete options so a percentile sample of ~1000 rows is precise enough
to pin one down.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

from shardyfusion._writer_core import kv_inner_factory
from shardyfusion.config import BaseShardedWriteConfig
from shardyfusion.errors import ConfigValidationError
from shardyfusion.logging import get_logger, log_event
from shardyfusion.sqlite_page_size import (
    VEC_INDEX_ROWID_MAX_BYTES,
    CellShape,
    recommend_page_size_for_cells,
)

_logger = get_logger(__name__)

# Default sample size for engine-percentile page_size picking.  Large
# enough that a single outlier doesn't shift the p95 estimate by an
# entire page-size bucket; small enough that .collect()/.take() costs
# a few ms on any backend.
DEFAULT_ENGINE_PROFILE_SAMPLE_SIZE: int = 1000

# Conservative default for the kv cell's max key bytes when the engine
# sample does not observe key widths.  Covers U64BE / U64LE / typical
# UTF-8 keys; workloads with wider keys (long composite keys, URL keys,
# >64 B blobs) should pin ``page_size`` explicitly on the factory.
_ENGINE_DEFAULT_KV_KEY_BYTES: int = 64


def maybe_apply_engine_page_size(
    config: BaseShardedWriteConfig,
    *,
    value_byte_samples: list[int],
    writer_kind: str,
) -> None:
    """Substitute ``config.kv.adapter_factory.page_size`` from a percentile sample.

    No-op when the flag is unset, when the sample is empty, or when the
    factory does not expose a ``page_size`` attribute (only the SQLite
    factories do).  Otherwise rebuilds the factory via
    :func:`dataclasses.replace` and reassigns it on ``config.kv``.

    ``value_byte_samples`` is a list of ``len(encoded_value_bytes)`` —
    integers, already collected to the driver.  ``writer_kind`` is a
    free-form string ("spark"/"dask"/"ray") used only for logging.

    For unified KV+vector factories (``SqliteVecFactory`` directly or
    wrapped in ``CompositeFactory`` with a vector-aware inner) the
    factory's :meth:`vec_payload_bytes_in_kv_db` reports the per-row
    cost the ``vec_index`` cell adds to the same .db; that cell shape is
    sized alongside the kv cell so the smaller page-size choice still
    fits the embedding inline.

    Caveat: key bytes are not observed by the engine sample.  The kv
    cell is sized with :data:`_ENGINE_DEFAULT_KV_KEY_BYTES` (64 B);
    workloads with wider keys should pin ``page_size`` explicitly on
    the factory.  The ``page_size="auto"`` VACUUM path measures actual
    key widths from the written DB, so the two strategies can pick
    different sizes on wide-key workloads — this is expected and
    intentional.

    The flag ``profile_value_sizes_for_page_size`` is cleared on every
    success or no-op return path so downstream re-validation does not
    trip the mutual-exclusion check.  When the helper raises (e.g.
    factory cannot be rebuilt), the flag is intentionally LEFT set so
    a caller's retry sees the user's original opt-in.
    """

    if not getattr(config.kv, "profile_value_sizes_for_page_size", False):
        return

    if not value_byte_samples:
        config.kv.profile_value_sizes_for_page_size = False
        return

    from shardyfusion.composite_adapter import CompositeFactory

    factory = config.kv.adapter_factory
    inner = kv_inner_factory(factory)
    if inner is None or not hasattr(inner, "page_size"):
        # Non-SQLite backends silently ignore the flag — the alternative
        # (raising) would surprise users who switch backends without
        # touching their kv config.
        config.kv.profile_value_sizes_for_page_size = False
        return

    sorted_sizes = sorted(int(n) for n in value_byte_samples)
    # Nearest-rank percentile: rank = ceil(p * N) (1-indexed) → index =
    # rank - 1 (0-indexed).  Plain `int(...) - 1` rounds toward zero and
    # picks one rank below the true p95 whenever `p * N` is non-integer.
    idx = max(0, math.ceil(len(sorted_sizes) * 0.95) - 1)
    p95 = sorted_sizes[idx]

    cells = [CellShape(payload_bytes=p95, max_key_bytes=_ENGINE_DEFAULT_KV_KEY_BYTES)]
    vec_payload = (
        inner.vec_payload_bytes_in_kv_db()
        if hasattr(inner, "vec_payload_bytes_in_kv_db")
        else 0
    )
    if vec_payload > 0:
        # vec_index leaf cells use a small rowid varint — sizing them
        # against the kv key width would over-budget the embedding and
        # bump one page-size bucket on a narrow band of (dim, key) combos.
        cells.append(
            CellShape(
                payload_bytes=vec_payload, max_key_bytes=VEC_INDEX_ROWID_MAX_BYTES
            )
        )

    effective_p95 = max(p95, vec_payload)
    try:
        target = recommend_page_size_for_cells(cells)
    except ConfigValidationError as exc:
        config.kv.profile_value_sizes_for_page_size = False
        log_event(
            "engine_page_size_recommend_failed",
            level=logging.WARNING,
            logger=_logger,
            error=str(exc),
            cells=[(c.payload_bytes, c.max_key_bytes) for c in cells],
        )
        return

    current = getattr(inner, "page_size", None)
    if current == target:
        config.kv.profile_value_sizes_for_page_size = False
        # Emit the picker event with ``no_change=True`` so operators can
        # tell "picker ran and accepted the existing size" from "picker
        # never ran" (flag still set) and "picker errored"
        # (engine_page_size_recommend_failed).
        log_event(
            "engine_page_size_picked",
            level=logging.DEBUG,
            logger=_logger,
            writer_kind=writer_kind,
            sample_size=len(sorted_sizes),
            p95_value_bytes=int(p95),
            vec_payload_bytes=int(vec_payload),
            effective_p95_bytes=int(effective_p95),
            from_page_size=str(current),
            to_page_size=int(target),
            no_change=True,
        )
        return

    try:
        # Protocol typing can't see the dataclass shape of concrete
        # factories; the TypeError below is the runtime guard.
        new_inner = dataclasses.replace(inner, page_size=target)  # type: ignore[type-var]
    except TypeError as exc:
        # Factory is not a dataclass; do not silently fail — the user
        # explicitly asked for engine-percentile picking and we can't
        # honour it.  Leave the flag set so a retry path with a
        # dataclass-friendly factory still profiles.
        raise ConfigValidationError(
            f"profile_value_sizes_for_page_size=True is set but the "
            f"adapter factory {type(inner).__name__!r} cannot be rebuilt "
            "with a different page_size.  Use a SqliteFactory or "
            "SqliteVecFactory, or pass page_size explicitly."
        ) from exc

    if isinstance(factory, CompositeFactory):
        try:
            new_factory = dataclasses.replace(factory, kv_factory=new_inner)
        except TypeError as exc:
            raise ConfigValidationError(
                f"Cannot rebuild wrapper {type(factory).__name__!r} with the "
                "engine-picked page_size; CompositeFactory must be a dataclass."
            ) from exc
    else:
        new_factory = new_inner
    config.kv.adapter_factory = new_factory
    # Clear the flag only AFTER the substitution succeeds so a caller's
    # retry path (after a raise from dataclasses.replace above) still
    # sees the user's original opt-in.
    config.kv.profile_value_sizes_for_page_size = False
    log_event(
        "engine_page_size_picked",
        level=logging.DEBUG,
        logger=_logger,
        writer_kind=writer_kind,
        sample_size=len(sorted_sizes),
        p95_value_bytes=int(p95),
        vec_payload_bytes=int(vec_payload),
        effective_p95_bytes=int(effective_p95),
        from_page_size=str(current),
        to_page_size=int(target),
    )


def collect_value_byte_samples(
    *,
    rows: list[Any],
    value_spec: Any,
) -> list[int]:
    """Encode a list of driver-side rows via ``value_spec`` and return sizes.

    Errors in a single row's encoder are logged and the row is skipped
    — the picker only needs a representative sample, not perfect
    coverage.  If ``rows`` was non-empty but every encode raised, a
    WARNING fires so operators can diagnose "opted in, got no signal."
    A workload where every value legitimately encodes to ``b""`` does
    NOT warn — zero bytes is itself a valid signal (the picker will
    correctly pick the smallest page size).

    Raises ``TypeError`` if ``value_spec.encode`` returns something
    other than ``bytes``/``bytearray``/``memoryview`` — a protocol
    violation that would silently miscount (``len(str)`` is code
    points, not bytes) if not caught.
    """
    sizes: list[int] = []
    encode_failures = 0
    for row in rows:
        try:
            encoded = value_spec.encode(row)
        except Exception as exc:  # pragma: no cover - defensive
            encode_failures += 1
            log_event(
                "engine_page_size_sample_encode_failed",
                level=logging.DEBUG,
                logger=_logger,
                error=str(exc),
            )
            continue
        if not isinstance(encoded, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"value_spec.encode must return bytes-like, got "
                f"{type(encoded).__name__}; len() on a str counts code "
                "points and would silently undersize the page."
            )
        sizes.append(len(encoded))
    if rows and encode_failures == len(rows):
        log_event(
            "engine_page_size_all_samples_failed",
            level=logging.WARNING,
            logger=_logger,
            row_count=len(rows),
        )
    return sizes
