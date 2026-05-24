"""Engine-percentile page_size substitution for distributed writers.

When ``KeyValueWriteConfig.profile_value_sizes_for_page_size`` is set,
the distributed writer samples value-bytes from the input, computes a
95th-percentile, picks a SQLite ``page_size`` via
:func:`shardyfusion.sqlite_page_size.recommend_page_size`, and rebuilds
the adapter factory with that size before partition dispatch.

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
from typing import Any

from shardyfusion.config import BaseShardedWriteConfig
from shardyfusion.errors import ConfigValidationError
from shardyfusion.logging import get_logger, log_event
from shardyfusion.sqlite_page_size import recommend_page_size

_logger = get_logger(__name__)

# Default sample size for engine-percentile page_size picking.  Large
# enough that a single outlier doesn't shift the p95 estimate by an
# entire page-size bucket; small enough that .collect()/.take() costs
# a few ms on any backend.
DEFAULT_ENGINE_PROFILE_SAMPLE_SIZE: int = 1000


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

    The mutation is on the user-supplied config; this matches existing
    convention (the writers similarly read and assume ownership of
    config-derived state during a single run).
    """

    if not getattr(config.kv, "profile_value_sizes_for_page_size", False):
        return
    if not value_byte_samples:
        return

    factory = config.kv.adapter_factory
    if factory is None or not hasattr(factory, "page_size"):
        # Non-SQLite backends silently ignore the flag — the alternative
        # (raising) would surprise users who switch backends without
        # touching their kv config.
        return

    sorted_sizes = sorted(int(n) for n in value_byte_samples)
    # 95th percentile via the same OFFSET trick used by the local picker
    # (kept as nearest-rank to avoid pulling numpy in for one operation).
    idx = max(0, int(len(sorted_sizes) * 0.95) - 1)
    p95 = sorted_sizes[idx]

    try:
        target = recommend_page_size(p95_value_bytes=p95)
    except ConfigValidationError:
        return

    current = getattr(factory, "page_size", None)
    if current == target:
        return

    try:
        # Protocol typing can't see the dataclass shape of concrete
        # factories; the TypeError below is the runtime guard.
        new_factory = dataclasses.replace(factory, page_size=target)  # type: ignore[type-var]
    except TypeError as exc:
        # Factory is not a dataclass; do not silently fail — the user
        # explicitly asked for engine-percentile picking and we can't
        # honour it.
        raise ConfigValidationError(
            f"profile_value_sizes_for_page_size=True is set but the "
            f"adapter factory {type(factory).__name__!r} cannot be rebuilt "
            "with a different page_size.  Use a SqliteFactory or "
            "SqliteVecFactory, or pass page_size explicitly."
        ) from exc

    config.kv.adapter_factory = new_factory
    log_event(
        "engine_page_size_picked",
        level=logging.DEBUG,
        logger=_logger,
        writer_kind=writer_kind,
        sample_size=len(sorted_sizes),
        p95_value_bytes=p95,
        from_page_size=str(current),
        to_page_size=target,
    )


def collect_value_byte_samples(
    *,
    rows: list[Any],
    value_spec: Any,
) -> list[int]:
    """Encode a list of driver-side rows via ``value_spec`` and return sizes.

    Errors in a single row's encoder are logged and the row is skipped
    — the picker only needs a representative sample, not perfect
    coverage.  An empty result tells the caller to skip the
    substitution and use the default page_size.
    """
    sizes: list[int] = []
    for row in rows:
        try:
            encoded = value_spec.encode(row)
        except Exception as exc:  # pragma: no cover - defensive
            log_event(
                "engine_page_size_sample_encode_failed",
                level=logging.DEBUG,
                logger=_logger,
                error=str(exc),
            )
            continue
        sizes.append(len(encoded))
    return sizes
