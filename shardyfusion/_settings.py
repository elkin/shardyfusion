"""Typed settings for the slatedb.uniffi backend.

The slatedb 0.12 uniffi bindings configure a ``DbBuilder`` via a
:class:`slatedb.uniffi.Settings` object. ``Settings`` itself only
exposes :meth:`Settings.set(dotted_key, value_json)` — every value must
already be a JSON literal, and there is no compile-time validation of
the keys.

:class:`SlateDbSettings` is a thin typed wrapper:

* the small handful of fields shardyfusion actively recommends are
  first-class dataclass attributes,
* an open-ended ``raw_overrides`` mapping is the documented escape
  hatch for everything else (compactor tuning, GC, cache layout, …).

We intentionally do *not* mirror every field of
``slatedb::config::Settings`` here. Doing so would couple shardyfusion
releases to upstream Rust struct churn for no real ergonomic gain;
``raw_overrides`` is type-safe enough for advanced use cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ["SlateDbSettings"]


def _to_json_literal(value: Any) -> str:
    """Encode a Python value as a JSON literal accepted by ``Settings.set``."""
    return json.dumps(value, separators=(",", ":"))


@dataclass(slots=True)
class SlateDbSettings:
    """Typed configuration for the slatedb.uniffi backend.

    Attributes
    ----------
    flush_interval:
        Duration string (e.g. ``"250ms"``, ``"1s"``) controlling how
        often the WAL is flushed to object storage. ``None`` keeps the
        slatedb default.
    l0_sst_size_bytes:
        Target size (bytes) for L0 SSTs before they are sealed.
    default_ttl_ms:
        Default TTL in milliseconds applied to writes that don't carry
        an explicit TTL. ``None`` disables a default TTL.
    raw_overrides:
        Free-form ``{dotted_key: python_value}`` mapping applied on top
        of the typed fields. Values are JSON-encoded before being passed
        to :meth:`slatedb.uniffi.Settings.set`. Use this for advanced
        compactor / cache / GC tuning.

    Examples
    --------
    >>> SlateDbSettings(flush_interval="250ms").to_uniffi_pairs()
    [('flush_interval', '"250ms"')]

    >>> SlateDbSettings(
    ...     raw_overrides={"compactor_options.max_sst_size": 33554432}
    ... ).to_uniffi_pairs()
    [('compactor_options.max_sst_size', '33554432')]
    """

    flush_interval: str | None = None
    l0_sst_size_bytes: int | None = None
    default_ttl_ms: int | None = None
    raw_overrides: dict[str, Any] = field(default_factory=dict)

    def to_uniffi_pairs(self) -> list[tuple[str, str]]:
        """Return ``(dotted_key, json_literal)`` pairs for ``Settings.set``.

        The order is: typed fields first (in declaration order), then
        ``raw_overrides`` in insertion order. Later entries with the
        same dotted key win — i.e. ``raw_overrides`` can override a
        typed field if a caller really wants to.
        """
        pairs: list[tuple[str, str]] = []
        if self.flush_interval is not None:
            pairs.append(("flush_interval", _to_json_literal(self.flush_interval)))
        if self.l0_sst_size_bytes is not None:
            pairs.append(
                ("l0_sst_size_bytes", _to_json_literal(self.l0_sst_size_bytes))
            )
        if self.default_ttl_ms is not None:
            pairs.append(("default_ttl", _to_json_literal(self.default_ttl_ms)))
        for key, value in self.raw_overrides.items():
            pairs.append((key, _to_json_literal(value)))
        return pairs

    def apply(self, settings: Any) -> None:
        """Apply this configuration onto a ``slatedb.uniffi.Settings`` instance."""
        for key, json_literal in self.to_uniffi_pairs():
            settings.set(key, json_literal)
