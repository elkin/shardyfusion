"""Snapshot routing helpers for sharded manifests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import xxhash

from .manifest import RequiredBuildMeta, RequiredShardMeta
from .serde import make_key_encoder
from .sharding_types import (
    KeyEncoding,
    RoutingValue,
    ShardHashAlgorithm,
    ShardingStrategy,
    validate_routing_values,
)
from .type_defs import KeyInput


class ShardLookup(Protocol):
    """Protocol for shard metadata access — eager (list) or lazy (SQLite)."""

    def get_shard(self, db_id: int) -> RequiredShardMeta:
        """Return metadata for a shard by db_id.

        Must return a synthetic empty-shard entry (``db_url=None``,
        ``row_count=0``) for db_ids that have no data.
        """
        ...


# ---------------------------------------------------------------------------
# Hash algorithms
# ---------------------------------------------------------------------------

_XXH3_SEED = 0
_INT64_SIGNED_MIN = -(1 << 63)
_INT64_SIGNED_MAX = (1 << 63) - 1


def canonical_bytes(key: KeyInput) -> bytes:
    """Convert a key to its canonical byte representation for hashing.

    - ``int`` → 8-byte signed little-endian (range [-2^63, 2^63-1])
    - ``str`` → UTF-8 encoded bytes
    - ``bytes`` / ``bytearray`` → passed through as-is
    """
    if isinstance(key, int):
        if key < _INT64_SIGNED_MIN or key > _INT64_SIGNED_MAX:
            raise ValueError(
                f"Integer key {key} out of range [{_INT64_SIGNED_MIN}, {_INT64_SIGNED_MAX}]"
            )
        return key.to_bytes(8, "little", signed=True)
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, (bytes, bytearray)):
        return bytes(key)
    raise ValueError(f"Unsupported key type for hashing: {type(key)!r}")


def xxh3_digest(key: KeyInput) -> int:
    """Compute the xxh3_64 digest of a key's canonical bytes.

    This is the single authoritative hash step used by both HASH routing
    (``xxh3_db_id``) and the CEL ``shard_hash()`` function.
    """
    return xxhash.xxh3_64_intdigest(canonical_bytes(key), seed=_XXH3_SEED)


def hash_digest(
    key: KeyInput,
    algorithm: ShardHashAlgorithm | str = ShardHashAlgorithm.XXH3_64,
) -> int:
    """Compute a shard-routing digest using the selected hash algorithm."""

    algorithm = ShardHashAlgorithm.from_value(algorithm)
    if algorithm == ShardHashAlgorithm.XXH3_64:
        return xxh3_digest(key)
    raise ValueError(f"Unsupported shard hash algorithm: {algorithm!r}")


def hash_db_id(
    key: KeyInput,
    num_dbs: int,
    algorithm: ShardHashAlgorithm | str = ShardHashAlgorithm.XXH3_64,
) -> int:
    """Route a key to a shard db_id using the selected hash algorithm."""

    return hash_digest(key, algorithm) % num_dbs


def xxh3_db_id(key: KeyInput, num_dbs: int) -> int:
    """Route a key to a shard db_id using xxh3_64.

    ``xxh3_digest(key) % num_dbs``
    """
    return hash_db_id(key, num_dbs, ShardHashAlgorithm.XXH3_64)


# ---------------------------------------------------------------------------
# SnapshotRouter
# ---------------------------------------------------------------------------


class SnapshotRouter:
    """Route point lookups to a shard database id using manifest sharding metadata.

    Operates in two modes:

    * **Eager** (default ``__init__``): preloads all shards into a list.
      Best for small-to-medium shard counts (up to ~10 K).
    * **Lazy** (:meth:`from_build_meta`): delegates shard lookups to a
      :class:`ShardLookup` provider (e.g. backed by SQLite with range
      reads).  Constant memory regardless of shard count.

    Both modes expose :meth:`get_shard` for uniform shard access.
    """

    route_one: Callable[[KeyInput], int]
    encode_lookup_key: Callable[[KeyInput], bytes]

    def __init__(
        self, required_build: RequiredBuildMeta, shards: list[RequiredShardMeta]
    ) -> None:
        self.required_build = required_build
        # Build a complete shard list covering all db_ids in [0, num_dbs).
        # Missing db_ids (empty shards omitted from the manifest) get synthetic
        # metadata-only entries with db_url=None, row_count=0, and db_bytes=0.
        shard_by_id = {s.db_id: s for s in shards}
        self.shards: list[RequiredShardMeta] = [
            shard_by_id.get(
                db_id,
                RequiredShardMeta(db_id=db_id, attempt=0, row_count=0, db_bytes=0),
            )
            for db_id in range(required_build.num_dbs)
        ]
        self._shard_lookup: ShardLookup | None = None
        self._init_routing(required_build)

    @classmethod
    def from_build_meta(
        cls,
        required_build: RequiredBuildMeta,
        *,
        shard_lookup: ShardLookup,
    ) -> SnapshotRouter:
        """Create a router with lazy shard lookup (no preloaded shard list).

        Use this when the manifest has too many shards to load into memory.
        The *shard_lookup* provider supplies :class:`RequiredShardMeta` on
        demand (e.g. via indexed SQLite queries).

        .. note::

           ``router.shards`` is an empty list in lazy mode.  Code that
           iterates over all shards must use the lookup provider directly
           or switch to :meth:`get_shard`.
        """
        router = object.__new__(cls)
        router.required_build = required_build
        router.shards = []
        router._shard_lookup = shard_lookup
        router._init_routing(required_build)
        return router

    def _init_routing(self, required_build: RequiredBuildMeta) -> None:
        """Shared routing setup (strategy, key encoder, CEL state)."""
        self.strategy = required_build.sharding.strategy
        self.num_dbs = required_build.num_dbs
        self.key_encoding = required_build.key_encoding
        self.hash_algorithm = required_build.sharding.hash_algorithm

        routing_values = required_build.sharding.routing_values
        self._routing_values = (
            list(routing_values) if routing_values is not None else None
        )
        if self._routing_values is not None:
            validate_routing_values(self._routing_values)
            self._routing_lookup: dict[RoutingValue, int] | None = {
                value: idx for idx, value in enumerate(self._routing_values)
            }
        else:
            self._routing_lookup = None
        self._cel_compiled: object | None = None
        self._cel_expr = required_build.sharding.cel_expr
        self._cel_columns = required_build.sharding.cel_columns
        self.route_one = self._build_route_one()
        self.encode_lookup_key = self._build_lookup_key_encoder()

    def get_shard(self, db_id: int) -> RequiredShardMeta:
        """Look up shard metadata by *db_id*.

        Works in both eager (list) and lazy (lookup) mode.  Returns a
        synthetic empty-shard entry for db_ids with no data.
        """
        if self._shard_lookup is not None:
            return self._shard_lookup.get_shard(db_id)
        return self.shards[db_id]

    @property
    def is_lazy(self) -> bool:
        """True when this router uses lazy shard lookup (no preloaded list)."""
        return self._shard_lookup is not None

    def route(
        self, key: KeyInput, *, routing_context: dict[str, object] | None = None
    ) -> int:
        """Route a key, using routing_context for CEL multi-column mode if provided."""
        if routing_context is not None:
            return self.route_with_context(routing_context)
        return self.route_one(key)

    def group_keys(
        self,
        keys: list[KeyInput],
        *,
        routing_context: dict[str, object] | None = None,
    ) -> dict[int, list[KeyInput]]:
        """Group keys by routed db id while preserving order within each shard bucket."""

        grouped, missing = self.group_keys_allow_missing(
            keys,
            routing_context=routing_context,
        )
        if missing:
            raise ValueError(
                f"One or more keys could not be routed for this snapshot: {missing!r}"
            )
        return grouped

    def group_keys_allow_missing(
        self,
        keys: list[KeyInput],
        *,
        routing_context: dict[str, object] | None = None,
    ) -> tuple[dict[int, list[KeyInput]], list[KeyInput]]:
        """Group keys by routed db id, returning unroutable categorical keys separately."""

        from .cel import UnknownRoutingTokenError

        grouped: dict[int, list[KeyInput]] = {}
        missing: list[KeyInput] = []
        if routing_context is not None:
            try:
                db_id = self.route_with_context(routing_context)
            except UnknownRoutingTokenError:
                return {}, list(keys)
            grouped[db_id] = list(keys)
        else:
            for key in keys:
                try:
                    db_id = self.route_one(key)
                except UnknownRoutingTokenError:
                    missing.append(key)
                    continue
                grouped.setdefault(db_id, []).append(key)
        return grouped, missing

    def _build_lookup_key_encoder(self) -> Callable[[KeyInput], bytes]:
        if self.key_encoding == KeyEncoding.U64BE:
            encoder = make_key_encoder(KeyEncoding.U64BE)

            def _encode_u64be(key: KeyInput) -> bytes:
                if isinstance(key, bytes):
                    if len(key) != 8:
                        raise ValueError("u64be key bytes must have length 8")
                    return key
                if not isinstance(key, int):
                    raise ValueError("u64be key encoding requires integer lookup keys")
                return encoder(key)

            return _encode_u64be

        if self.key_encoding == KeyEncoding.U32BE:
            encoder = make_key_encoder(KeyEncoding.U32BE)

            def _encode_u32be(key: KeyInput) -> bytes:
                if isinstance(key, bytes):
                    if len(key) != 4:
                        raise ValueError("u32be key bytes must have length 4")
                    return key
                if not isinstance(key, int):
                    raise ValueError("u32be key encoding requires integer lookup keys")
                return encoder(key)

            return _encode_u32be

        if self.key_encoding in (KeyEncoding.UTF8, KeyEncoding.RAW):
            encoder = make_key_encoder(self.key_encoding)
            return encoder  # type: ignore[return-value]

        def _encode_fallback(key: KeyInput) -> bytes:
            if isinstance(key, bytes):
                return key
            if isinstance(key, str):
                return key.encode("utf-8")
            if isinstance(key, int):
                return str(key).encode("utf-8")
            raise ValueError(f"Unsupported key type for lookup: {type(key)!r}")

        return _encode_fallback

    def _build_route_one(self) -> Callable[[KeyInput], int]:
        if self.strategy == ShardingStrategy.HASH:
            num_dbs = self.num_dbs
            algorithm = self.hash_algorithm
            return lambda key: hash_db_id(key, num_dbs, algorithm)

        if self.strategy == ShardingStrategy.CEL:
            from .cel import compile_cel, resolve_cel_routing_key

            assert self._cel_expr is not None and self._cel_columns is not None
            compiled = compile_cel(self._cel_expr, self._cel_columns)
            routing_values = self._routing_values
            routing_lookup = self._routing_lookup

            # Key-only mode: cel_columns has only "key" — auto-wrap key value
            cel_col_keys = set(self._cel_columns.keys())
            if cel_col_keys == {"key"}:
                return lambda key: resolve_cel_routing_key(
                    compiled.evaluate({"key": key}),
                    routing_values=routing_values,
                    lookup=routing_lookup,
                )

            # Multi-column mode: route_one cannot work without routing_context
            def _route_multi_column_error(key: KeyInput) -> int:
                raise ValueError(
                    "Multi-column CEL sharding requires routing_context; "
                    "use route(key, routing_context=...) instead of route_one(key)"
                )

            return _route_multi_column_error

        raise ValueError(f"Unsupported sharding strategy for routing: {self.strategy}")

    def route_with_context(self, routing_context: dict[str, object]) -> int:
        """Route using CEL with explicit routing context.

        In multi-column mode, the CEL expression evaluates over multiple columns
        from the routing context, not just the key.
        """
        from .cel import resolve_cel_routing_key

        if self._cel_compiled is None:
            from .cel import compile_cel

            assert self._cel_expr is not None and self._cel_columns is not None
            self._cel_compiled = compile_cel(self._cel_expr, self._cel_columns)
        return resolve_cel_routing_key(
            self._cel_compiled.evaluate(routing_context),  # type: ignore[union-attr]
            routing_values=self._routing_values,
            lookup=self._routing_lookup,
        )
