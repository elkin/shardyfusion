"""Snapshot routing helpers for sharded SlateDB manifests."""

from collections.abc import Callable

import xxhash

from .manifest import RequiredBuildMeta, RequiredShardMeta
from .serde import make_key_encoder
from .sharding_types import KeyEncoding, ShardingStrategy
from .type_defs import KeyInput

# ---------------------------------------------------------------------------
# Universal hash: xxh3_64 with seed=0
# ---------------------------------------------------------------------------

_XXH3_SEED = 0
_INT64_SIGNED_MIN = -(1 << 63)
_INT64_SIGNED_MAX = (1 << 63) - 1


def canonical_bytes(key: KeyInput) -> bytes:
    """Convert a key to its canonical byte representation for hashing.

    - ``int`` → 8-byte signed little-endian (range [-2^63, 2^63-1])
    - ``str`` → UTF-8 encoded bytes
    - ``bytes`` → passed through as-is
    """
    if isinstance(key, int):
        if key < _INT64_SIGNED_MIN or key > _INT64_SIGNED_MAX:
            raise ValueError(
                f"Integer key {key} out of range [{_INT64_SIGNED_MIN}, {_INT64_SIGNED_MAX}]"
            )
        return key.to_bytes(8, "little", signed=True)
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, bytes):
        return key
    raise ValueError(f"Unsupported key type for hashing: {type(key)!r}")


def xxh3_digest(key: KeyInput) -> int:
    """Compute the xxh3_64 digest of a key's canonical bytes.

    This is the single authoritative hash step used by both HASH routing
    (``xxh3_db_id``) and the CEL ``shard_hash()`` function.
    """
    return xxhash.xxh3_64_intdigest(canonical_bytes(key), seed=_XXH3_SEED)


def xxh3_db_id(key: KeyInput, num_dbs: int) -> int:
    """Route a key to a shard db_id using xxh3_64.

    ``xxh3_digest(key) % num_dbs``
    """
    return xxh3_digest(key) % num_dbs


# ---------------------------------------------------------------------------
# SnapshotRouter
# ---------------------------------------------------------------------------


class SnapshotRouter:
    """Route point lookups to a shard database id using manifest sharding metadata."""

    route_one: Callable[[KeyInput], int]
    encode_lookup_key: Callable[[KeyInput], bytes]

    def __init__(
        self, required_build: RequiredBuildMeta, shards: list[RequiredShardMeta]
    ) -> None:
        self.required_build = required_build
        # Build a complete shard list covering all db_ids in [0, num_dbs).
        # Missing db_ids (empty shards omitted from the manifest) get synthetic
        # metadata-only entries with db_url=None and row_count=0.
        shard_by_id = {s.db_id: s for s in shards}
        self.shards = [
            shard_by_id.get(
                db_id, RequiredShardMeta(db_id=db_id, attempt=0, row_count=0)
            )
            for db_id in range(required_build.num_dbs)
        ]
        self.strategy = required_build.sharding.strategy
        self.num_dbs = required_build.num_dbs
        self.key_encoding = required_build.key_encoding

        self._boundaries = list(required_build.sharding.boundaries or [])
        self._cel_compiled: object | None = None
        self._cel_expr = required_build.sharding.cel_expr
        self._cel_columns = required_build.sharding.cel_columns
        self.route_one = self._build_route_one()
        self.encode_lookup_key = self._build_lookup_key_encoder()

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

        grouped: dict[int, list[KeyInput]] = {}
        if routing_context is not None:
            db_id = self.route_with_context(routing_context)
            grouped[db_id] = list(keys)
        else:
            for key in keys:
                db_id = self.route_one(key)
                grouped.setdefault(db_id, []).append(key)
        return grouped

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
            return lambda key: xxh3_db_id(key, num_dbs)

        if self.strategy == ShardingStrategy.CEL:
            from .cel import compile_cel, route_cel

            assert self._cel_expr is not None and self._cel_columns is not None
            compiled = compile_cel(self._cel_expr, self._cel_columns)
            boundaries = self._boundaries

            # Key-only mode: cel_columns has only "key" — auto-wrap key value
            cel_col_keys = set(self._cel_columns.keys())
            if cel_col_keys == {"key"}:
                if boundaries:
                    return lambda key: route_cel(compiled, {"key": key}, boundaries)
                return lambda key: route_cel(compiled, {"key": key}, None)

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
        from .cel import route_cel

        if self._cel_compiled is None:
            from .cel import compile_cel

            assert self._cel_expr is not None and self._cel_columns is not None
            self._cel_compiled = compile_cel(self._cel_expr, self._cel_columns)
        return route_cel(self._cel_compiled, routing_context, self._boundaries or None)  # type: ignore[arg-type]
