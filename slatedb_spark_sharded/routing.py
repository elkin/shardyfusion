"""Snapshot routing helpers for sharded SlateDB manifests."""

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable

import xxhash

from .manifest import RequiredBuildMeta, RequiredShardMeta
from .ordering import compare_ordered
from .serde import encode_key
from .sharding_types import KeyEncoding, ShardingStrategy
from .type_defs import KeyInput

RangeValue = int | float | str
_RANGE_INTERVAL_MISMATCH = (
    "Range shard intervals use mixed bound types and are not routable"
)
_RANGE_KEY_MISMATCH = "Range key type does not match shard bound type in manifest"


@dataclass(slots=True)
class _RangeInterval:
    db_id: int
    lower: RangeValue | None
    upper: RangeValue | None


class SnapshotRouter:
    """Route point lookups to a shard database id using manifest sharding metadata."""

    def __init__(
        self, required_build: RequiredBuildMeta, shards: list[RequiredShardMeta]
    ) -> None:
        self.required_build = required_build
        self.shards = sorted(shards, key=lambda shard: shard.db_id)
        self.strategy = required_build.sharding.strategy
        self.num_dbs = required_build.num_dbs
        self.key_encoding = required_build.key_encoding

        self._boundaries = list(required_build.sharding.boundaries or [])
        self._range_intervals = self._build_range_intervals(self.shards)
        self._route_one_impl = self._build_route_one_impl()

    def route_one(self, key: KeyInput) -> int:
        """Route one key to db_id."""

        return self._route_one_impl(key)

    def group_keys(self, keys: list[KeyInput]) -> dict[int, list[KeyInput]]:
        """Group keys by routed db id while preserving order within each shard bucket."""

        grouped: dict[int, list[KeyInput]] = {}
        for key in keys:
            db_id = self.route_one(key)
            grouped.setdefault(db_id, []).append(key)
        return grouped

    def encode_lookup_key(self, key: KeyInput) -> bytes:
        """Encode lookup key for SlateDB read calls."""

        if self.key_encoding == KeyEncoding.U64BE:
            if isinstance(key, bytes):
                if len(key) != 8:
                    raise ValueError("u64be key bytes must have length 8")
                return key
            if not isinstance(key, int):
                raise ValueError("u64be key encoding requires integer lookup keys")
            return encode_key(key, encoding=KeyEncoding.U64BE)

        if self.key_encoding == KeyEncoding.U32BE:
            if isinstance(key, bytes):
                if len(key) != 4:
                    raise ValueError("u32be key bytes must have length 4")
                return key
            if not isinstance(key, int):
                raise ValueError("u32be key encoding requires integer lookup keys")
            return encode_key(key, encoding=KeyEncoding.U32BE)

        if isinstance(key, bytes):
            return key
        if isinstance(key, str):
            return key.encode("utf-8")
        if isinstance(key, int):
            return str(key).encode("utf-8")

        raise ValueError(f"Unsupported key type for lookup: {type(key)!r}")

    def _route_range(self, key: KeyInput) -> int:
        key_value = self._normalize_range_key(key)

        if self._range_intervals:
            db_id = _search_intervals(self._range_intervals, key_value)
            if db_id is not None:
                return db_id

        if self._boundaries:
            return bisect_right(self._boundaries, key_value)

        raise ValueError(
            "Range routing requires shard min/max ranges or sharding boundaries in manifest."
        )

    def _build_route_one_impl(self) -> Callable[[KeyInput], int]:
        if self.strategy == ShardingStrategy.HASH:
            return lambda key: _xxhash64_db_id(
                key,
                self.num_dbs,
                self.key_encoding,
            )

        if self.strategy == ShardingStrategy.RANGE:
            return self._route_range

        if self.strategy == ShardingStrategy.CUSTOM_EXPR:
            if self._range_intervals or self._boundaries:
                return self._route_range
            raise ValueError(
                "Sharding strategy custom_expr is not directly routable at read time; "
                "manifest must include explicit boundaries or shard ranges."
            )

        raise ValueError(f"Unsupported sharding strategy for routing: {self.strategy}")

    def _normalize_range_key(self, key: KeyInput) -> RangeValue:
        if isinstance(key, (int, float, str)):
            return key

        if isinstance(key, bytes):
            if self.key_encoding == KeyEncoding.U64BE:
                if len(key) != 8:
                    raise ValueError("u64be key bytes must have length 8")
                return int.from_bytes(key, byteorder="big", signed=False)
            if self.key_encoding == KeyEncoding.U32BE:
                if len(key) != 4:
                    raise ValueError("u32be key bytes must have length 4")
                return int.from_bytes(key, byteorder="big", signed=False)
            try:
                return key.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    "Range routing cannot decode bytes key as UTF-8"
                ) from exc

        raise ValueError(f"Unsupported range key type: {type(key)!r}")

    @staticmethod
    def _build_range_intervals(shards: list[RequiredShardMeta]) -> list[_RangeInterval]:
        intervals = [
            _RangeInterval(db_id=shard.db_id, lower=shard.min_key, upper=shard.max_key)
            for shard in shards
        ]

        if not intervals:
            return []

        has_any_bound = any(
            item.lower is not None or item.upper is not None for item in intervals
        )
        if not has_any_bound:
            return []

        _validate_interval_bound_types(intervals)

        sorted_intervals = sorted(intervals, key=_interval_sort_key)

        prev_upper: int | float | str | None = None
        for current in sorted_intervals:
            if prev_upper is not None and current.lower is not None:
                if (
                    compare_ordered(
                        current.lower,
                        prev_upper,
                        mismatch_message=_RANGE_INTERVAL_MISMATCH,
                    )
                    <= 0
                ):
                    raise ValueError(
                        "Range shard intervals overlap and are not routable"
                    )
            if current.upper is not None:
                prev_upper = current.upper

        return sorted_intervals


# SHARDING INVARIANT: This seed MUST match Spark's xxhash64 default seed (42).
# See: org.apache.spark.sql.catalyst.expressions.XxHash64, default seed = 42.
# Cross-checked by: tests/unit/writer/test_routing_contract.py
_XXHASH64_SEED = 42
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1
_INT64_MAX = (1 << 63) - 1
_INT64_MOD = 1 << 64


def _xxhash64_db_id(key: KeyInput, num_dbs: int, key_encoding: KeyEncoding) -> int:
    """Route key with ``pmod(xxhash64(...), num_dbs)`` semantics.

    SHARDING INVARIANT: This function replicates Spark's
    ``pmod(xxhash64(cast(key as long)), num_dbs)``.  The payload is
    8-byte little-endian (matching JVM Long.reverseBytes), the digest
    is converted to signed int64 (matching JVM's signed long), and
    Python ``%`` with positive ``num_dbs`` equals Spark ``pmod``.
    Verified at runtime by ``writer.spark.writer._verify_routing_agreement``
    and cross-checked by ``tests/unit/writer/test_routing_contract.py``.
    """

    digest = _xxhash64_signed(_xxhash64_payload(key, key_encoding))
    return digest % num_dbs


def _xxhash64_payload(key: KeyInput, key_encoding: KeyEncoding) -> bytes:
    if key_encoding == KeyEncoding.U64BE:
        if isinstance(key, bytes):
            if len(key) != 8:
                raise ValueError("u64be key bytes must have length 8")
            numeric_key = int.from_bytes(key, byteorder="big", signed=False)
            return numeric_key.to_bytes(8, byteorder="little", signed=False)

        if not isinstance(key, int):
            raise ValueError("u64be hash routing requires integer lookup keys")
        if key < 0 or key > _UINT64_MAX:
            raise ValueError("u64be hash routing requires key in [0, 2^64-1]")
        return key.to_bytes(8, byteorder="little", signed=False)

    if key_encoding == KeyEncoding.U32BE:
        # Zero-extend to 8-byte little-endian to match Spark's xxhash64(cast(key as long))
        if isinstance(key, bytes):
            if len(key) != 4:
                raise ValueError("u32be key bytes must have length 4")
            numeric_key = int.from_bytes(key, byteorder="big", signed=False)
            return numeric_key.to_bytes(8, byteorder="little", signed=False)

        if not isinstance(key, int):
            raise ValueError("u32be hash routing requires integer lookup keys")
        if key < 0 or key > _UINT32_MAX:
            raise ValueError("u32be hash routing requires key in [0, 2^32-1]")
        return key.to_bytes(8, byteorder="little", signed=False)

    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8")
    if isinstance(key, int):
        return str(key).encode("utf-8")

    raise ValueError(f"Unsupported key type for hash routing: {type(key)!r}")


def _xxhash64_signed(payload: bytes) -> int:
    digest = xxhash.xxh64_intdigest(payload, seed=_XXHASH64_SEED)
    return digest if digest <= _INT64_MAX else digest - _INT64_MOD


def _interval_sort_key(interval: _RangeInterval) -> tuple[int, RangeValue]:
    if interval.lower is None:
        return (0, 0)
    return (1, interval.lower)


def _search_intervals(intervals: list[_RangeInterval], key: RangeValue) -> int | None:
    lo = 0
    hi = len(intervals) - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        interval = intervals[mid]

        if (
            interval.upper is not None
            and compare_ordered(
                key,
                interval.upper,
                mismatch_message=_RANGE_KEY_MISMATCH,
            )
            > 0
        ):
            lo = mid + 1
            continue

        if (
            interval.lower is not None
            and compare_ordered(
                key,
                interval.lower,
                mismatch_message=_RANGE_KEY_MISMATCH,
            )
            < 0
        ):
            hi = mid - 1
            continue

        return interval.db_id

    return None


def _validate_interval_bound_types(intervals: list[_RangeInterval]) -> None:
    expected_kind: bool | None = None
    for interval in intervals:
        for bound in (interval.lower, interval.upper):
            if bound is None:
                continue
            is_str = isinstance(bound, str)
            if expected_kind is None:
                expected_kind = is_str
            elif expected_kind != is_str:
                raise ValueError(_RANGE_INTERVAL_MISMATCH)
