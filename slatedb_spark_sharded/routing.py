"""Snapshot routing helpers for sharded SlateDB manifests."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Callable, cast

import xxhash

from .manifest import RequiredBuildMeta, RequiredShardMeta
from .serde import encode_key
from .sharding import ShardingStrategy


@dataclass(slots=True)
class _RangeInterval:
    db_id: int
    lower: int | float | str | None
    upper: int | float | str | None


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

    def route_one(self, key: int | str | bytes) -> int:
        """Route one key to db_id."""

        return self._route_one_impl(key)

    def group_keys(
        self, keys: list[int | str | bytes]
    ) -> dict[int, list[int | str | bytes]]:
        """Group keys by routed db id while preserving order within each shard bucket."""

        grouped: dict[int, list[int | str | bytes]] = {}
        for key in keys:
            db_id = self.route_one(key)
            grouped.setdefault(db_id, []).append(key)
        return grouped

    def encode_lookup_key(self, key: int | str | bytes) -> bytes:
        """Encode lookup key for SlateDB read calls."""

        if self.key_encoding == "u64be":
            if isinstance(key, bytes):
                if len(key) != 8:
                    raise ValueError("u64be key bytes must have length 8")
                return key
            if not isinstance(key, int):
                raise ValueError("u64be key encoding requires integer lookup keys")
            return encode_key(key, encoding="u64be")

        if isinstance(key, bytes):
            return key
        if isinstance(key, str):
            return key.encode("utf-8")
        if isinstance(key, int):
            return str(key).encode("utf-8")

        raise ValueError(f"Unsupported key type for lookup: {type(key)!r}")

    def _route_range(self, key: int | str | bytes) -> int:
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

    def _build_route_one_impl(self) -> Callable[[int | str | bytes], int]:
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

    def _normalize_range_key(self, key: int | str | bytes) -> int | float | str:
        if isinstance(key, (int, float, str)):
            return key

        if isinstance(key, bytes):
            if self.key_encoding == "u64be":
                if len(key) != 8:
                    raise ValueError("u64be key bytes must have length 8")
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

        sorted_intervals = sorted(intervals, key=_interval_sort_key)

        prev_upper: int | float | str | None = None
        for current in sorted_intervals:
            if (
                prev_upper is not None
                and current.lower is not None
                and cast(Any, current.lower) <= cast(Any, prev_upper)
            ):
                raise ValueError("Range shard intervals overlap and are not routable")
            if current.upper is not None:
                prev_upper = current.upper

        return sorted_intervals


_XXHASH64_SEED = 42
_UINT64_MAX = (1 << 64) - 1
_INT64_MAX = (1 << 63) - 1
_INT64_MOD = 1 << 64


def _xxhash64_db_id(key: int | str | bytes, num_dbs: int, key_encoding: str) -> int:
    """Route key with `pmod(xxhash64(...), num_dbs)` semantics."""

    digest = _xxhash64_signed(_xxhash64_payload(key, key_encoding))
    return digest % num_dbs


def _xxhash64_payload(key: int | str | bytes, key_encoding: str) -> bytes:
    if key_encoding == "u64be":
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


def _interval_sort_key(interval: _RangeInterval) -> tuple[int, Any]:
    if interval.lower is None:
        return (0, 0)
    return (1, interval.lower)


def _search_intervals(
    intervals: list[_RangeInterval], key: int | float | str
) -> int | None:
    lo = 0
    hi = len(intervals) - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        interval = intervals[mid]

        if interval.upper is not None and cast(Any, key) > cast(Any, interval.upper):
            lo = mid + 1
            continue

        if interval.lower is not None and cast(Any, key) < cast(Any, interval.lower):
            hi = mid - 1
            continue

        return interval.db_id

    return None
