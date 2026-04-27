"""Unit tests for the adaptive (auto) SQLite reader factories.

Covers:

* :func:`shardyfusion.sqlite_adapter.decide_access_mode` — pure policy logic.
* :class:`AdaptiveSqliteReaderFactory` / :class:`AsyncAdaptiveSqliteReaderFactory`
  — single-slot ``run_id`` cache, threshold dispatch, kwarg propagation.
* :class:`AdaptiveSqliteVecReaderFactory` /
  :class:`AsyncAdaptiveSqliteVecReaderFactory` — same contract over sqlite-vec.

All tests stub :func:`make_sqlite_reader_factory` /
:func:`make_async_sqlite_reader_factory` /
:func:`make_sqlite_vec_reader_factory` /
:func:`make_async_sqlite_vec_reader_factory` so the suite never opens real
SQLite/APSW handles or contacts S3.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shardyfusion.sqlite_adapter import (
    DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES,
    DEFAULT_AUTO_TOTAL_BUDGET_BYTES,
    AdaptiveSqliteReaderFactory,
    AsyncAdaptiveSqliteReaderFactory,
    SqliteAccessPolicy,
    _ThresholdPolicy,
    decide_access_mode,
    make_threshold_policy,
)
from shardyfusion.sqlite_vec_adapter import (
    AdaptiveSqliteVecReaderFactory,
    AsyncAdaptiveSqliteVecReaderFactory,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


@dataclass
class _StubShard:
    db_url: str | None
    db_bytes: int
    row_count: int = 0


@dataclass
class _StubBuildMeta:
    run_id: str


@dataclass
class _StubManifest:
    """Minimal duck-typed ``Manifest`` for adaptive factory tests."""

    run_id: str
    shard_sizes: Sequence[int]
    required_build: _StubBuildMeta = field(init=False)
    shards: list[_StubShard] = field(init=False)

    def __post_init__(self) -> None:
        self.required_build = _StubBuildMeta(run_id=self.run_id)
        self.shards = [
            _StubShard(db_url=f"s3://bucket/db={i:05d}.db", db_bytes=size)
            for i, size in enumerate(self.shard_sizes)
        ]


class _RecordingPolicy:
    """:class:`SqliteAccessPolicy` that records every ``decide`` call."""

    def __init__(self, *, returns: str = "download") -> None:
        self.calls: list[list[int]] = []
        self._returns = returns

    def decide(self, db_bytes_per_shard: Sequence[int]) -> str:  # type: ignore[override]
        self.calls.append(list(db_bytes_per_shard))
        return self._returns  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# decide_access_mode
# ---------------------------------------------------------------------------


class TestDecideAccessMode:
    def test_empty_input_returns_download(self) -> None:
        assert decide_access_mode(db_bytes_per_shard=[]) == "download"

    def test_all_small_shards_return_download(self) -> None:
        # 4 shards of 1 MiB each, defaults are 16 MiB / 2 GiB
        assert decide_access_mode(db_bytes_per_shard=[1_000_000] * 4) == "download"

    def test_per_shard_threshold_triggers_range(self) -> None:
        # one shard at exactly the per-shard threshold
        assert (
            decide_access_mode(
                db_bytes_per_shard=[1_000, DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES],
            )
            == "range"
        )

    def test_per_shard_threshold_minus_one_stays_download(self) -> None:
        assert (
            decide_access_mode(
                db_bytes_per_shard=[DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES - 1],
            )
            == "download"
        )

    def test_total_budget_triggers_range(self) -> None:
        # many small shards summing to >= total budget
        # use per_shard_threshold large so only the sum triggers
        sizes = [1_000_000] * 10  # 10 MB total
        assert (
            decide_access_mode(
                db_bytes_per_shard=sizes,
                per_shard_threshold=100_000_000,
                total_budget=10_000_000,
            )
            == "range"
        )

    def test_zero_byte_shards_are_download(self) -> None:
        assert decide_access_mode(db_bytes_per_shard=[0, 0, 0]) == "download"

    def test_custom_thresholds_respected(self) -> None:
        # lowered per-shard threshold flips a shard that defaults wouldn't
        assert (
            decide_access_mode(
                db_bytes_per_shard=[5_000_000],
                per_shard_threshold=1_000_000,
                total_budget=DEFAULT_AUTO_TOTAL_BUDGET_BYTES,
            )
            == "range"
        )


# ---------------------------------------------------------------------------
# _ThresholdPolicy / make_threshold_policy
# ---------------------------------------------------------------------------


class TestThresholdPolicy:
    def test_default_thresholds(self) -> None:
        policy = _ThresholdPolicy()
        assert policy.per_shard_threshold == DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES
        assert policy.total_budget == DEFAULT_AUTO_TOTAL_BUDGET_BYTES

    def test_decide_delegates_to_helper(self) -> None:
        policy = _ThresholdPolicy(per_shard_threshold=100, total_budget=1_000)
        assert policy.decide([50, 50]) == "download"
        assert policy.decide([200]) == "range"
        assert policy.decide([500, 600]) == "range"  # sum >= 1000

    def test_make_threshold_policy_returns_protocol_compatible_object(self) -> None:
        policy = make_threshold_policy(per_shard_threshold=42, total_budget=84)
        assert isinstance(policy, SqliteAccessPolicy)
        assert isinstance(policy, _ThresholdPolicy)
        assert policy.per_shard_threshold == 42
        assert policy.total_budget == 84


# ---------------------------------------------------------------------------
# Sync KV adaptive factory
# ---------------------------------------------------------------------------


def _make_call_kwargs(manifest: _StubManifest, *, db_id: int = 0) -> dict[str, Any]:
    return {
        "db_url": f"s3://bucket/db={db_id:05d}.db",
        "local_dir": Path(f"/tmp/shard-{db_id}"),
        "checkpoint_id": "ckpt-1",
        "manifest": manifest,
    }


class TestAdaptiveSqliteReaderFactory:
    def test_first_call_resolves_via_policy_and_caches(self) -> None:
        policy = _RecordingPolicy(returns="download")
        adaptive = AdaptiveSqliteReaderFactory(policy=policy)

        manifest = _StubManifest(run_id="run-A", shard_sizes=[100, 200, 300])
        sub_factory = MagicMock(name="DownloadFactory")
        sub_factory.return_value = MagicMock(name="ShardReader")

        with patch(
            "shardyfusion.sqlite_adapter.make_sqlite_reader_factory",
            return_value=sub_factory,
        ) as mk:
            adaptive(**_make_call_kwargs(manifest, db_id=0))
            adaptive(**_make_call_kwargs(manifest, db_id=1))

        # policy called exactly once, with the snapshot's full size sequence
        assert policy.calls == [[100, 200, 300]]
        # sub-factory built exactly once, in download mode
        mk.assert_called_once()
        assert mk.call_args.kwargs["mode"] == "download"
        # but invoked twice, once per shard
        assert sub_factory.call_count == 2

    def test_new_run_id_replaces_cache(self) -> None:
        policy = _RecordingPolicy(returns="download")
        adaptive = AdaptiveSqliteReaderFactory(policy=policy)

        m1 = _StubManifest(run_id="run-A", shard_sizes=[1, 2])
        m2 = _StubManifest(run_id="run-B", shard_sizes=[3, 4, 5])

        download_factory = MagicMock(name="DownloadFactory")
        download_factory.return_value = MagicMock()
        range_factory = MagicMock(name="RangeFactory")
        range_factory.return_value = MagicMock()

        with patch(
            "shardyfusion.sqlite_adapter.make_sqlite_reader_factory",
            side_effect=[download_factory, range_factory],
        ) as mk:
            adaptive(**_make_call_kwargs(m1))
            policy._returns = "range"  # second snapshot picks range
            adaptive(**_make_call_kwargs(m2))
            adaptive(**_make_call_kwargs(m2))  # cache hit on run-B

        assert policy.calls == [[1, 2], [3, 4, 5]]
        assert mk.call_count == 2
        assert mk.call_args_list[0].kwargs["mode"] == "download"
        assert mk.call_args_list[1].kwargs["mode"] == "range"

    def test_threshold_overrides_propagated_to_default_policy(self) -> None:
        adaptive = AdaptiveSqliteReaderFactory(
            per_shard_threshold=10,
            total_budget=1_000,
        )
        policy = adaptive._policy
        assert isinstance(policy, _ThresholdPolicy)
        assert policy.per_shard_threshold == 10
        assert policy.total_budget == 1_000

    def test_explicit_policy_overrides_thresholds(self) -> None:
        custom = _RecordingPolicy()
        adaptive = AdaptiveSqliteReaderFactory(
            policy=custom,
            per_shard_threshold=999,  # ignored when explicit policy supplied
            total_budget=999,
        )
        assert adaptive._policy is custom

    def test_sub_factory_construction_kwargs_propagated(self) -> None:
        creds = MagicMock(name="CredentialProvider")
        s3opts = MagicMock(name="S3ConnectionOptions")
        adaptive = AdaptiveSqliteReaderFactory(
            mmap_size=12345,
            page_cache_pages=77,
            s3_connection_options=s3opts,
            credential_provider=creds,
        )
        manifest = _StubManifest(run_id="r", shard_sizes=[1])
        sub_factory = MagicMock()
        sub_factory.return_value = MagicMock()
        with patch(
            "shardyfusion.sqlite_adapter.make_sqlite_reader_factory",
            return_value=sub_factory,
        ) as mk:
            adaptive(**_make_call_kwargs(manifest))
        kwargs = mk.call_args.kwargs
        assert kwargs["mmap_size"] == 12345
        assert kwargs["page_cache_pages"] == 77
        assert kwargs["s3_connection_options"] is s3opts
        assert kwargs["credential_provider"] is creds

    def test_call_forwards_kwargs_to_sub_factory(self) -> None:
        adaptive = AdaptiveSqliteReaderFactory(policy=_RecordingPolicy())
        manifest = _StubManifest(run_id="r", shard_sizes=[1])
        sub_factory = MagicMock()
        sub_factory.return_value = MagicMock(name="ShardReader")
        with patch(
            "shardyfusion.sqlite_adapter.make_sqlite_reader_factory",
            return_value=sub_factory,
        ):
            kwargs = _make_call_kwargs(manifest, db_id=42)
            result = adaptive(**kwargs)
        sub_factory.assert_called_once_with(**kwargs)
        assert result is sub_factory.return_value


# ---------------------------------------------------------------------------
# Async KV adaptive factory
# ---------------------------------------------------------------------------


class TestAsyncAdaptiveSqliteReaderFactory:
    @pytest.mark.asyncio
    async def test_first_call_resolves_and_caches(self) -> None:
        policy = _RecordingPolicy(returns="range")
        adaptive = AsyncAdaptiveSqliteReaderFactory(policy=policy)
        manifest = _StubManifest(run_id="run-X", shard_sizes=[10, 20])

        async def _sub_factory(**_kwargs: Any) -> MagicMock:
            return MagicMock(name="AsyncShardReader")

        sub_factory_mock = MagicMock(side_effect=_sub_factory)
        with patch(
            "shardyfusion.sqlite_adapter.make_async_sqlite_reader_factory",
            return_value=sub_factory_mock,
        ) as mk:
            await adaptive(**_make_call_kwargs(manifest, db_id=0))
            await adaptive(**_make_call_kwargs(manifest, db_id=1))

        assert policy.calls == [[10, 20]]
        mk.assert_called_once()
        assert mk.call_args.kwargs["mode"] == "range"
        assert sub_factory_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_new_run_id_replaces_cache(self) -> None:
        policy = _RecordingPolicy(returns="download")
        adaptive = AsyncAdaptiveSqliteReaderFactory(policy=policy)
        m1 = _StubManifest(run_id="r1", shard_sizes=[1])
        m2 = _StubManifest(run_id="r2", shard_sizes=[1])

        async def _f(**_kwargs: Any) -> MagicMock:
            return MagicMock()

        sub1 = MagicMock(side_effect=_f)
        sub2 = MagicMock(side_effect=_f)
        with patch(
            "shardyfusion.sqlite_adapter.make_async_sqlite_reader_factory",
            side_effect=[sub1, sub2],
        ) as mk:
            await adaptive(**_make_call_kwargs(m1))
            await adaptive(**_make_call_kwargs(m2))

        assert mk.call_count == 2


# ---------------------------------------------------------------------------
# sqlite-vec adaptive factories
# ---------------------------------------------------------------------------


class TestAdaptiveSqliteVecReaderFactory:
    def test_first_call_resolves_and_caches(self) -> None:
        policy = _RecordingPolicy(returns="download")
        adaptive = AdaptiveSqliteVecReaderFactory(policy=policy)
        manifest = _StubManifest(run_id="vec-A", shard_sizes=[7, 8, 9])
        sub_factory = MagicMock()
        sub_factory.return_value = MagicMock()
        with patch(
            "shardyfusion.sqlite_vec_adapter.make_sqlite_vec_reader_factory",
            return_value=sub_factory,
        ) as mk:
            adaptive(**_make_call_kwargs(manifest, db_id=0))
            adaptive(**_make_call_kwargs(manifest, db_id=1))
            adaptive(**_make_call_kwargs(manifest, db_id=2))

        assert policy.calls == [[7, 8, 9]]
        mk.assert_called_once()
        assert mk.call_args.kwargs["mode"] == "download"
        assert sub_factory.call_count == 3

    def test_new_run_id_replaces_cache(self) -> None:
        policy = _RecordingPolicy(returns="download")
        adaptive = AdaptiveSqliteVecReaderFactory(policy=policy)
        m1 = _StubManifest(run_id="vec-A", shard_sizes=[1])
        m2 = _StubManifest(run_id="vec-B", shard_sizes=[2])

        sub1 = MagicMock()
        sub1.return_value = MagicMock()
        sub2 = MagicMock()
        sub2.return_value = MagicMock()
        with patch(
            "shardyfusion.sqlite_vec_adapter.make_sqlite_vec_reader_factory",
            side_effect=[sub1, sub2],
        ) as mk:
            adaptive(**_make_call_kwargs(m1))
            policy._returns = "range"
            adaptive(**_make_call_kwargs(m2))

        assert mk.call_count == 2
        assert mk.call_args_list[0].kwargs["mode"] == "download"
        assert mk.call_args_list[1].kwargs["mode"] == "range"

    def test_default_policy_uses_default_thresholds(self) -> None:
        adaptive = AdaptiveSqliteVecReaderFactory()
        policy = adaptive._policy
        assert isinstance(policy, _ThresholdPolicy)
        assert policy.per_shard_threshold == DEFAULT_AUTO_PER_SHARD_THRESHOLD_BYTES
        assert policy.total_budget == DEFAULT_AUTO_TOTAL_BUDGET_BYTES


class TestAsyncAdaptiveSqliteVecReaderFactory:
    @pytest.mark.asyncio
    async def test_first_call_resolves_and_caches(self) -> None:
        policy = _RecordingPolicy(returns="range")
        adaptive = AsyncAdaptiveSqliteVecReaderFactory(policy=policy)
        manifest = _StubManifest(run_id="async-vec-A", shard_sizes=[100, 200])

        async def _f(**_kwargs: Any) -> MagicMock:
            return MagicMock()

        sub_factory = MagicMock(side_effect=_f)
        with patch(
            "shardyfusion.sqlite_vec_adapter.make_async_sqlite_vec_reader_factory",
            return_value=sub_factory,
        ) as mk:
            await adaptive(**_make_call_kwargs(manifest, db_id=0))
            await adaptive(**_make_call_kwargs(manifest, db_id=1))

        assert policy.calls == [[100, 200]]
        mk.assert_called_once()
        assert mk.call_args.kwargs["mode"] == "range"
        assert sub_factory.call_count == 2

    @pytest.mark.asyncio
    async def test_threshold_overrides_propagated(self) -> None:
        adaptive = AsyncAdaptiveSqliteVecReaderFactory(
            per_shard_threshold=4096,
            total_budget=65536,
        )
        policy = adaptive._policy
        assert isinstance(policy, _ThresholdPolicy)
        assert policy.per_shard_threshold == 4096
        assert policy.total_budget == 65536
