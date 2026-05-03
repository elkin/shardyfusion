"""Tests for publish_to_store() error paths in _writer_core.py.

Covers the PublishCurrentError retry loop, retry exhaustion,
re-raise when manifest_ref is None, and generic publish failures.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from shardyfusion._writer_core import publish_to_store
from shardyfusion.config import (
    HashShardedWriteConfig,
    WriterManifestConfig,
    WriterOutputConfig,
)
from shardyfusion.errors import PublishCurrentError, PublishManifestError
from shardyfusion.manifest import (
    ManifestRef,
    ManifestShardingSpec,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from shardyfusion.metrics import MetricEvent
from shardyfusion.sharding_types import HashShardingSpec, KeyEncoding, ShardingStrategy


def _build(run_id: str = "r1") -> RequiredBuildMeta:
    return RequiredBuildMeta(
        run_id=run_id,
        created_at="2026-01-01T00:00:00+00:00",
        num_dbs=2,
        s3_prefix="s3://bucket/prefix",
        key_col="id",
        key_encoding=KeyEncoding.U64BE,
        sharding=ManifestShardingSpec(
            strategy=ShardingStrategy.HASH,
            hash_algorithm="xxh3_64",
        ),
        db_path_template="db={db_id:05d}",
        shard_prefix="shards",
    )


def _shard(db_id: int = 0) -> RequiredShardMeta:
    return RequiredShardMeta(
        db_id=db_id,
        db_url=f"s3://bucket/prefix/db={db_id:05d}",
        attempt=0,
        row_count=10,
        checkpoint_id=None,
        writer_info={},
        db_bytes=0,
    )


class _FakeManifestStore:
    """Configurable fake that can inject failures at publish or set_current."""

    def __init__(
        self,
        *,
        publish_raises: Exception | None = None,
        set_current_raises: list[Exception | None] | None = None,
    ) -> None:
        self._publish_raises = publish_raises
        self._set_current_raises = list(set_current_raises or [])
        self._set_current_call = 0
        self.published_ref: str | None = None

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        if self._publish_raises is not None:
            raise self._publish_raises
        self.published_ref = f"mem://manifests/run_id={run_id}/manifest"
        return self.published_ref

    def set_current(self, ref: str) -> None:
        if self._set_current_call < len(self._set_current_raises):
            exc = self._set_current_raises[self._set_current_call]
            self._set_current_call += 1
            if exc is not None:
                raise exc
        self._set_current_call += 1

    def load_current(self) -> ManifestRef | None:
        return None

    def load_manifest(self, ref: str) -> ParsedManifest:
        raise NotImplementedError

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        return []


def _config(store: _FakeManifestStore) -> HashShardedWriteConfig:
    return HashShardedWriteConfig(
        num_dbs=2,
        s3_prefix="s3://bucket/prefix",
        manifest=WriterManifestConfig(store=store),
        output=WriterOutputConfig(run_id="r1"),
    )


class TestPublishToStoreHappyPath:
    def test_returns_manifest_ref(self) -> None:
        store = _FakeManifestStore()
        cfg = _config(store)
        ref = publish_to_store(
            config=cfg,
            run_id="r1",
            resolved_sharding=HashShardingSpec(),
            winners=[_shard(0), _shard(1)],
            num_dbs=2,
        )
        assert "run_id=r1" in ref

    def test_metrics_emitted_on_success(self) -> None:
        mc = MagicMock()
        store = _FakeManifestStore()
        cfg = HashShardedWriteConfig(
            num_dbs=2,
            s3_prefix="s3://bucket/prefix",
            manifest=WriterManifestConfig(store=store),
            output=WriterOutputConfig(run_id="r1"),
            metrics_collector=mc,
        )
        publish_to_store(
            config=cfg,
            run_id="r1",
            resolved_sharding=HashShardingSpec(),
            winners=[_shard(0)],
            num_dbs=2,
        )
        events = [c.args[0] for c in mc.emit.call_args_list]
        assert MetricEvent.MANIFEST_PUBLISHED in events


class TestPublishCurrentErrorRetry:
    @patch("shardyfusion._writer_core.time.sleep")
    def test_retry_succeeds_on_set_current(self, mock_sleep: MagicMock) -> None:
        """PublishCurrentError with manifest_ref triggers set_current retry."""
        store = _FakeManifestStore(
            publish_raises=PublishCurrentError(
                "pointer failed", manifest_ref="s3://bucket/prefix/manifest"
            ),
            set_current_raises=[None],  # first set_current succeeds
        )
        cfg = _config(store)
        ref = publish_to_store(
            config=cfg,
            run_id="r1",
            resolved_sharding=HashShardingSpec(),
            winners=[_shard(0)],
            num_dbs=2,
        )
        assert ref == "s3://bucket/prefix/manifest"

    @patch("shardyfusion._writer_core.time.sleep")
    def test_retry_succeeds_on_second_attempt(self, mock_sleep: MagicMock) -> None:
        """set_current fails once, then succeeds on retry."""
        store = _FakeManifestStore(
            publish_raises=PublishCurrentError(
                "pointer failed", manifest_ref="s3://bucket/prefix/manifest"
            ),
            set_current_raises=[RuntimeError("transient"), None],
        )
        cfg = _config(store)
        ref = publish_to_store(
            config=cfg,
            run_id="r1",
            resolved_sharding=HashShardingSpec(),
            winners=[_shard(0)],
            num_dbs=2,
        )
        assert ref == "s3://bucket/prefix/manifest"
        assert mock_sleep.call_count == 1  # slept once between retries

    @patch("shardyfusion._writer_core.time.sleep")
    def test_all_retries_exhausted_raises_publish_manifest_error(
        self, mock_sleep: MagicMock
    ) -> None:
        """All set_current retries fail → PublishManifestError."""
        store = _FakeManifestStore(
            publish_raises=PublishCurrentError(
                "pointer failed", manifest_ref="s3://bucket/prefix/manifest"
            ),
            set_current_raises=[
                RuntimeError("fail1"),
                RuntimeError("fail2"),
                RuntimeError("fail3"),
            ],
        )
        cfg = _config(store)
        with pytest.raises(PublishManifestError, match="CURRENT pointer after retries"):
            publish_to_store(
                config=cfg,
                run_id="r1",
                resolved_sharding=HashShardingSpec(),
                winners=[_shard(0)],
                num_dbs=2,
            )

    def test_publish_current_error_no_manifest_ref_reraises(self) -> None:
        """PublishCurrentError with manifest_ref=None is re-raised immediately."""
        store = _FakeManifestStore(
            publish_raises=PublishCurrentError("pointer failed", manifest_ref=None),
        )
        cfg = _config(store)
        with pytest.raises(PublishCurrentError, match="pointer failed"):
            publish_to_store(
                config=cfg,
                run_id="r1",
                resolved_sharding=HashShardingSpec(),
                winners=[_shard(0)],
                num_dbs=2,
            )


class TestPublishGenericFailure:
    def test_generic_exception_wraps_in_publish_manifest_error(self) -> None:
        """Non-PublishCurrentError exceptions → PublishManifestError."""
        store = _FakeManifestStore(
            publish_raises=RuntimeError("S3 connection timeout"),
        )
        cfg = _config(store)
        with pytest.raises(PublishManifestError, match="Failed to publish manifest"):
            publish_to_store(
                config=cfg,
                run_id="r1",
                resolved_sharding=HashShardingSpec(),
                winners=[_shard(0)],
                num_dbs=2,
            )
