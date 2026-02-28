"""Tests for the Dask single-database writer."""

from __future__ import annotations

import json
import pathlib
from collections.abc import Iterable
from typing import Self

import pandas as pd
import pytest

dd = pytest.importorskip("dask.dataframe")
import dask  # noqa: E402

from slatedb_spark_sharded.config import (  # noqa: E402
    ManifestOptions,
    OutputOptions,
    WriteConfig,
)
from slatedb_spark_sharded.errors import ConfigValidationError  # noqa: E402
from slatedb_spark_sharded.manifest import BuildResult, ManifestArtifact  # noqa: E402
from slatedb_spark_sharded.publish import ManifestPublisher  # noqa: E402
from slatedb_spark_sharded.serde import ValueSpec, make_key_encoder  # noqa: E402
from slatedb_spark_sharded.sharding_types import KeyEncoding  # noqa: E402
from slatedb_spark_sharded.testing import (  # noqa: E402
    file_backed_adapter_factory,
    file_backed_load_db,
)
from slatedb_spark_sharded.writer.dask.single_db_writer import (  # noqa: E402
    DaskCacheContext,
    write_single_db_dask,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _synchronous_scheduler():
    """Force synchronous Dask scheduler for deterministic test behavior."""
    with dask.config.set(scheduler="synchronous"):
        yield


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class _TrackingAdapter:
    """In-memory adapter that records all write_batch calls."""

    def __init__(self) -> None:
        self.write_calls: list[list[tuple[bytes, bytes]]] = []
        self.flushed = False
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def write_batch(self, pairs: Iterable[tuple[bytes, bytes]]) -> None:
        self.write_calls.append(list(pairs))

    def flush(self) -> None:
        self.flushed = True

    def checkpoint(self) -> str | None:
        return "fake-checkpoint"

    def close(self) -> None:
        self.closed = True


class _TrackingFactory:
    """Factory that creates tracking adapters and keeps references."""

    def __init__(self) -> None:
        self.adapters: list[_TrackingAdapter] = []

    def __call__(self, *, db_url: str, local_dir: str) -> _TrackingAdapter:
        adapter = _TrackingAdapter()
        self.adapters.append(adapter)
        return adapter


class _InMemoryPublisher(ManifestPublisher):
    def __init__(self) -> None:
        self.objects: dict[str, ManifestArtifact] = {}

    def publish_manifest(
        self, *, name: str, artifact: ManifestArtifact, run_id: str
    ) -> str:
        ref = f"mem://manifests/run_id={run_id}/{name}"
        self.objects[ref] = artifact
        return ref

    def publish_current(self, *, name: str, artifact: ManifestArtifact) -> str | None:
        ref = f"mem://{name}"
        self.objects[ref] = artifact
        return ref


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    factory: _TrackingFactory | None = None,
    batch_size: int = 50_000,
    key_encoding: KeyEncoding = KeyEncoding.U64BE,
    num_dbs: int = 1,
) -> WriteConfig:
    return WriteConfig(
        num_dbs=num_dbs,
        s3_prefix="s3://bucket/prefix",
        key_encoding=key_encoding,
        batch_size=batch_size,
        adapter_factory=factory or _TrackingFactory(),
        output=OutputOptions(run_id="test-run"),
        manifest=ManifestOptions(publisher=_InMemoryPublisher()),
    )


def _make_dask_df(
    records: list[dict[str, object]], npartitions: int = 2
) -> dd.DataFrame:
    pdf = pd.DataFrame(records)
    return dd.from_pandas(pdf, npartitions=npartitions)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_sorted_write() -> None:
    factory = _TrackingFactory()
    config = _make_config(factory=factory, batch_size=100)

    # Unsorted keys
    records = [{"id": k} for k in [5, 3, 1, 4, 2]]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert isinstance(result, BuildResult)
    assert len(result.winners) == 1
    assert result.winners[0].db_id == 0
    assert result.winners[0].row_count == 5
    assert result.stats.rows_written == 5

    # Verify keys are written in sorted order
    adapter = factory.adapters[0]
    all_keys = [pair[0] for call in adapter.write_calls for pair in call]
    assert all_keys == sorted(all_keys)


def test_sort_keys_false() -> None:
    factory = _TrackingFactory()
    config = _make_config(factory=factory, batch_size=100)

    records = [{"id": k} for k in [5, 3, 1, 4, 2]]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        sort_keys=False,
    )

    assert result.winners[0].row_count == 5


def test_validates_num_dbs_1() -> None:
    config = _make_config(num_dbs=2)

    records = [{"id": 1}]
    ddf = _make_dask_df(records, npartitions=1)

    with pytest.raises(ConfigValidationError, match="num_dbs=1"):
        write_single_db_dask(
            ddf,
            config,
            key_col="id",
            value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        )


def test_batch_size_controls_write_calls() -> None:
    factory = _TrackingFactory()
    config = _make_config(factory=factory, batch_size=2)

    records = [{"id": k} for k in range(7)]
    ddf = _make_dask_df(records, npartitions=1)

    write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    adapter = factory.adapters[0]
    total_pairs = sum(len(call) for call in adapter.write_calls)
    assert total_pairs == 7
    # With batch_size=2 and 7 rows: at least 3 full batches + 1 partial
    assert len(adapter.write_calls) >= 4


def test_rate_limiting() -> None:
    factory = _TrackingFactory()
    config = _make_config(factory=factory, batch_size=2)

    records = [{"id": k} for k in range(5)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        max_writes_per_second=1000.0,
    )

    assert result.stats.rows_written == 5


def test_empty_dataframe() -> None:
    factory = _TrackingFactory()
    config = _make_config(factory=factory)

    pdf = pd.DataFrame({"id": pd.Series(dtype="int64")})
    ddf = dd.from_pandas(pdf, npartitions=1)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].row_count == 0
    assert result.winners[0].min_key is None
    assert result.winners[0].max_key is None


def test_min_max_keys() -> None:
    factory = _TrackingFactory()
    config = _make_config(factory=factory)

    records = [{"id": k} for k in [50, 10, 90, 30]]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].min_key == 10
    assert result.winners[0].max_key == 90


def test_manifest_structure() -> None:
    publisher = _InMemoryPublisher()
    config = WriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        adapter_factory=_TrackingFactory(),
        output=OutputOptions(run_id="test-manifest"),
        manifest=ManifestOptions(publisher=publisher),
    )

    records = [{"id": k} for k in range(5)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.manifest_ref.startswith("mem://manifests/")
    assert result.current_ref == "mem://_CURRENT"

    manifest_artifact = publisher.objects[result.manifest_ref]
    manifest_data = json.loads(manifest_artifact.payload.decode("utf-8"))
    assert manifest_data["required"]["num_dbs"] == 1
    assert len(manifest_data["shards"]) == 1
    assert manifest_data["shards"][0]["db_id"] == 0


def test_key_encoding_u32be() -> None:
    factory = _TrackingFactory()
    config = _make_config(factory=factory, key_encoding=KeyEncoding.U32BE)

    records = [{"id": 1}, {"id": 256}]
    ddf = _make_dask_df(records, npartitions=1)

    write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    adapter = factory.adapters[0]
    all_keys = [pair[0] for call in adapter.write_calls for pair in call]
    assert all(len(k) == 4 for k in all_keys)
    assert all_keys[0] == b"\x00\x00\x00\x01"
    assert all_keys[1] == b"\x00\x00\x01\x00"


def test_cache_input_false() -> None:
    factory = _TrackingFactory()
    config = _make_config(factory=factory)

    records = [{"id": k} for k in range(5)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        cache_input=False,
    )

    assert result.stats.rows_written == 5


def test_prefetch_and_no_prefetch_same_result() -> None:
    records = [{"id": k} for k in range(20)]
    ddf = _make_dask_df(records, npartitions=4)

    factory1 = _TrackingFactory()
    config1 = _make_config(factory=factory1)
    result1 = write_single_db_dask(
        ddf,
        config1,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        prefetch_partitions=True,
    )

    factory2 = _TrackingFactory()
    config2 = _make_config(factory=factory2)
    result2 = write_single_db_dask(
        ddf,
        config2,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        prefetch_partitions=False,
    )

    assert result1.stats.rows_written == result2.stats.rows_written
    assert result1.winners[0].row_count == result2.winners[0].row_count


def test_dask_cache_context_enabled_false() -> None:
    pdf = pd.DataFrame({"id": [1, 2, 3]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    with DaskCacheContext(ddf, enabled=False) as result_ddf:
        assert result_ddf is ddf


def test_dask_cache_context_enabled_true() -> None:
    pdf = pd.DataFrame({"id": [1, 2, 3]})
    ddf = dd.from_pandas(pdf, npartitions=1)

    with DaskCacheContext(ddf, enabled=True) as result_ddf:
        # Persisted Dask DataFrames are a different object
        assert result_ddf is not ddf


def test_checkpoint_id_in_result() -> None:
    factory = _TrackingFactory()
    config = _make_config(factory=factory)

    records = [{"id": 1}]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.winners[0].checkpoint_id == "fake-checkpoint"


def test_explicit_num_partitions_skips_count() -> None:
    """When num_partitions is provided, len(ddf) is not called."""
    factory = _TrackingFactory()
    config = _make_config(factory=factory, batch_size=2)

    records = [{"id": k} for k in range(10)]
    ddf = _make_dask_df(records, npartitions=2)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
        num_partitions=3,
    )

    assert result.winners[0].row_count == 10


def test_shard_duration_is_zero() -> None:
    """Single-db mode has no sharding phase; shard_duration_ms must be 0."""
    factory = _TrackingFactory()
    config = _make_config(factory=factory)

    records = [{"id": k} for k in range(5)]
    ddf = _make_dask_df(records, npartitions=1)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(lambda row: b"v"),
    )

    assert result.stats.durations.sharding_ms == 0


def test_data_integrity_file_backed(tmp_path: pathlib.Path) -> None:
    root_dir = str(tmp_path / "file_backed")
    config = WriteConfig(
        num_dbs=1,
        s3_prefix="s3://bucket/prefix",
        adapter_factory=file_backed_adapter_factory(root_dir),
        output=OutputOptions(
            run_id="test-run",
            local_root=str(tmp_path / "local"),
        ),
        manifest=ManifestOptions(publisher=_InMemoryPublisher()),
    )

    records = [{"id": i} for i in range(50)]
    ddf = _make_dask_df(records, npartitions=3)

    result = write_single_db_dask(
        ddf,
        config,
        key_col="id",
        value_spec=ValueSpec.callable_encoder(
            lambda row: f"value-{row['id']}".encode()
        ),
    )

    assert result.stats.rows_written == 50

    all_kv: dict[bytes, bytes] = {}
    for winner in result.winners:
        shard_data = file_backed_load_db(root_dir, winner.db_url)
        all_kv.update(shard_data)

    assert len(all_kv) == 50
    for i in range(50):
        key_bytes = make_key_encoder(config.key_encoding)(i)
        assert all_kv[key_bytes] == f"value-{i}".encode()
