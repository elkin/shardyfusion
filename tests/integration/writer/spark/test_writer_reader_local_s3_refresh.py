from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.s3_test_scenarios import run_writer_reader_refresh_scenario


@pytest.mark.spark
def test_reader_refreshes_after_new_writer_batch(
    spark, tmp_path, local_s3_service
) -> None:
    import slatedb

    from shardyfusion.testing import (
        local_dir_for_file_shard,
        map_s3_db_url_to_file_url,
        real_file_adapter_factory,
    )

    object_store_root = str(tmp_path / "object-store")

    def _reader(*, db_url: str, local_dir: Path, checkpoint_id: str | None, manifest=None):  # type: ignore[no-untyped-def]
        return slatedb.SlateDBReader(
            str(local_dir_for_file_shard(object_store_root, db_url)),
            url=map_s3_db_url_to_file_url(db_url, object_store_root),
            checkpoint_id=checkpoint_id,
        )

    run_writer_reader_refresh_scenario(
        spark,
        local_s3_service,
        tmp_path,
        adapter_factory=real_file_adapter_factory(object_store_root),
        reader_factory=_reader,
    )
