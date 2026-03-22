from __future__ import annotations

import pytest

from tests.helpers.s3_test_scenarios import run_writer_publishes_manifest_scenario


@pytest.mark.spark
def test_writer_publishes_manifest_and_current_to_local_s3(
    spark, local_s3_service, tmp_path
) -> None:
    from shardyfusion.testing import real_file_adapter_factory

    object_store_root = str(tmp_path / "object-store")

    run_writer_publishes_manifest_scenario(
        spark,
        local_s3_service,
        tmp_path,
        adapter_factory=real_file_adapter_factory(object_store_root),
    )
