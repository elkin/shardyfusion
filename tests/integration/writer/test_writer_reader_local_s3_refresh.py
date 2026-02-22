from __future__ import annotations

import pytest

from tests.helpers.s3_test_scenarios import run_writer_reader_refresh_scenario


@pytest.mark.spark
def test_reader_refreshes_after_new_writer_batch(
    spark, tmp_path, local_s3_service
) -> None:
    run_writer_reader_refresh_scenario(spark, local_s3_service, tmp_path)
