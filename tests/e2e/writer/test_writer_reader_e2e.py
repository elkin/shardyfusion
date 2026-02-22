from __future__ import annotations

import pytest

from tests.e2e.conftest import s3_client_config_from_service
from tests.helpers.s3_test_scenarios import run_writer_reader_refresh_scenario


@pytest.mark.e2e
@pytest.mark.spark
def test_reader_refreshes_after_writer_batch_against_garage(
    spark, tmp_path, garage_s3_service
) -> None:
    run_writer_reader_refresh_scenario(
        spark,
        garage_s3_service,
        tmp_path,
        s3_client_config=s3_client_config_from_service(garage_s3_service),
    )
