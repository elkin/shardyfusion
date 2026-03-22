from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.s3_test_scenarios import run_writer_reader_refresh_scenario


@pytest.mark.e2e
@pytest.mark.spark
def test_reader_refreshes_after_writer_batch_against_garage(
    spark, tmp_path, garage_s3_service, backend
) -> None:
    run_writer_reader_refresh_scenario(
        spark,
        garage_s3_service,
        tmp_path,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
