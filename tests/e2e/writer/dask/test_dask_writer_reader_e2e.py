from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.s3_test_scenarios import run_dask_writer_reader_refresh_scenario


@pytest.mark.e2e
@pytest.mark.dask
def test_reader_refreshes_after_dask_writer_batch_against_garage(
    garage_s3_service, tmp_path
) -> None:
    run_dask_writer_reader_refresh_scenario(
        garage_s3_service,
        tmp_path,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
