from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.s3_test_scenarios import run_dask_writer_publishes_manifest_scenario


@pytest.mark.e2e
@pytest.mark.dask
def test_dask_writer_publishes_manifest_to_garage(garage_s3_service, tmp_path) -> None:
    run_dask_writer_publishes_manifest_scenario(
        garage_s3_service,
        tmp_path,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
