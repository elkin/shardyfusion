from __future__ import annotations

import pytest

from tests.e2e.conftest import s3_client_config_from_service
from tests.helpers.s3_test_scenarios import run_dask_writer_publishes_manifest_scenario


@pytest.mark.e2e
@pytest.mark.dask
def test_dask_writer_publishes_manifest_to_garage(garage_s3_service, tmp_path) -> None:
    run_dask_writer_publishes_manifest_scenario(
        garage_s3_service,
        tmp_path,
        s3_client_config=s3_client_config_from_service(garage_s3_service),
    )
