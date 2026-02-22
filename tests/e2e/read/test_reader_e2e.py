from __future__ import annotations

import pytest

from tests.e2e.conftest import s3_client_config_from_service
from tests.helpers.s3_test_scenarios import run_reader_loads_manifest_scenario


@pytest.mark.e2e
def test_reader_loads_current_and_manifest_from_garage(
    tmp_path, garage_s3_service
) -> None:
    run_reader_loads_manifest_scenario(
        garage_s3_service,
        tmp_path,
        s3_client_config=s3_client_config_from_service(garage_s3_service),
    )
