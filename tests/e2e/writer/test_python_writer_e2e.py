from __future__ import annotations

import pytest

from tests.e2e.conftest import s3_client_config_from_service
from tests.helpers.s3_test_scenarios import (
    run_python_writer_publishes_manifest_scenario,
)


@pytest.mark.e2e
def test_python_writer_publishes_manifest_sequential(
    garage_s3_service, tmp_path
) -> None:
    run_python_writer_publishes_manifest_scenario(
        garage_s3_service,
        tmp_path,
        parallel=False,
        s3_client_config=s3_client_config_from_service(garage_s3_service),
    )


@pytest.mark.e2e
def test_python_writer_publishes_manifest_parallel(garage_s3_service, tmp_path) -> None:
    run_python_writer_publishes_manifest_scenario(
        garage_s3_service,
        tmp_path,
        parallel=True,
        s3_client_config=s3_client_config_from_service(garage_s3_service),
    )
