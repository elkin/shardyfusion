from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.s3_test_scenarios import (
    run_python_writer_publishes_manifest_scenario,
)


@pytest.mark.e2e
def test_python_writer_publishes_manifest_sequential(
    garage_s3_service, tmp_path, backend
) -> None:
    run_python_writer_publishes_manifest_scenario(
        garage_s3_service,
        tmp_path,
        parallel=False,
        adapter_factory=backend.adapter_factory,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )


@pytest.mark.e2e
def test_python_writer_publishes_manifest_parallel(
    garage_s3_service, tmp_path, backend
) -> None:
    run_python_writer_publishes_manifest_scenario(
        garage_s3_service,
        tmp_path,
        parallel=True,
        adapter_factory=backend.adapter_factory,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
