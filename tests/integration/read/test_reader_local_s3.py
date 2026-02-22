from __future__ import annotations

from tests.helpers.s3_test_scenarios import run_reader_loads_manifest_scenario


def test_reader_loads_current_and_manifest_from_local_s3(
    tmp_path, local_s3_service
) -> None:
    run_reader_loads_manifest_scenario(local_s3_service, tmp_path)
