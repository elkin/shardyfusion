from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.smoke_scenarios import run_smoke_write_then_read_scenario


@pytest.mark.e2e
def test_smoke_write_read_python(garage_s3_service, tmp_path) -> None:
    from shardyfusion.writer.python import write_sharded

    def write_fn(data, config):
        return write_sharded(data, config, key_fn=lambda r: r[0], value_fn=lambda r: r[1])

    run_smoke_write_then_read_scenario(
        write_fn,
        garage_s3_service,
        tmp_path,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
