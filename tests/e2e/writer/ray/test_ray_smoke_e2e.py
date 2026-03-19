from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.smoke_scenarios import run_smoke_write_then_read_scenario


@pytest.mark.e2e
@pytest.mark.ray
def test_smoke_write_read_ray(garage_s3_service, tmp_path) -> None:
    import ray.data

    from shardyfusion.serde import ValueSpec
    from shardyfusion.writer.ray import write_sharded

    def write_fn(data, config):
        items = [{"key": k, "value": v} for k, v in data]
        ds = ray.data.from_items(items, override_num_blocks=2)
        return write_sharded(ds, config, key_col="key", value_spec=ValueSpec.binary_col("value"))

    run_smoke_write_then_read_scenario(
        write_fn,
        garage_s3_service,
        tmp_path,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
