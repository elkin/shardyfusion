from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.smoke_scenarios import SMOKE_DATA, run_smoke_write_then_read_scenario


@pytest.mark.e2e
@pytest.mark.spark
def test_smoke_write_read_spark(spark, garage_s3_service, tmp_path) -> None:
    from shardyfusion.serde import ValueSpec
    from shardyfusion.writer.spark import write_sharded

    def write_fn(data, config):
        df = spark.createDataFrame(data, ["key", "value"])
        return write_sharded(df, config, key_col="key", value_spec=ValueSpec.binary_col("value"))

    run_smoke_write_then_read_scenario(
        write_fn,
        garage_s3_service,
        tmp_path,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
