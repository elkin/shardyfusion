from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.smoke_scenarios import run_smoke_write_then_read_scenario


@pytest.mark.e2e
@pytest.mark.dask
def test_smoke_write_read_dask(garage_s3_service, tmp_path) -> None:
    import dask
    import dask.dataframe as dd
    import pandas as pd

    from shardyfusion.serde import ValueSpec
    from shardyfusion.writer.dask import write_sharded

    def write_fn(data, config):
        pdf = pd.DataFrame(data, columns=["key", "value"])
        ddf = dd.from_pandas(pdf, npartitions=2)
        with dask.config.set(scheduler="synchronous"):
            return write_sharded(
                ddf, config, key_col="key", value_spec=ValueSpec.binary_col("value")
            )

    run_smoke_write_then_read_scenario(
        write_fn,
        garage_s3_service,
        tmp_path,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
