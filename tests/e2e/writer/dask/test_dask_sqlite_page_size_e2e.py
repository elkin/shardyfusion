from __future__ import annotations

import pytest

from tests.helpers.sqlite_page_size_e2e import (
    PAGE_SIZE_MODES,
    run_sqlite_page_size_e2e_scenario,
)


def _make_dask_dataset(rows):
    import dask
    import dask.dataframe as dd
    import pandas as pd

    # Dask's dask_expr backend auto-converts object-dtype columns to PyArrow
    # string at compute time; binary payloads with bytes >= 0x80 then fail the
    # UTF-8 decode (e.g. uniqueness suffixes built from int.to_bytes). Disable
    # the conversion globally for this test process — set persists for the
    # session, which is fine: other dask tests in this suite use ASCII payloads
    # and don't care either way.
    dask.config.set({"dataframe.convert-string": False})
    pdf = pd.DataFrame(rows, columns=["id", "payload"])
    return dd.from_pandas(pdf, npartitions=4)


@pytest.mark.e2e
@pytest.mark.dask
@pytest.mark.parametrize(
    "mode_id, factory_kwargs, expected_page_size",
    PAGE_SIZE_MODES,
    ids=[m[0] for m in PAGE_SIZE_MODES],
)
def test_dask_sqlite_page_size_round_trip(
    garage_s3_service,
    tmp_path,
    mode_id,
    factory_kwargs,
    expected_page_size,
) -> None:
    pytest.importorskip("apsw")
    pytest.importorskip("dask.dataframe")
    from tests.helpers.writer_api import write_dask_hash_sharded

    run_sqlite_page_size_e2e_scenario(
        writer_fn=write_dask_hash_sharded,
        make_dataset=_make_dask_dataset,
        writer_kind="dask",
        mode_id=mode_id,
        factory_kwargs=factory_kwargs,
        expected_page_size=expected_page_size,
        garage_s3_service=garage_s3_service,
        tmp_path=tmp_path,
    )
