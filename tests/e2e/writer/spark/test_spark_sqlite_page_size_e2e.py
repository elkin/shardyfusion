from __future__ import annotations

import pytest

from tests.helpers.sqlite_page_size_e2e import (
    PAGE_SIZE_MODES,
    run_sqlite_page_size_e2e_scenario,
)


@pytest.mark.e2e
@pytest.mark.spark
@pytest.mark.parametrize(
    "mode_id, factory_kwargs, expected_page_size",
    PAGE_SIZE_MODES,
    ids=[m[0] for m in PAGE_SIZE_MODES],
)
def test_spark_sqlite_page_size_round_trip(
    spark,
    garage_s3_service,
    tmp_path,
    mode_id,
    factory_kwargs,
    expected_page_size,
) -> None:
    pytest.importorskip("apsw")
    from tests.helpers.writer_api import write_spark_hash_sharded

    run_sqlite_page_size_e2e_scenario(
        writer_fn=write_spark_hash_sharded,
        make_dataset=lambda rows: spark.createDataFrame(rows, ["id", "payload"]),
        writer_kind="spark",
        mode_id=mode_id,
        factory_kwargs=factory_kwargs,
        expected_page_size=expected_page_size,
        garage_s3_service=garage_s3_service,
        tmp_path=tmp_path,
    )
