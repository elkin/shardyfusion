from __future__ import annotations

import pytest

from tests.helpers.sqlite_page_size_e2e import (
    PAGE_SIZE_MODES,
    run_sqlite_page_size_e2e_scenario,
)


def _make_ray_dataset(rows):
    import ray.data

    return ray.data.from_items(
        [{"id": k, "payload": v} for k, v in rows],
        override_num_blocks=4,
    )


@pytest.mark.e2e
@pytest.mark.ray
@pytest.mark.parametrize(
    "mode_id, factory_kwargs, expected_page_size",
    PAGE_SIZE_MODES,
    ids=[m[0] for m in PAGE_SIZE_MODES],
)
def test_ray_sqlite_page_size_round_trip(
    garage_s3_service,
    tmp_path,
    mode_id,
    factory_kwargs,
    expected_page_size,
) -> None:
    pytest.importorskip("apsw")
    pytest.importorskip("ray.data")
    from tests.helpers.writer_api import write_ray_hash_sharded

    run_sqlite_page_size_e2e_scenario(
        writer_fn=write_ray_hash_sharded,
        make_dataset=_make_ray_dataset,
        writer_kind="ray",
        mode_id=mode_id,
        factory_kwargs=factory_kwargs,
        expected_page_size=expected_page_size,
        garage_s3_service=garage_s3_service,
        tmp_path=tmp_path,
    )
