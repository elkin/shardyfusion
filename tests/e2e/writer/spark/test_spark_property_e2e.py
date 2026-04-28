from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.property_scenarios import (
    PROPERTY_CASE_KINDS,
    PropertyCaseKind,
    PropertyInput,
    property_input_strategy,
    row_dicts,
    run_property_kv_e2e_scenario,
)


def _make_write_fn(spark):
    def write_fn(data, config):
        from shardyfusion.serde import ValueSpec
        from shardyfusion.writer.spark import write_sharded

        df = spark.createDataFrame(row_dicts(data))
        return write_sharded(
            df,
            config,
            key_col="key",
            value_spec=ValueSpec.binary_col("value"),
        )

    return write_fn


@pytest.mark.e2e
@pytest.mark.spark
@pytest.mark.parametrize("kind", PROPERTY_CASE_KINDS)
@given(case=property_input_strategy())
@settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_property_kv_round_trip_and_shard_placement(
    kind: PropertyCaseKind,
    case: PropertyInput,
    spark,
    garage_s3_service,
    tmp_path,
    backend,
) -> None:
    if kind in {PropertyCaseKind.CEL_CONTEXT, PropertyCaseKind.CEL_CATEGORICAL}:
        pytest.importorskip("cel_expr_python")

    run_property_kv_e2e_scenario(
        _make_write_fn(spark),
        garage_s3_service,
        tmp_path,
        kind=kind,
        case=case,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
