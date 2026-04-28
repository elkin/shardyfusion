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
    run_property_kv_e2e_scenario,
)


def _write_fn(data, config):
    from shardyfusion.writer.python import write_sharded

    return write_sharded(
        data,
        config,
        key_fn=lambda row: row.key,
        value_fn=lambda row: row.value,
        columns_fn=lambda row: {"tenant_id": row.tenant_id, "group": row.group},
    )


@pytest.mark.e2e
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
    garage_s3_service,
    tmp_path,
    backend,
) -> None:
    if kind in {PropertyCaseKind.CEL_CONTEXT, PropertyCaseKind.CEL_CATEGORICAL}:
        pytest.importorskip("cel_expr_python")

    run_property_kv_e2e_scenario(
        _write_fn,
        garage_s3_service,
        tmp_path,
        kind=kind,
        case=case,
        adapter_factory=backend.adapter_factory,
        reader_factory=backend.reader_factory,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
