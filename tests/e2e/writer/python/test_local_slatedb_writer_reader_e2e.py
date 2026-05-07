"""Real LocalSlateDbAdapter → Garage → SlateDbReader round-trip.

The parameterised ``backend`` fixture covers a file-backed ``slatedb``
variant that bypasses Garage and a ``sqlite`` variant that uses it; this
test fills the remaining gap — SlateDB *and* Garage end-to-end — via the
dedicated :func:`local_slatedb_backend` fixture.
"""

from __future__ import annotations

import pytest

from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)
from tests.helpers.s3_test_scenarios import run_python_writer_reader_refresh_scenario


@pytest.mark.e2e
def test_local_slatedb_garage_writer_reader_round_trip(
    tmp_path, garage_s3_service, local_slatedb_backend
) -> None:
    """Write via ``LocalSlateDbFactory`` → Garage; read via ``SlateDbReaderFactory``.

    Reuses :func:`run_python_writer_reader_refresh_scenario` so the
    refresh path (write v1 → reader opens → write v2 → ``refresh()`` →
    reads see v2) is also covered, not just the initial open.
    """
    run_python_writer_reader_refresh_scenario(
        garage_s3_service,
        tmp_path,
        adapter_factory=local_slatedb_backend.adapter_factory,
        reader_factory=local_slatedb_backend.reader_factory,
        credential_provider=credential_provider_from_service(garage_s3_service),
        s3_connection_options=s3_connection_options_from_service(garage_s3_service),
    )
