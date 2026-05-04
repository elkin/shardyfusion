from __future__ import annotations

from pathlib import Path

from tests.helpers.s3_test_scenarios import run_reader_loads_manifest_scenario


def test_reader_loads_current_and_manifest_from_local_s3(
    tmp_path, local_s3_service
) -> None:
    from shardyfusion.reader._types import SlateDbReaderFactory
    from shardyfusion.testing import (
        map_s3_db_url_to_file_url,
        real_file_adapter_factory,
    )

    object_store_root = str(tmp_path / "object-store")
    _delegate = SlateDbReaderFactory()

    def _reader(  # type: ignore[no-untyped-def]
        *,
        db_url: str,
        local_dir: Path,
        checkpoint_id: str | None,
        manifest=None,
    ):
        # Tests use s3:// URLs but data is materialized on local disk;
        # remap to a file:// URL before opening the shard reader.
        return _delegate(
            db_url=map_s3_db_url_to_file_url(db_url, object_store_root),
            local_dir=local_dir,
            checkpoint_id=checkpoint_id,
            manifest=manifest,
        )

    run_reader_loads_manifest_scenario(
        local_s3_service,
        tmp_path,
        adapter_factory=real_file_adapter_factory(object_store_root),
        reader_factory=_reader,
    )
