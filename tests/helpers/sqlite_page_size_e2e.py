"""Shared e2e scenario: write a SQLite-backed shard at a given page_size mode,
then open the shard in a subprocess and assert it round-trips.

Used by ``tests/e2e/writer/{spark,dask,ray}/test_*_sqlite_page_size_e2e.py``.
The matrix covers every page_size configuration the public API supports:

* 5 explicit ints (4096 / 8192 / 16384 / 32768 / 65536)
* the ``"auto"`` sentinel (post-VACUUM recommender at seal time)
* engine-percentile profiling (``KeyValueWriteConfig.profile_value_sizes_for_page_size``)

For each mode the scenario asserts that:

* the writer published a manifest + per-shard sidecar v6 (``shard.sidecar``)
* the manifest's ``custom.sqlite_sidecar.page_size`` field matches the
  expected page_size (with the documented asymmetry for ``"auto"`` — it stays
  as the literal string ``"auto"`` in the manifest, not the resolved int)
* a subprocess reader (``tests.helpers.readers.sqlite_get_driver``) opens the
  shard prefix in a clean Python process, observes the expected page_size on
  every shard's range-mode reader, and round-trips every key/value
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shardyfusion.config import (
    HashShardedWriteConfig,
    KeyValueWriteConfig,
    WriterManifestConfig,
    WriterOutputConfig,
)
from shardyfusion.manifest import BuildResult
from shardyfusion.manifest_store import S3ManifestStore
from shardyfusion.serde import ValueSpec
from shardyfusion.sqlite_adapter import SqliteFactory
from shardyfusion.storage import ObstoreBackend, create_s3_store, parse_s3_url
from tests.e2e.conftest import (
    credential_provider_from_service,
    s3_connection_options_from_service,
)

if TYPE_CHECKING:
    from tests.conftest import LocalS3Service


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

# Format: (mode_id, factory_kwargs_or_sentinel, expected_resolved_page_size)
#
# ``factory_kwargs`` is the kwargs dict for SqliteFactory in the 6 non-engine
# modes. The string sentinel ``"ENGINE"`` selects the engine-percentile path —
# in that mode the factory is built with default page_size=4096 and the writer
# config carries ``profile_value_sizes_for_page_size=True``.
#
# ``expected_resolved_page_size`` is what the range reader should observe and
# what should appear in the manifest's ``sqlite_sidecar.page_size`` field
# (with the exception: for the ``"auto"`` mode the manifest stores the literal
# string ``"auto"``, not the resolved int — see ``_expected_manifest_page_size``
# below).
PAGE_SIZE_MODES: tuple[tuple[str, Any, int], ...] = (
    ("explicit_4096", {"page_size": 4096}, 4096),
    ("explicit_8192", {"page_size": 8192}, 8192),
    ("explicit_16384", {"page_size": 16384}, 16384),
    ("explicit_32768", {"page_size": 32768}, 32768),
    ("explicit_65536", {"page_size": 65536}, 65536),
    ("auto", {"page_size": "auto"}, 32768),
    ("engine_percentile", "ENGINE", 32768),
)

NUM_DBS = 4


# ---------------------------------------------------------------------------
# Test-data construction
# ---------------------------------------------------------------------------


def _value_size_for_mode(mode_id: str) -> int:
    """Pick a value size that exercises each mode's interesting code path.

    * For the 5 explicit modes a uniform 1200-byte value spills into overflow
      at ``page_size=4096`` (inline threshold 1002) and stays inline at the
      larger sizes. Either way it exercises a non-trivial B-tree.
    * For ``"auto"`` and engine-percentile a 5000-byte value gives
      ``recommend_page_size(p95=5000)`` → 32768 deterministically
      (5000 + 64 key + 12 cell overhead = 5076 → exceeds 16384's 4086 inline
      threshold, fits 32768's 8198 with headroom).
    """
    if mode_id in {"auto", "engine_percentile"}:
        return 5000
    return 1200


def _make_rows(mode_id: str, num_rows: int = 256) -> list[tuple[int, bytes]]:
    """Build (id, payload) tuples; payload is unique per row so we catch
    'writer overwrote the wrong key' regressions."""
    size = _value_size_for_mode(mode_id)
    return [(i, (b"x" * size) + i.to_bytes(8, "big")) for i in range(num_rows)]


# ---------------------------------------------------------------------------
# Config construction
# ---------------------------------------------------------------------------


def _build_config(
    *,
    mode_id: str,
    factory_kwargs: Any,
    s3_prefix: str,
    local_root: str,
    run_id: str,
    s3_service: LocalS3Service,
) -> HashShardedWriteConfig:
    cred = credential_provider_from_service(s3_service)
    conn_opts = s3_connection_options_from_service(s3_service)

    if factory_kwargs == "ENGINE":
        # Engine-percentile mode: factory keeps the default page_size=4096
        # (the "no choice made" baseline) and the kv sub-config flips the
        # profiling flag, which the writer reads to scan input and substitute
        # the factory's page_size via dataclasses.replace before dispatch.
        factory = SqliteFactory(
            s3_connection_options=conn_opts,
            credential_provider=cred,
        )
        kv = KeyValueWriteConfig(
            adapter_factory=factory,
            profile_value_sizes_for_page_size=True,
        )
        return HashShardedWriteConfig(
            num_dbs=NUM_DBS,
            s3_prefix=s3_prefix,
            credential_provider=cred,
            s3_connection_options=conn_opts,
            output=WriterOutputConfig(run_id=run_id, local_root=local_root),
            kv=kv,
            manifest=WriterManifestConfig(
                credential_provider=cred,
                s3_connection_options=conn_opts,
            ),
        )

    assert isinstance(factory_kwargs, dict)
    factory = SqliteFactory(
        s3_connection_options=conn_opts,
        credential_provider=cred,
        **factory_kwargs,
    )
    return HashShardedWriteConfig(
        num_dbs=NUM_DBS,
        s3_prefix=s3_prefix,
        credential_provider=cred,
        s3_connection_options=conn_opts,
        output=WriterOutputConfig(run_id=run_id, local_root=local_root),
        adapter_factory=factory,
        manifest=WriterManifestConfig(
            credential_provider=cred,
            s3_connection_options=conn_opts,
        ),
    )


# ---------------------------------------------------------------------------
# Sidecar + manifest assertions (run in the writer process)
# ---------------------------------------------------------------------------


def _expected_manifest_page_size(mode_id: str, expected_resolved: int) -> Any:
    """``"auto"`` keeps the literal string in the manifest; everything else is
    the resolved int. See ``_writer_core.inject_sqlite_sidecar_manifest_field``
    and ``sqlite_adapter._detect_sidecar_page_size`` for the asymmetry."""
    if mode_id == "auto":
        return "auto"
    return expected_resolved


def _assert_sidecar_present_for_every_shard(
    result: BuildResult, s3_service: LocalS3Service
) -> None:
    """boto3 head_object on shard.sidecar for each winner db_url."""
    from shardyfusion.sqlite_adapter import _SIDECAR_FILENAME

    client = s3_service["client"]
    bucket = s3_service["bucket"]
    for winner in result.winners:
        assert winner.db_url is not None, (
            f"winner db_id={winner.db_id} has no db_url after a SQLite-backed write"
        )
        sidecar_url = f"{winner.db_url.rstrip('/')}/{_SIDECAR_FILENAME}"
        _, key = parse_s3_url(sidecar_url)
        # Raises ClientError(404) if missing — surfaces as a test failure.
        client.head_object(Bucket=bucket, Key=key)


def _assert_manifest_page_size(
    result: BuildResult,
    s3_service: LocalS3Service,
    s3_prefix: str,
    expected_manifest_page_size: Any,
) -> None:
    cred = credential_provider_from_service(s3_service)
    conn_opts = s3_connection_options_from_service(s3_service)
    bucket, _ = parse_s3_url(s3_prefix.rstrip("/") + "/_CURRENT")
    backend = ObstoreBackend(
        create_s3_store(
            bucket=bucket,
            credentials=cred.resolve(),
            connection_options=conn_opts,
        )
    )
    store = S3ManifestStore(backend, s3_prefix)
    parsed = store.load_manifest(result.manifest_ref)

    sidecar = parsed.custom.get("sqlite_sidecar")
    assert sidecar is not None, (
        f"manifest at {result.manifest_ref} is missing 'sqlite_sidecar' "
        f"custom field; got custom={parsed.custom}"
    )
    assert sidecar.get("page_size") == expected_manifest_page_size, (
        f"manifest sqlite_sidecar.page_size mismatch: "
        f"expected {expected_manifest_page_size!r}, got {sidecar.get('page_size')!r} "
        f"(full sidecar block: {sidecar})"
    )


# ---------------------------------------------------------------------------
# Subprocess reader invocation
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """Repo root resolved from this file's location: tests/helpers/<file>."""
    return Path(__file__).resolve().parents[2]


def _invoke_subprocess_reader(
    *,
    s3_prefix: str,
    local_root: Path,
    keys: list[int],
    s3_service: LocalS3Service,
) -> dict[str, Any]:
    keys_path = local_root / "keys.json"
    keys_path.parent.mkdir(parents=True, exist_ok=True)
    keys_path.write_text(json.dumps(keys), encoding="utf-8")

    env = {
        **os.environ,
        "AWS_ACCESS_KEY_ID": s3_service["access_key_id"],
        "AWS_SECRET_ACCESS_KEY": s3_service["secret_access_key"],
        "AWS_REGION": s3_service["region_name"],
        "PYTHONPATH": str(_repo_root()),
    }

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.helpers.readers.sqlite_get_driver",
            "--s3-prefix",
            s3_prefix,
            "--local-root",
            str(local_root / "reader-cache"),
            "--keys-json",
            str(keys_path),
            "--endpoint-url",
            s3_service["endpoint_url"],
            "--region-name",
            s3_service["region_name"],
            "--addressing-style",
            "path",
        ],
        env=env,
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess reader exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    return json.loads(result.stdout)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_sqlite_page_size_e2e_scenario(
    *,
    writer_fn: Callable[..., BuildResult],
    make_dataset: Callable[[list[tuple[int, bytes]]], Any],
    writer_kind: str,
    mode_id: str,
    factory_kwargs: Any,
    expected_page_size: int,
    garage_s3_service: LocalS3Service,
    tmp_path: Path,
) -> None:
    """Writer publishes, subprocess reader rounds-trips, every shard's
    page_size matches ``expected_page_size``."""

    s3_prefix = (
        f"s3://{garage_s3_service['bucket']}/page-size-e2e/"
        f"{writer_kind}/{mode_id}/{tmp_path.name}"
    )
    run_id = f"{writer_kind}-{mode_id}-{tmp_path.name}"
    local_root = tmp_path / "writer-local"

    config = _build_config(
        mode_id=mode_id,
        factory_kwargs=factory_kwargs,
        s3_prefix=s3_prefix,
        local_root=str(local_root),
        run_id=run_id,
        s3_service=garage_s3_service,
    )
    rows = _make_rows(mode_id)
    dataset = make_dataset(rows)

    result = writer_fn(
        dataset,
        config,
        key_col="id",
        value_spec=ValueSpec.binary_col("payload"),
    )

    assert len(result.winners) == NUM_DBS
    total_row_count = sum(w.row_count for w in result.winners)
    assert total_row_count == len(rows), (
        f"published shards cover {total_row_count} rows but the input had {len(rows)}"
    )

    _assert_sidecar_present_for_every_shard(result, garage_s3_service)
    _assert_manifest_page_size(
        result,
        garage_s3_service,
        s3_prefix,
        _expected_manifest_page_size(mode_id, expected_page_size),
    )

    keys = [k for k, _ in rows]
    payload = _invoke_subprocess_reader(
        s3_prefix=s3_prefix,
        local_root=tmp_path,
        keys=keys,
        s3_service=garage_s3_service,
    )

    assert payload["num_shards"] == NUM_DBS, (
        f"subprocess saw {payload['num_shards']} shards, expected {NUM_DBS}"
    )

    observed = payload["observed_page_sizes"]
    distinct = set(observed.values())
    assert distinct == {expected_page_size}, (
        f"observed page sizes {sorted(distinct)} on shards {observed}; "
        f"expected every shard to be {expected_page_size}"
    )

    expected_values = {str(k): v.hex() for k, v in rows}
    assert payload["results"] == expected_values, (
        "round-tripped values do not match the input "
        f"(first mismatch among {len(expected_values)} keys; "
        f"sample expected={next(iter(expected_values.items()))}, "
        f"sample got={next(iter(payload['results'].items()))})"
    )
