"""Standalone subprocess driver: open a SQLite-backed shard prefix and round-trip keys.

Runs as ``python -m tests.helpers.readers.sqlite_get_driver``. Imports only
``shardyfusion`` and stdlib — no pytest, no test helpers. Used by the
``tests/e2e/writer/{spark,dask,ray}/test_*_sqlite_page_size_e2e.py`` suite to
verify that a freshly-written shard opens in a clean Python process and that
the range reader observes the page_size the writer committed to.

Stdin/stdout contract:
  stdin/argv → arguments via argparse
  stdout      → exactly one JSON document (see schema below) on success
  stderr      → human-readable logs / traceback on failure

Exit codes:
  0 — success
  2 — argparse error (default)
  3 — caught exception during reader setup or read
  4 — structural inconsistency (e.g. shard reader exposed a non-int page_size)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tests.helpers.readers.sqlite_get_driver",
        description="Subprocess driver for SQLite range-reader round-trip e2e tests.",
    )
    p.add_argument("--s3-prefix", required=True, help="e.g. s3://bucket/prefix")
    p.add_argument("--local-root", required=True, help="Local cache dir (per-test)")
    p.add_argument(
        "--keys-json",
        required=True,
        help="Path to JSON file containing a list of integer keys",
    )
    p.add_argument("--endpoint-url", required=True, help="S3 endpoint URL")
    p.add_argument("--region-name", default="garage")
    p.add_argument(
        "--addressing-style",
        default="path",
        choices=("path", "virtual"),
        help="S3 URL addressing style (Garage requires 'path')",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Optional output file; defaults to writing the JSON to stdout",
    )
    return p


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Open the reader, multi_get the keys, and collect the per-shard page_size."""

    from shardyfusion.credentials import StaticCredentialProvider
    from shardyfusion.manifest_store import S3ManifestStore
    from shardyfusion.reader import ShardedReader
    from shardyfusion.sqlite_adapter import SqliteRangeReaderFactory
    from shardyfusion.storage import (
        ObstoreBackend,
        create_s3_store,
        parse_s3_url,
    )
    from shardyfusion.type_defs import S3ConnectionOptions

    access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
    secret_access_key = os.environ["AWS_SECRET_ACCESS_KEY"]

    creds = StaticCredentialProvider(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
    )
    conn_opts = S3ConnectionOptions(
        endpoint_url=args.endpoint_url,
        region_name=args.region_name,
        addressing_style=args.addressing_style,
    )

    reader_factory = SqliteRangeReaderFactory(
        s3_connection_options=conn_opts,
        credential_provider=creds,
    )

    bucket, _ = parse_s3_url(args.s3_prefix.rstrip("/") + "/_CURRENT")
    backend = ObstoreBackend(
        create_s3_store(
            bucket=bucket,
            credentials=creds.resolve(),
            connection_options=conn_opts,
        )
    )
    manifest_store = S3ManifestStore(backend, args.s3_prefix)

    local_root = Path(args.local_root)
    local_root.mkdir(parents=True, exist_ok=True)

    with open(args.keys_json, encoding="utf-8") as fh:
        keys: list[int] = [int(k) for k in json.load(fh)]

    reader = ShardedReader(
        s3_prefix=args.s3_prefix,
        local_root=str(local_root),
        manifest_store=manifest_store,
        reader_factory=reader_factory,
        credential_provider=creds,
        s3_connection_options=conn_opts,
    )
    try:
        # The reader returns {key: bytes | None}; serialize as {str(key): hex | None}
        # so the JSON document round-trips arbitrary binary safely.
        raw = reader.multi_get(keys)
        results = {str(k): (v.hex() if v is not None else None) for k, v in raw.items()}

        observed: dict[str, int] = {}
        for db_id, shard_reader in reader._state.readers.items():
            page_size = getattr(
                getattr(shard_reader, "_s3_file", None), "page_size", None
            )
            if not isinstance(page_size, int):
                raise RuntimeError(
                    f"shard db_id={db_id} reader exposed non-int page_size: "
                    f"{type(shard_reader).__name__}.{type(getattr(shard_reader, '_s3_file', None)).__name__} "
                    f"page_size={page_size!r}"
                )
            observed[str(db_id)] = page_size

        return {
            "results": results,
            "observed_page_sizes": observed,
            "manifest_ref": reader._state.manifest_ref,
            "num_shards": len(reader._state.readers),
        }
    finally:
        reader.close()


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    try:
        payload = _run(args)
    except RuntimeError as exc:
        print(f"[driver] structural error: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 4
    except Exception as exc:
        print(f"[driver] reader setup/read failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 3

    serialized = json.dumps(payload, sort_keys=True)
    if args.out_json:
        Path(args.out_json).write_text(serialized, encoding="utf-8")
    else:
        sys.stdout.write(serialized)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
