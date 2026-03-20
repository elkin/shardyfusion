#!/usr/bin/env python3
"""Playground script — writes sample data, reads it back, launches the shardy REPL.

Uses a local Garage instance (S3-compatible) started by playground/compose.yaml.

Run via: just playground
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from typing import Any

# ── Colour helpers ───────────────────────────────────────────────────────────

NO_COLOR = os.environ.get("NO_COLOR") is not None


def _c(code: str, text: str) -> str:
    return text if NO_COLOR else f"\033[{code}m{text}\033[0m"


def bold(t: str) -> str:
    return _c("1", t)


def green(t: str) -> str:
    return _c("32", t)


def cyan(t: str) -> str:
    return _c("36", t)


def dim(t: str) -> str:
    return _c("2", t)


def heading(t: str) -> str:
    return bold(cyan(t))


# ── Garage config ────────────────────────────────────────────────────────────

ENDPOINT = os.environ.get("PLAYGROUND_ENDPOINT", "http://localhost:3900")
ADMIN_URL = os.environ.get("PLAYGROUND_ADMIN_URL", "http://localhost:3903")
ADMIN_TOKEN = "playground-admin-token"
REGION = "garage"
BUCKET = "playground"
S3_PREFIX = f"s3://{BUCKET}/demo"
NUM_SHARDS = 4
NUM_RECORDS = 10_000

PLANS = ["free", "starter", "pro", "enterprise"]


# ── Garage admin helpers ─────────────────────────────────────────────────────


def _admin_request(
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
) -> Any:
    """Make an HTTP request to the Garage Admin API."""
    url = f"{ADMIN_URL}/v2{path}"
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {ADMIN_TOKEN}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def _setup_garage() -> tuple[str, str]:
    """Create an API key and bucket on the local Garage instance.

    Returns (access_key_id, secret_access_key).
    """
    # Create API key
    key_resp = _admin_request(
        "/CreateKey",
        method="POST",
        body={"name": "playground-key"},
    )
    access_key_id: str = key_resp["accessKeyId"]
    secret_access_key: str = key_resp["secretAccessKey"]

    # Create bucket
    _admin_request(
        "/CreateBucket",
        method="POST",
        body={"globalAlias": BUCKET},
    )

    # Get bucket ID and grant permissions
    bucket_info = _admin_request(f"/GetBucketInfo?globalAlias={BUCKET}")
    bucket_id = bucket_info["id"]

    _admin_request(
        "/AllowBucketKey",
        method="POST",
        body={
            "bucketId": bucket_id,
            "accessKeyId": access_key_id,
            "permissions": {"read": True, "write": True, "owner": True},
        },
    )

    return access_key_id, secret_access_key


# ── Helpers ──────────────────────────────────────────────────────────────────


def _generate_records() -> list[dict[str, object]]:
    """Generate sample user profiles."""
    records: list[dict[str, object]] = []
    for i in range(NUM_RECORDS):
        records.append(
            {
                "id": i,
                "name": f"user-{i}",
                "email": f"user-{i}@example.com",
                "signup_day": f"2025-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "plan": PLANS[i % len(PLANS)],
            }
        )
    return records


def _bar(count: int, total: int, width: int = 30) -> str:
    filled = int(width * count / total) if total else 0
    bar_str = "\u2588" * filled + "\u2591" * (width - filled)
    return bar_str


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    from shardyfusion import ShardedReader, WriteConfig
    from shardyfusion.config import ManifestOptions
    from shardyfusion.credentials import StaticCredentialProvider
    from shardyfusion.type_defs import S3ConnectionOptions
    from shardyfusion.writer.python import write_sharded

    # ── Step 1: Setup Garage (API key + bucket) ──────────────────────────
    print()
    print(heading("  1. Preparing Garage"))
    access_key, secret_key = _setup_garage()
    print(f"     {green('ok')}  bucket '{BUCKET}' ready at {ENDPOINT}")

    creds = StaticCredentialProvider(
        access_key_id=access_key,
        secret_access_key=secret_key,
    )
    conn = S3ConnectionOptions(
        endpoint_url=ENDPOINT,
        region_name=REGION,
        addressing_style="path",
    )

    # ── Step 2: Write snapshot ───────────────────────────────────────────
    print()
    print(heading("  2. Writing sharded snapshot"))
    print(f"     {NUM_RECORDS:,} user profiles -> {NUM_SHARDS} shards")

    records = _generate_records()

    config = WriteConfig(
        num_dbs=NUM_SHARDS,
        s3_prefix=S3_PREFIX,
        credential_provider=creds,
        s3_connection_options=conn,
        manifest=ManifestOptions(
            credential_provider=creds,
            s3_connection_options=conn,
        ),
    )

    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r["id"],
        value_fn=lambda r: json.dumps(
            {k: v for k, v in r.items() if k != "id"},
            separators=(",", ":"),
        ).encode(),
    )

    print(f"     {green('ok')}  snapshot written (run_id={result.run_id})")

    # ── Step 3: Read it back ─────────────────────────────────────────────
    print()
    print(heading("  3. Reading snapshot"))

    with ShardedReader(
        s3_prefix=S3_PREFIX,
        local_root="/tmp/shardyfusion-playground",
        credential_provider=creds,
        s3_connection_options=conn,
    ) as reader:
        # Single key lookup
        sample_key = 42
        value = reader.get(sample_key)
        if value is not None:
            parsed = json.loads(value)
            print(f"     get({sample_key}) -> {parsed}")
        else:
            print(f"     get({sample_key}) -> None")

        # Batch lookup
        batch_keys = [0, 100, 500, 9999]
        batch = reader.multi_get(batch_keys)
        print(f"     multi_get({batch_keys}) -> {len([v for v in batch.values() if v is not None])} found")

        # Snapshot info
        info = reader.snapshot_info()
        print()
        print(heading("  4. Snapshot info"))
        print(f"     run_id:      {info.run_id}")
        print(f"     shards:      {info.num_shards}")
        print(f"     total rows:  {info.row_count:,}")
        print(f"     key encoding: {info.key_encoding}")
        print(f"     sharding:    {info.sharding}")

        # Shard distribution
        details = reader.shard_details()
        print()
        print(heading("  5. Shard distribution"))
        total_rows = sum(d.row_count for d in details)
        for d in details:
            bar = _bar(d.row_count, total_rows)
            pct = (d.row_count / total_rows * 100) if total_rows else 0
            print(f"     shard {d.db_id:>2}  {bar}  {d.row_count:>5,} rows ({pct:.1f}%)")

    # ── Step 4: Launch REPL ──────────────────────────────────────────────
    print()
    print(heading("  6. Launching shardy REPL"))
    print(dim("     Type 'help' for commands, Ctrl-D to exit."))
    print()

    # Build the shardy CLI command
    shardy_args = [
        sys.executable, "-m", "shardyfusion.cli.app",
        "--s3-prefix", S3_PREFIX,
        "--s3-option", f"endpoint_url={ENDPOINT}",
        "--s3-option", f"region={REGION}",
        "--s3-option", "addressing_style=path",
        "--s3-option", f"access_key_id={access_key}",
        "--s3-option", f"secret_access_key={secret_key}",
    ]
    os.execvp(sys.executable, shardy_args)


if __name__ == "__main__":
    main()
