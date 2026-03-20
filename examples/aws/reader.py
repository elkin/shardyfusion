"""Read from a sharded snapshot on AWS S3.

Prerequisites:
    pip install shardyfusion[read]

Authentication:
    Uses the default boto3 credential chain (env vars, ~/.aws/credentials,
    IAM instance profile, ECS task role, etc.).
"""

from shardyfusion import ShardedReader

# --- Single-threaded reader (scripts, single-threaded services) ----------

with ShardedReader(
    s3_prefix="s3://my-bucket/snapshots/features",
    local_root="/tmp/shardyfusion-reader",
) as reader:
    value = reader.get(42)
    print(f"key=42 → {value}")

    batch = reader.multi_get([1, 2, 3, 100])
    for key, val in zip([1, 2, 3, 100], batch):
        print(f"key={key} → {val}")

    # Atomic swap to latest snapshot (e.g. after a new write completes)
    reader.refresh()

    # Inspect snapshot metadata
    info = reader.snapshot_info()
    print(f"Run ID: {info.run_id}, Shards: {info.num_shards}")
