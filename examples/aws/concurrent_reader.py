"""Thread-safe reader for multi-threaded services on AWS S3.

Use ConcurrentShardedReader when serving lookups from Flask, Django, or
any multi-threaded Python service.

Prerequisites:
    pip install shardyfusion[read]

Authentication:
    Uses the default boto3 credential chain.
"""

from shardyfusion import ConcurrentShardedReader

# --- Lock mode (default) — simple mutex around shard access --------------

reader = ConcurrentShardedReader(
    s3_prefix="s3://my-bucket/snapshots/features",
    local_root="/tmp/shardyfusion-reader",
    thread_safety="lock",
)

value = reader.get(42)

# Refresh from another thread (e.g. a background scheduler)
reader.refresh()

reader.close()

# --- Pool mode — configurable checkout timeout --------------------------

reader = ConcurrentShardedReader(
    s3_prefix="s3://my-bucket/snapshots/features",
    local_root="/tmp/shardyfusion-reader",
    thread_safety="pool",
    pool_checkout_timeout=10.0,  # seconds
)

with reader:
    value = reader.get(42)
    batch = reader.multi_get([1, 2, 3])
