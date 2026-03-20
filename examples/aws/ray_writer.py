"""Write a sharded snapshot to AWS S3 using the Ray writer.

Prerequisites:
    pip install shardyfusion[writer-ray]

Authentication:
    Uses the default boto3 credential chain.
"""

import ray
import ray.data

from shardyfusion import WriteConfig
from shardyfusion.writer.ray import write_sharded

ray.init()

# --- Sample data ---------------------------------------------------------

ds = ray.data.from_items(
    [{"id": i, "payload": f"value-{i}"} for i in range(50_000)]
)

# --- Write to S3 ---------------------------------------------------------

config = WriteConfig(
    num_dbs=8,
    s3_prefix="s3://my-bucket/snapshots/features",
)

result = write_sharded(
    ds,
    config,
    key_col="id",
    value_spec={"binary_col": "payload"},
)

print(f"Snapshot written: {result.manifest_ref}")
print(f"Shards: {len(result.shards)}")

ray.shutdown()
