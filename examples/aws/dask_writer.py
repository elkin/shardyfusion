"""Write a sharded snapshot to AWS S3 using the Dask writer.

Prerequisites:
    pip install shardyfusion[writer-dask]

Authentication:
    Uses the default boto3 credential chain.
"""

import dask.dataframe as dd
import pandas as pd

from shardyfusion import WriteConfig
from shardyfusion.writer.dask import write_sharded

# --- Sample data ---------------------------------------------------------

pdf = pd.DataFrame({"id": range(50_000), "payload": [f"value-{i}" for i in range(50_000)]})
ddf = dd.from_pandas(pdf, npartitions=8)

# --- Write to S3 ---------------------------------------------------------

config = WriteConfig(
    num_dbs=8,
    s3_prefix="s3://my-bucket/snapshots/features",
)

result = write_sharded(
    ddf,
    config,
    key_col="id",
    value_spec={"binary_col": "payload"},
)

print(f"Snapshot written: {result.manifest_ref}")
print(f"Shards: {len(result.shards)}")
