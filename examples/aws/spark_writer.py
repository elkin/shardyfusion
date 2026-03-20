"""Write a sharded snapshot to AWS S3 using the Spark writer.

Prerequisites:
    pip install shardyfusion[writer-spark]
    Java 17 required.

Authentication:
    Uses the default boto3 credential chain.  On EMR, credentials are
    available automatically via the instance profile.
"""

from pyspark.sql import SparkSession

from shardyfusion import WriteConfig
from shardyfusion.writer.spark import write_sharded

spark = SparkSession.builder.appName("shardyfusion-example").getOrCreate()

# --- Sample data ---------------------------------------------------------

data = [{"id": i, "payload": f"value-{i}"} for i in range(100_000)]
df = spark.createDataFrame(data)

# --- Write to S3 ---------------------------------------------------------

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/features",
)

result = write_sharded(
    df,
    config,
    key_col="id",
    value_spec={"binary_col": "payload"},
)

print(f"Snapshot written: {result.manifest_ref}")
print(f"Shards: {len(result.shards)}")

spark.stop()
