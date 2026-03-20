"""Write a sharded snapshot to Google Cloud Storage using the Spark writer.

GCS provides an S3-compatible XML API accessed via HMAC keys.

Prerequisites:
    pip install shardyfusion[writer-spark]
    Java 17 required.

See: https://cloud.google.com/storage/docs/interoperability
"""

from pyspark.sql import SparkSession

from shardyfusion import WriteConfig
from shardyfusion.config import ManifestOptions
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.spark import write_sharded

spark = SparkSession.builder.appName("shardyfusion-gcp").getOrCreate()

# --- GCS connection via S3-compatible API --------------------------------

GCS_HMAC_ACCESS_KEY = "GOOG..."
GCS_HMAC_SECRET = "..."

gcs_creds = StaticCredentialProvider(
    access_key_id=GCS_HMAC_ACCESS_KEY,
    secret_access_key=GCS_HMAC_SECRET,
)
gcs_conn = S3ConnectionOptions(
    endpoint_url="https://storage.googleapis.com",
    region_name="auto",
)

# --- Sample data ---------------------------------------------------------

data = [{"id": i, "payload": f"value-{i}"} for i in range(100_000)]
df = spark.createDataFrame(data)

# --- Write to GCS -------------------------------------------------------

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-gcs-bucket/snapshots/features",
    credential_provider=gcs_creds,
    s3_connection_options=gcs_conn,
    manifest=ManifestOptions(
        credential_provider=gcs_creds,
        s3_connection_options=gcs_conn,
    ),
)

result = write_sharded(
    df,
    config,
    key_col="id",
    value_spec={"binary_col": "payload"},
)

print(f"Snapshot written: {result.manifest_ref}")

spark.stop()
