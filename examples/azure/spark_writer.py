"""Write a sharded snapshot via an S3-compatible gateway on Azure with Spark.

Azure Blob Storage does not natively expose an S3-compatible API.
Deploy an S3-compatible server (e.g. Garage, SeaweedFS) on Azure.
See python_writer.py for setup details.

Prerequisites:
    pip install shardyfusion[writer-spark]
    Java 17 required.
"""

from pyspark.sql import SparkSession

from shardyfusion import WriteConfig
from shardyfusion.config import ManifestOptions
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.spark import write_sharded

spark = SparkSession.builder.appName("shardyfusion-azure").getOrCreate()

# --- S3-compatible gateway connection ------------------------------------

GATEWAY_ENDPOINT = "http://localhost:3900"
ACCESS_KEY = "..."
SECRET_KEY = "..."

azure_creds = StaticCredentialProvider(
    access_key_id=ACCESS_KEY,
    secret_access_key=SECRET_KEY,
)
azure_conn = S3ConnectionOptions(
    endpoint_url=GATEWAY_ENDPOINT,
    region_name="garage",
    addressing_style="path",
)

# --- Sample data ---------------------------------------------------------

data = [{"id": i, "payload": f"value-{i}"} for i in range(100_000)]
df = spark.createDataFrame(data)

# --- Write ---------------------------------------------------------------

config = WriteConfig(
    num_dbs=16,
    s3_prefix="s3://my-bucket/snapshots/features",
    credential_provider=azure_creds,
    s3_connection_options=azure_conn,
    manifest=ManifestOptions(
        credential_provider=azure_creds,
        s3_connection_options=azure_conn,
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
