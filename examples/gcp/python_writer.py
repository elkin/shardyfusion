"""Write a sharded snapshot to Google Cloud Storage using the Python writer.

GCS provides an S3-compatible XML API.  To use it:
  1. Create HMAC keys for your service account:
     gsutil hmac create SERVICE_ACCOUNT_EMAIL
  2. Use the HMAC access key / secret as S3 credentials.
  3. Point the endpoint to https://storage.googleapis.com.

Prerequisites:
    pip install shardyfusion[writer-python]

See: https://cloud.google.com/storage/docs/interoperability
"""

from shardyfusion import WriteConfig
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.python import write_sharded

# --- Sample data ---------------------------------------------------------

records = [{"id": i, "payload": f"value-{i}".encode()} for i in range(10_000)]

# --- GCS connection via S3-compatible API --------------------------------

GCS_HMAC_ACCESS_KEY = "GOOG..."  # from: gsutil hmac create
GCS_HMAC_SECRET = "..."

config = WriteConfig(
    num_dbs=4,
    s3_prefix="s3://my-gcs-bucket/snapshots/features",
    credential_provider=StaticCredentialProvider(
        access_key_id=GCS_HMAC_ACCESS_KEY,
        secret_access_key=GCS_HMAC_SECRET,
    ),
    s3_connection_options=S3ConnectionOptions(
        endpoint_url="https://storage.googleapis.com",
        region_name="auto",
    ),
)

result = write_sharded(
    records,
    config,
    key_fn=lambda r: r["id"],
    value_fn=lambda r: r["payload"],
)

print(f"Snapshot written: {result.manifest_ref}")
print(f"Shards: {len(result.shards)}")
