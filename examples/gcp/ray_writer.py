"""Write a sharded snapshot to Google Cloud Storage using the Ray writer.

GCS provides an S3-compatible XML API accessed via HMAC keys.

Prerequisites:
    pip install shardyfusion[writer-ray]

See: https://cloud.google.com/storage/docs/interoperability
"""

import ray
import ray.data

from shardyfusion import WriteConfig
from shardyfusion.config import ManifestOptions
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.ray import write_sharded

ray.init()

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

ds = ray.data.from_items(
    [{"id": i, "payload": f"value-{i}"} for i in range(50_000)]
)

# --- Write to GCS -------------------------------------------------------

config = WriteConfig(
    num_dbs=8,
    s3_prefix="s3://my-gcs-bucket/snapshots/features",
    credential_provider=gcs_creds,
    s3_connection_options=gcs_conn,
    manifest=ManifestOptions(
        credential_provider=gcs_creds,
        s3_connection_options=gcs_conn,
    ),
)

result = write_sharded(
    ds,
    config,
    key_col="id",
    value_spec={"binary_col": "payload"},
)

print(f"Snapshot written: {result.manifest_ref}")

ray.shutdown()
