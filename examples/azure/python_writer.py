"""Write a sharded snapshot using MinIO Gateway on Azure.

Azure Blob Storage does not natively expose an S3-compatible API.
The recommended approach is to deploy MinIO as an S3 gateway in front
of Azure Blob Storage:

    # Deploy MinIO gateway (Docker example)
    docker run -p 9000:9000 \
      -e MINIO_ROOT_USER=<azure-storage-account> \
      -e MINIO_ROOT_PASSWORD=<azure-storage-key> \
      minio/minio gateway azure

    # Or use Azure Container Instances / AKS for production.

Once the gateway is running, shardyfusion connects to it like any
S3-compatible endpoint.

Prerequisites:
    pip install shardyfusion[writer-python]

See: https://min.io/docs/minio/linux/integrations/azure.html
"""

from shardyfusion import WriteConfig
from shardyfusion.config import ManifestOptions
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.python import write_sharded

# --- Sample data ---------------------------------------------------------

records = [{"id": i, "payload": f"value-{i}".encode()} for i in range(10_000)]

# --- MinIO gateway connection --------------------------------------------

MINIO_ENDPOINT = "http://localhost:9000"  # or your Azure-hosted MinIO URL
AZURE_STORAGE_ACCOUNT = "myaccount"
AZURE_STORAGE_KEY = "..."

azure_creds = StaticCredentialProvider(
    access_key_id=AZURE_STORAGE_ACCOUNT,
    secret_access_key=AZURE_STORAGE_KEY,
)
azure_conn = S3ConnectionOptions(
    endpoint_url=MINIO_ENDPOINT,
    region_name="us-east-1",  # MinIO default
    addressing_style="path",  # required for MinIO
)

# --- Write ---------------------------------------------------------------

config = WriteConfig(
    num_dbs=4,
    s3_prefix="s3://my-container/snapshots/features",
    credential_provider=azure_creds,
    s3_connection_options=azure_conn,
    manifest=ManifestOptions(
        credential_provider=azure_creds,
        s3_connection_options=azure_conn,
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
