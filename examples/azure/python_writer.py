"""Write a sharded snapshot via an S3-compatible gateway on Azure.

Azure Blob Storage does not natively expose an S3-compatible API.
Deploy an S3-compatible server (e.g. Garage, SeaweedFS) on Azure
Container Instances or AKS:

    # Example: Garage on Azure Container Instances
    # See https://garagehq.deuxfleurs.fr/documentation/quick-start/

Once the gateway is running, shardyfusion connects to it like any
S3-compatible endpoint.

Prerequisites:
    pip install shardyfusion[writer-python]
"""

from shardyfusion import WriteConfig
from shardyfusion.config import ManifestOptions
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.python import write_sharded

# --- Sample data ---------------------------------------------------------

records = [{"id": i, "payload": f"value-{i}".encode()} for i in range(10_000)]

# --- S3-compatible gateway connection ------------------------------------

GATEWAY_ENDPOINT = "http://localhost:3900"  # your Azure-hosted S3 gateway
ACCESS_KEY = "..."  # gateway access key
SECRET_KEY = "..."  # gateway secret key

azure_creds = StaticCredentialProvider(
    access_key_id=ACCESS_KEY,
    secret_access_key=SECRET_KEY,
)
azure_conn = S3ConnectionOptions(
    endpoint_url=GATEWAY_ENDPOINT,
    region_name="garage",
    addressing_style="path",  # required for most S3-compatible servers
)

# --- Write ---------------------------------------------------------------

config = WriteConfig(
    num_dbs=4,
    s3_prefix="s3://my-bucket/snapshots/features",
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
