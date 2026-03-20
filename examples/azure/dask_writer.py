"""Write a sharded snapshot using MinIO Gateway on Azure with Dask.

Azure Blob Storage does not natively expose an S3-compatible API.
Deploy MinIO as an S3 gateway in front of Azure Blob Storage
(see python_writer.py for gateway setup instructions).

Prerequisites:
    pip install shardyfusion[writer-dask]

See: https://min.io/docs/minio/linux/integrations/azure.html
"""

import dask.dataframe as dd
import pandas as pd

from shardyfusion import WriteConfig
from shardyfusion.config import ManifestOptions
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions
from shardyfusion.writer.dask import write_sharded

# --- MinIO gateway connection --------------------------------------------

MINIO_ENDPOINT = "http://localhost:9000"
AZURE_STORAGE_ACCOUNT = "myaccount"
AZURE_STORAGE_KEY = "..."

azure_creds = StaticCredentialProvider(
    access_key_id=AZURE_STORAGE_ACCOUNT,
    secret_access_key=AZURE_STORAGE_KEY,
)
azure_conn = S3ConnectionOptions(
    endpoint_url=MINIO_ENDPOINT,
    region_name="us-east-1",
    addressing_style="path",
)

# --- Sample data ---------------------------------------------------------

pdf = pd.DataFrame({"id": range(50_000), "payload": [f"value-{i}" for i in range(50_000)]})
ddf = dd.from_pandas(pdf, npartitions=8)

# --- Write ---------------------------------------------------------------

config = WriteConfig(
    num_dbs=8,
    s3_prefix="s3://my-container/snapshots/features",
    credential_provider=azure_creds,
    s3_connection_options=azure_conn,
    manifest=ManifestOptions(
        credential_provider=azure_creds,
        s3_connection_options=azure_conn,
    ),
)

result = write_sharded(
    ddf,
    config,
    key_col="id",
    value_spec={"binary_col": "payload"},
)

print(f"Snapshot written: {result.manifest_ref}")
