"""Async reader via MinIO Gateway on Azure.

Azure Blob Storage does not natively expose an S3-compatible API.
Deploy MinIO as an S3 gateway in front of Azure Blob Storage
(see python_writer.py for gateway setup instructions).

Prerequisites:
    pip install shardyfusion[read-async]

See: https://min.io/docs/minio/linux/integrations/azure.html
"""

import asyncio

from shardyfusion import AsyncShardedReader
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions

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


async def main() -> None:
    async with await AsyncShardedReader.open(
        s3_prefix="s3://my-container/snapshots/features",
        local_root="/tmp/shardyfusion-reader",
        credential_provider=azure_creds,
        s3_connection_options=azure_conn,
    ) as reader:
        value = await reader.get(42)
        print(f"key=42 → {value}")

        batch = await reader.multi_get([1, 2, 3])
        for key, val in zip([1, 2, 3], batch):
            print(f"key={key} → {val}")


if __name__ == "__main__":
    asyncio.run(main())
