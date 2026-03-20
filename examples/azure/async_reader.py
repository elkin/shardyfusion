"""Async reader via an S3-compatible gateway on Azure.

Azure Blob Storage does not natively expose an S3-compatible API.
Deploy an S3-compatible server (e.g. Garage, SeaweedFS) on Azure.
See python_writer.py for setup details.

Prerequisites:
    pip install shardyfusion[read-async]
"""

import asyncio

from shardyfusion import AsyncShardedReader
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions

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


async def main() -> None:
    async with await AsyncShardedReader.open(
        s3_prefix="s3://my-bucket/snapshots/features",
        local_root="/tmp/shardyfusion-reader",
        credential_provider=azure_creds,
        s3_connection_options=azure_conn,
    ) as reader:
        value = await reader.get(42)
        print(f"key=42 -> {value}")

        batch = await reader.multi_get([1, 2, 3])
        for key, val in zip([1, 2, 3], batch):
            print(f"key={key} -> {val}")


if __name__ == "__main__":
    asyncio.run(main())
