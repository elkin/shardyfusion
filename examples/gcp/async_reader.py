"""Async reader for Google Cloud Storage.

GCS provides an S3-compatible XML API accessed via HMAC keys.

Prerequisites:
    pip install shardyfusion[read-async]

See: https://cloud.google.com/storage/docs/interoperability
"""

import asyncio

from shardyfusion import AsyncShardedReader
from shardyfusion.credentials import StaticCredentialProvider
from shardyfusion.type_defs import S3ConnectionOptions

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


async def main() -> None:
    async with await AsyncShardedReader.open(
        s3_prefix="s3://my-gcs-bucket/snapshots/features",
        local_root="/tmp/shardyfusion-reader",
        credential_provider=gcs_creds,
        s3_connection_options=gcs_conn,
    ) as reader:
        value = await reader.get(42)
        print(f"key=42 → {value}")

        batch = await reader.multi_get([1, 2, 3])
        for key, val in zip([1, 2, 3], batch):
            print(f"key={key} → {val}")


if __name__ == "__main__":
    asyncio.run(main())
