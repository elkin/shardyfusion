"""Async reader for asyncio services on AWS S3.

Use AsyncShardedReader with FastAPI, aiohttp, or any asyncio-based service.

Prerequisites:
    pip install shardyfusion[read-async]

Authentication:
    Uses the default boto3 credential chain.  The async reader uses
    aiobotocore under the hood for native async S3 access.
"""

import asyncio

from shardyfusion import AsyncShardedReader


async def main() -> None:
    async with await AsyncShardedReader.open(
        s3_prefix="s3://my-bucket/snapshots/features",
        local_root="/tmp/shardyfusion-reader",
    ) as reader:
        value = await reader.get(42)
        print(f"key=42 → {value}")

        batch = await reader.multi_get([1, 2, 3, 100])
        for key, val in zip([1, 2, 3, 100], batch):
            print(f"key={key} → {val}")

        # Atomic refresh
        await reader.refresh()

        # Inspect snapshot metadata (sync — no I/O)
        info = reader.snapshot_info()
        print(f"Run ID: {info.run_id}, Shards: {info.num_shards}")


if __name__ == "__main__":
    asyncio.run(main())
