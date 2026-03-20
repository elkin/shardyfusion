"""Write a sharded snapshot to AWS S3 using the Python writer.

Prerequisites:
    pip install shardyfusion[writer-python]

Authentication:
    Uses the default boto3 credential chain (env vars, ~/.aws/credentials,
    IAM instance profile, ECS task role, etc.).  No explicit credentials
    needed when running on EC2/ECS/Lambda with an attached IAM role.

    To use explicit credentials instead, see the StaticCredentialProvider
    example at the bottom of this file.
"""

from shardyfusion import WriteConfig
from shardyfusion.writer.python import write_sharded

# --- Sample data ---------------------------------------------------------

records = [{"id": i, "payload": f"value-{i}".encode()} for i in range(10_000)]

# --- Write to S3 ---------------------------------------------------------

config = WriteConfig(
    num_dbs=4,
    s3_prefix="s3://my-bucket/snapshots/features",
)

result = write_sharded(
    records,
    config,
    key_fn=lambda r: r["id"],
    value_fn=lambda r: r["payload"],
)

print(f"Snapshot written: {result.manifest_ref}")
print(f"Shards: {len(result.shards)}")

# --- With explicit credentials (optional) --------------------------------
#
# from shardyfusion.credentials import StaticCredentialProvider
#
# config = WriteConfig(
#     num_dbs=4,
#     s3_prefix="s3://my-bucket/snapshots/features",
#     credential_provider=StaticCredentialProvider(
#         access_key_id="AKIA...",
#         secret_access_key="wJal...",
#     ),
# )
