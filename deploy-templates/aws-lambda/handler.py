"""AWS Lambda handler for serving lookups from a shardyfusion snapshot.

The reader is initialized once on cold start and reused across invocations
(Lambda container reuse).  Uses the sync ShardedReader since Lambda
invocations are single-threaded.

Configure via environment variables:
    S3_PREFIX    s3://bucket/prefix (required)

Credentials come from the Lambda execution role (default boto3 chain).

Event format:
    {"action": "get", "key": "42"}
    {"action": "multi_get", "keys": ["1", "2", "3"]}
    {"action": "info"}
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Module-level reader survives across Lambda invocations (container reuse)
_reader = None


def _get_reader():
    """Lazy-init the reader on first invocation."""
    global _reader
    if _reader is not None:
        return _reader

    from shardyfusion import ShardedReader

    s3_prefix = os.environ["S3_PREFIX"]

    _reader = ShardedReader(
        s3_prefix=s3_prefix,
        local_root="/tmp/shardyfusion",
    )
    logger.info("Reader initialized: %s", s3_prefix)
    return _reader


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ARG001
    """Route to get, multi_get, or info based on event action."""
    action = event.get("action", "get")

    try:
        reader = _get_reader()

        if action == "get":
            key = event["key"]
            lookup_key: str | int = int(key) if str(key).lstrip("-").isdigit() else key
            value = reader.get(lookup_key)
            encoded = base64.b64encode(value).decode("ascii") if value is not None else None
            return {"statusCode": 200, "body": json.dumps({"key": key, "value": encoded})}

        if action == "multi_get":
            keys = event["keys"]
            lookup_keys = [int(k) if str(k).lstrip("-").isdigit() else k for k in keys]
            results = reader.multi_get(lookup_keys)
            encoded = {
                str(k): base64.b64encode(v).decode("ascii") if v is not None else None
                for k, v in results.items()
            }
            return {"statusCode": 200, "body": json.dumps({"results": encoded})}

        if action == "info":
            info = reader.snapshot_info()
            return {
                "statusCode": 200,
                "body": json.dumps(
                    {
                        "run_id": info.run_id,
                        "num_shards": info.num_dbs,
                        "row_count": info.row_count,
                        "key_encoding": str(info.key_encoding),
                    }
                ),
            }

        return {"statusCode": 400, "body": json.dumps({"error": f"Unknown action: {action}"})}

    except KeyError as exc:
        return {"statusCode": 400, "body": json.dumps({"error": f"Missing field: {exc}"})}
    except Exception as exc:
        logger.exception("Handler error")
        retryable = getattr(exc, "retryable", False)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(exc), "retryable": retryable}),
        }
