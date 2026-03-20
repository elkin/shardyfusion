"""Airflow DAG — scheduled shardyfusion snapshot build.

Builds a sharded snapshot on a daily schedule using the Python writer,
then verifies the result by reading it back.

Configure via Airflow Variables:
    shardyfusion_s3_prefix    s3://bucket/prefix (required)
    shardyfusion_num_shards   Number of shards (default: 8)

Credentials: uses the default boto3 chain (IAM instance profile, env vars, etc.).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# ── DAG defaults ─────────────────────────────────────────────────────────────

default_args = {
    "owner": "data-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ── Tasks ────────────────────────────────────────────────────────────────────


def build_snapshot(**context: Any) -> str:
    """Write a sharded snapshot to S3."""
    from airflow.models import Variable

    from shardyfusion import WriteConfig
    from shardyfusion.writer.python import write_sharded

    s3_prefix = Variable.get("shardyfusion_s3_prefix")
    num_shards = int(Variable.get("shardyfusion_num_shards", default_var="8"))

    # --- Replace this with your actual data source -----------------------
    records = fetch_records()
    # ---------------------------------------------------------------------

    config = WriteConfig(
        num_dbs=num_shards,
        s3_prefix=s3_prefix,
    )

    result = write_sharded(
        records,
        config,
        key_fn=lambda r: r["id"],
        value_fn=lambda r: json.dumps(r, separators=(",", ":")).encode(),
    )

    logger.info(
        "Snapshot written: run_id=%s, shards=%d, rows=%d",
        result.run_id,
        len(result.shards),
        sum(s.row_count for s in result.shards),
    )

    # Pass manifest ref to downstream tasks
    context["ti"].xcom_push(key="manifest_ref", value=result.manifest_ref)
    context["ti"].xcom_push(key="run_id", value=result.run_id)
    return result.manifest_ref


def verify_snapshot(**context: Any) -> None:
    """Read the snapshot back and validate basic properties."""
    from airflow.models import Variable

    from shardyfusion import ShardedReader

    s3_prefix = Variable.get("shardyfusion_s3_prefix")
    expected_shards = int(Variable.get("shardyfusion_num_shards", default_var="8"))
    run_id = context["ti"].xcom_pull(task_ids="build_snapshot", key="run_id")

    with ShardedReader(
        s3_prefix=s3_prefix,
        local_root="/tmp/shardyfusion-verify",
    ) as reader:
        info = reader.snapshot_info()

        assert info.run_id == run_id, (
            f"Run ID mismatch: expected {run_id}, got {info.run_id}"
        )
        assert info.num_dbs == expected_shards, (
            f"Shard count mismatch: expected {expected_shards}, got {info.num_dbs}"
        )
        assert info.row_count > 0, "Snapshot is empty"

        # Spot-check a few keys
        sample_value = reader.get(0)
        assert sample_value is not None, "Key 0 not found in snapshot"

        logger.info(
            "Verification passed: run_id=%s, shards=%d, rows=%d",
            info.run_id,
            info.num_dbs,
            info.row_count,
        )


def fetch_records() -> list[dict[str, Any]]:
    """Placeholder — replace with your actual data source.

    Examples:
        - Query a database: `cursor.execute("SELECT * FROM features")`
        - Read a file: `pd.read_parquet("s3://data-lake/features/").to_dict("records")`
        - Call an API: `requests.get("https://api.example.com/features").json()`
    """
    return [
        {"id": i, "feature_a": i * 0.1, "feature_b": f"cat-{i % 10}"}
        for i in range(10_000)
    ]


# ── DAG definition ──────────────────────────────────────────────────────────

with DAG(
    dag_id="shardyfusion_snapshot_build",
    default_args=default_args,
    description="Build and verify a shardyfusion snapshot",
    schedule="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["shardyfusion", "snapshot"],
) as dag:
    build = PythonOperator(
        task_id="build_snapshot",
        python_callable=build_snapshot,
    )

    verify = PythonOperator(
        task_id="verify_snapshot",
        python_callable=verify_snapshot,
    )

    build >> verify
