# Airflow DAG Template

Scheduled shardyfusion snapshot build with post-write verification.

## What it does

1. **build_snapshot** — Writes a sharded snapshot to S3 using the Python writer
2. **verify_snapshot** — Reads the snapshot back, validates shard count and spot-checks keys

## Setup

1. Copy `snapshot_build_dag.py` to your Airflow `dags/` directory
2. Install shardyfusion in your Airflow environment:
   ```bash
   pip install shardyfusion[writer-python,read]
   ```
3. Set the required Airflow Variables:

| Variable | Required | Description |
|---|---|---|
| `shardyfusion_s3_prefix` | Yes | S3 prefix, e.g. `s3://my-bucket/snapshots/features` |
| `shardyfusion_num_shards` | No | Number of shards (default: 8) |

4. Ensure the Airflow worker has S3 access (IAM role, env vars, or Airflow Connection)

## Customize the data source

Replace the `fetch_records()` function with your actual data pipeline:

```python
def fetch_records():
    # Database query
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, feature_a, feature_b FROM features"))
        return [dict(row) for row in rows]

    # Or read from a data lake
    import pandas as pd
    return pd.read_parquet("s3://data-lake/features/").to_dict("records")
```

## Manual trigger

```bash
# Via Airflow CLI
airflow dags trigger shardyfusion_snapshot_build

# Via Airflow REST API
curl -X POST http://airflow:8080/api/v1/dags/shardyfusion_snapshot_build/dagRuns \
  -H "Content-Type: application/json" \
  -d '{}'
```
