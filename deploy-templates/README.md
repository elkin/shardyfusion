# Deploy Templates

Production-ready starting points for deploying shardyfusion. Fork, customize, and ship.

| Template | Use case | Reader type |
|---|---|---|
| [FastAPI Service](fastapi-service/) | Async lookup service with health checks, metrics, background refresh | `AsyncShardedReader` |
| [AWS Lambda](aws-lambda/) | Serverless lookups with container reuse, SAM deployment | `ShardedReader` |
| [Airflow DAG](airflow-dag/) | Scheduled snapshot builds with post-write verification | Python writer + `ShardedReader` |

## Getting started

Each template is self-contained. Pick the one that matches your deployment model, copy it into your project, and customize:

1. **FastAPI** — Best for always-on services. Async reader, Prometheus metrics, Kubernetes-ready.
2. **Lambda** — Best for infrequent or bursty lookups. Zero infrastructure when idle.
3. **Airflow** — Best for scheduled data pipeline builds. Pairs with either serving template above.

All templates use the default boto3 credential chain for S3 access. No hardcoded credentials.
