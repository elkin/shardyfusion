# FastAPI Service Template

A production-ready async lookup service for shardyfusion snapshots.

## Features

- Single-key and batch lookups via REST API
- Background refresh (configurable interval)
- Health check with staleness detection
- Prometheus metrics endpoint
- Rate limiting (reads/sec)

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `S3_PREFIX` | (required) | S3 prefix of the snapshot, e.g. `s3://bucket/prefix` |
| `LOCAL_ROOT` | `/tmp/shardyfusion` | Local cache directory for SlateDB |
| `MAX_READS_PER_SEC` | `0` (unlimited) | Rate limit for read operations |
| `REFRESH_INTERVAL_S` | `300` | Background refresh interval (0 = disable) |
| `STALENESS_THRESHOLD_S` | `3600` | Health check degrades after this many seconds |

AWS credentials are resolved via the default boto3 chain (env vars, instance profile, ECS task role, etc.).

## Run locally

Against the playground (start it first with `just playground`).
The playground creates dynamic credentials — check the script output for the access key and secret key.

```bash
S3_PREFIX=s3://playground/demo \
  SLATEDB_S3_ENDPOINT_URL=http://localhost:3900 \
  AWS_ACCESS_KEY_ID=<from playground output> \
  AWS_SECRET_ACCESS_KEY=<from playground output> \
  AWS_REGION=garage \
  uvicorn app:app --reload
```

## API

```bash
# Single lookup
curl http://localhost:8000/api/v1/get/42

# Batch lookup
curl -X POST http://localhost:8000/api/v1/multi-get \
  -H "Content-Type: application/json" \
  -d '{"keys": [1, 2, 3, 100]}'

# Health check
curl http://localhost:8000/health

# Snapshot info
curl http://localhost:8000/info

# Prometheus metrics
curl http://localhost:8000/metrics

# Manual refresh
curl -X POST http://localhost:8000/admin/refresh
```

## Deploy with Docker

```bash
docker build -t shardyfusion-service .
docker run -p 8000:8000 \
  -e S3_PREFIX=s3://my-bucket/snapshots/features \
  shardyfusion-service
```

## Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shardyfusion-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: shardyfusion-service
  template:
    metadata:
      labels:
        app: shardyfusion-service
    spec:
      containers:
        - name: app
          image: shardyfusion-service:latest
          ports:
            - containerPort: 8000
          env:
            - name: S3_PREFIX
              value: s3://my-bucket/snapshots/features
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: shardyfusion-service
spec:
  selector:
    app: shardyfusion-service
  ports:
    - port: 80
      targetPort: 8000
```
