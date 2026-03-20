# Playground

Try shardyfusion locally in under 60 seconds. No cloud accounts needed.

## Quick start

```bash
just playground
```

This single command:

1. Starts a local MinIO instance (S3-compatible object store)
2. Writes 10,000 sample user profiles across 4 shards
3. Reads them back, shows shard distribution, and launches the `shardy` interactive REPL

## Prerequisites

- Docker or Podman (`CONTAINER_ENGINE=docker` if using Docker)
- `just setup` (to install shardyfusion locally)

## Manual exploration

After the script runs, the MinIO instance stays running. You can:

```bash
# Use the shardy CLI directly
uv run shardy \
  --s3-prefix s3://playground/demo \
  --s3-option endpoint_url=http://localhost:9000 \
  --s3-option region=us-east-1 \
  --s3-option addressing_style=path \
  --s3-option access_key_id=minioadmin \
  --s3-option secret_access_key=minioadmin \
  get 42

# Browse the MinIO console
open http://localhost:9001  # login: minioadmin / minioadmin
```

## Cleanup

```bash
just playground-down
```

This stops the MinIO container and removes it. The `shardyfusion-playground-data` volume persists across runs — remove it manually with `docker volume rm shardyfusion-playground-data` if needed.
