# Playground

Try shardyfusion locally in under 60 seconds. No cloud accounts needed.

## Quick start

```bash
just playground
```

This single command:

1. Starts a local [Garage](https://garagehq.deuxfleurs.fr/) instance (S3-compatible object store)
2. Writes 10,000 sample user profiles across 4 shards
3. Reads them back, shows shard distribution, and launches the `shardy` interactive REPL

## Prerequisites

- Docker or Podman (`CONTAINER_ENGINE=docker` if using Docker)
- `just setup` (to install shardyfusion locally)

## Manual exploration

After the script runs, the Garage instance stays running. You can:

```bash
# Use the shardy CLI directly (credentials are created dynamically — check script output)
uv run shardy \
  --s3-prefix s3://playground/demo \
  --s3-option endpoint_url=http://localhost:3900 \
  --s3-option region=garage \
  --s3-option addressing_style=path \
  --s3-option access_key_id=<ACCESS_KEY> \
  --s3-option secret_access_key=<SECRET_KEY> \
  get 42
```

## Cleanup

```bash
just playground-down
```

This stops the Garage container and removes it.
