# AWS Lambda Template

Serverless key-value lookups from a shardyfusion snapshot on S3.

## How it works

- The `ShardedReader` initializes once on Lambda cold start
- Subsequent invocations reuse the reader (Lambda container reuse)
- Credentials come from the Lambda execution role — no keys to manage
- `/tmp/shardyfusion` is used as the local cache (Lambda's writable temp dir)

## Deploy with SAM

```bash
# Build
sam build

# Deploy (guided — prompts for S3Prefix and S3BucketArn)
sam deploy --guided

# Or with parameters
sam deploy \
  --stack-name shardyfusion-lookup \
  --parameter-overrides \
    S3Prefix=s3://my-bucket/snapshots/features \
    S3BucketArn=arn:aws:s3:::my-bucket \
  --capabilities CAPABILITY_IAM
```

## Test locally

```bash
# Invoke with a get event
sam local invoke LookupFunction \
  -e - <<< '{"action": "get", "key": "42"}'

# Invoke with a multi_get event
sam local invoke LookupFunction \
  -e - <<< '{"action": "multi_get", "keys": ["1", "2", "3"]}'

# Invoke info
sam local invoke LookupFunction \
  -e - <<< '{"action": "info"}'
```

## Event format

```json
{"action": "get", "key": "42"}
{"action": "multi_get", "keys": ["1", "2", "3"]}
{"action": "info"}
```

## Performance tuning

- **Memory**: 512MB default. Increase for large snapshots (more shards = more open file handles).
- **Provisioned concurrency**: Eliminates cold starts. Set to your expected concurrent invocation count.
- **Timeout**: 30s default. Cold start (reader init) takes 2-10s depending on shard count and S3 latency. Warm invocations are sub-millisecond for routing + S3 GET latency.
- **`/tmp` size**: Lambda provides 512MB by default (configurable up to 10GB). Each shard's local cache needs a few MB.
