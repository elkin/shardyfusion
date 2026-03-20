# Examples

Cloud provider examples for shardyfusion — write and read sharded snapshots on AWS, GCP, and Azure.

## AWS S3

Native S3 support. Uses the default boto3 credential chain (env vars, `~/.aws/credentials`, IAM instance profiles, ECS task roles).

| Example | Description |
|---|---|
| [aws/python_writer.py](aws/python_writer.py) | Python writer with default credential chain |
| [aws/spark_writer.py](aws/spark_writer.py) | Spark writer (requires Java 17) |
| [aws/dask_writer.py](aws/dask_writer.py) | Dask writer |
| [aws/ray_writer.py](aws/ray_writer.py) | Ray writer |
| [aws/reader.py](aws/reader.py) | Single-threaded reader |
| [aws/concurrent_reader.py](aws/concurrent_reader.py) | Thread-safe reader (lock and pool modes) |
| [aws/async_reader.py](aws/async_reader.py) | Async reader for asyncio services |

## Google Cloud Storage

GCS exposes an [S3-compatible XML API](https://cloud.google.com/storage/docs/interoperability). Create HMAC keys for your service account and point the endpoint to `https://storage.googleapis.com`.

```bash
# Create HMAC keys for your service account
gsutil hmac create SERVICE_ACCOUNT_EMAIL
```

| Example | Description |
|---|---|
| [gcp/python_writer.py](gcp/python_writer.py) | Python writer with HMAC keys |
| [gcp/spark_writer.py](gcp/spark_writer.py) | Spark writer with HMAC keys |
| [gcp/dask_writer.py](gcp/dask_writer.py) | Dask writer with HMAC keys |
| [gcp/ray_writer.py](gcp/ray_writer.py) | Ray writer with HMAC keys |
| [gcp/reader.py](gcp/reader.py) | Single-threaded reader |
| [gcp/async_reader.py](gcp/async_reader.py) | Async reader |

## Azure Blob Storage

Azure Blob Storage does not natively expose an S3-compatible API. These examples use an S3-compatible server (e.g. [Garage](https://garagehq.deuxfleurs.fr/), [SeaweedFS](https://github.com/seaweedfs/seaweedfs)) deployed alongside your Azure infrastructure.

See [Garage quick start](https://garagehq.deuxfleurs.fr/documentation/quick-start/) for deployment on Azure Container Instances or AKS.

| Example | Description |
|---|---|
| [azure/python_writer.py](azure/python_writer.py) | Python writer via S3 gateway |
| [azure/spark_writer.py](azure/spark_writer.py) | Spark writer via S3 gateway |
| [azure/dask_writer.py](azure/dask_writer.py) | Dask writer via S3 gateway |
| [azure/ray_writer.py](azure/ray_writer.py) | Ray writer via S3 gateway |
| [azure/reader.py](azure/reader.py) | Single-threaded reader via S3 gateway |
| [azure/async_reader.py](azure/async_reader.py) | Async reader via S3 gateway |

## Installation

```bash
# Pick the extras you need
pip install shardyfusion[writer-python]     # Python writer
pip install shardyfusion[writer-spark]      # Spark writer (requires Java 17)
pip install shardyfusion[writer-dask]       # Dask writer
pip install shardyfusion[writer-ray]        # Ray writer
pip install shardyfusion[read]              # Sync readers
pip install shardyfusion[read-async]        # Async reader (aiobotocore)
```
