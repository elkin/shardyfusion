# slatedb_spark_sharded

`slatedb_spark_sharded` is a PySpark sharded snapshot writer for building multiple independent SlateDB databases from one DataFrame, then publishing manifest metadata and a `_CURRENT` pointer.

## Install

```bash
pip install -e .
```

## Minimal usage

```python
from slatedb_spark_sharded import SlateDbConfig, ValueSpec, write_sharded_slatedb

config = SlateDbConfig(
    num_dbs=8,
    s3_prefix="s3://bucket/prefix",
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
)

result = write_sharded_slatedb(df, config)
```

## Custom manifest builder

```python
from dataclasses import asdict
import json

from slatedb_spark_sharded import JsonManifestBuilder, ManifestArtifact


class TeamJsonBuilder(JsonManifestBuilder):
    def build(self, *, required_build, shards, custom_fields):
        artifact = super().build(
            required_build=required_build,
            shards=shards,
            custom_fields={**custom_fields, "team": "analytics"},
        )
        return ManifestArtifact(
            payload=artifact.payload,
            content_type=artifact.content_type,
            headers={"x-manifest-flavor": "team-json"},
        )
```

## Custom publisher

```python
from slatedb_spark_sharded import ManifestPublisher, ManifestArtifact


class InMemoryPublisher(ManifestPublisher):
    def __init__(self):
        self.objects = {}

    def publish_manifest(self, *, name: str, artifact: ManifestArtifact, run_id: str) -> str:
        ref = f"mem://manifests/run_id={run_id}/{name}"
        self.objects[ref] = artifact
        return ref

    def publish_current(self, *, name: str, artifact: ManifestArtifact) -> str | None:
        ref = f"mem://{name}"
        self.objects[ref] = artifact
        return ref
```

Pass custom components through config:

```python
config = SlateDbConfig(
    num_dbs=8,
    s3_prefix="s3://bucket/prefix",
    key_col="id",
    value_spec=ValueSpec.binary_col("payload"),
    manifest_builder=TeamJsonBuilder(),
    publisher=InMemoryPublisher(),
)
```

## Notes

- Exactly `num_dbs` writer partitions are enforced via Spark partitioning by `db_id`.
- Attempt-isolated output URLs are used for retry/speculation safety.
- Winner selection is deterministic per `db_id`.
- `_CURRENT` pointer is always JSON (`application/json`) even if your manifest format is custom.
- Recommended Spark setting: `spark.speculation=false`.

## Integration test matrix (PySpark 3.5 and 4.x)

```bash
python3 -m pip install tox
tox -e spark35,spark4
```
