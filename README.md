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

## Phase 2: Service-side reads

### Default mode (S3 publisher + default reader)

```python
from slatedb_spark_sharded import SlateShardedReader

reader = SlateShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/slatedb-reader",
)

value = reader.get(123)
batch = reader.multi_get([1, 2, 3])
reader.refresh()
reader.close()
```

### Custom mode (custom publisher requires custom manifest reader)

If you provide a custom publisher, you must also provide a custom `ManifestReader`.

```python
from slatedb_spark_sharded import FunctionManifestReader, SlateShardedReader

manifest_reader = FunctionManifestReader(
    load_current_fn=my_load_current,
    load_manifest_fn=my_load_manifest,
)

reader = SlateShardedReader(
    s3_prefix="s3://bucket/prefix",
    local_root="/tmp/slatedb-reader",
    publisher=my_custom_publisher,
    manifest_reader=manifest_reader,
)
```

## Integration test matrix (PySpark 3.5 and 4.x)

```bash
python3 -m pip install tox
tox -e spark35,spark4
```
