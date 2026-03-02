# shardyfusion

`shardyfusion` builds and reads sharded SlateDB snapshots.

The package provides:

- writer-side APIs for building `num_dbs` independent SlateDB shard databases from a Spark DataFrame
- manifest + `_CURRENT` publishing protocol (default S3, pluggable publisher/reader interfaces)
- reader-side routing and refresh helpers for service-side point lookups

Use this site for:

- setup and developer workflow
- writer/reader usage guides
- operational and release runbooks
- API reference generated from docstrings
