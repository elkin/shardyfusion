# shardyfusion

`shardyfusion` builds and reads sharded SlateDB snapshots.

The package provides:

- writer-side APIs for building `num_dbs` independent SlateDB shard databases
  - **Spark** (`writer-spark`) — PySpark DataFrame-based, requires Java
  - **Dask** (`writer-dask`) — Dask DataFrame-based, no Java required
  - **Ray** (`writer-ray`) — Ray Data Dataset-based, no Java required
  - **Python** (`writer-python`) — pure-Python iterator-based, no Java required
- manifest + `_CURRENT` publishing protocol (default S3, pluggable publisher/reader interfaces)
- reader-side routing and refresh helpers for service-side point lookups

Use this site for:

- setup and developer workflow
- writer/reader usage guides
- operational and release runbooks
- API reference generated from docstrings
