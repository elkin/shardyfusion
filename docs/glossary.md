# Glossary

**Attempt**
: A single execution of a shard write. In Spark, task retries produce multiple attempts for the same `db_id`. The winner is selected deterministically (lowest attempt number, then `task_attempt_id`, then URL).

**Checkpoint ID**
: An opaque string returned by SlateDB after flushing a shard database. Stored in the manifest and used to open readers at the exact write position.

**CURRENT pointer**
: A JSON file at `{s3_prefix}/_CURRENT` that references the latest manifest. Updated atomically after a successful manifest publish. Contains `manifest_ref`, `run_id`, `updated_at`, and `format_version`.

**db_id**
: Zero-based integer identifying a shard database. Range: `[0, num_dbs)`. Assigned by the sharding strategy (hash, range, or custom).

**Key encoding**
: How routing keys are serialized to bytes for SlateDB storage. `u64be` = 8-byte big-endian unsigned int (default). `u32be` = 4-byte variant. Stored in the manifest so readers use the same encoding.

**Manifest**
: A JSON document describing a complete snapshot: which shards exist, where they are, how they were sharded, and metadata about the build. Published to `{s3_prefix}/manifests/run_id={run_id}/{manifest_name}`.

**num_dbs**
: Number of independent shard databases in a snapshot. Set in `WriteConfig`. All shards must be present for a valid manifest.

**Partition**
: In distributed frameworks (Spark, Dask, Ray), a unit of parallel execution. shardyfusion repartitions data so each partition maps to exactly one shard.

**Routing**
: The process of mapping a key to a `db_id`. Uses the sharding strategy (hash or range) and must produce identical results at write time and read time for correctness.

**run_id**
: A unique identifier for a single write execution. Auto-generated (UUID hex) if not provided. Used in S3 paths and manifest references to isolate runs.

**Shard**
: A single SlateDB database within a snapshot. Identified by `db_id`. Contains all key-value pairs that route to that `db_id`.

**Snapshot**
: The complete set of shards produced by a single `write_sharded` call, described by a manifest and pointed to by CURRENT. Readers load snapshots atomically.

**Two-phase publish**
: The write protocol where (1) the manifest is published first, then (2) the CURRENT pointer is updated. If step 2 fails, the manifest is orphaned but the previous snapshot remains valid.

**Winner**
: The selected attempt for each `db_id` when multiple attempts exist (e.g., Spark task retries). Selected deterministically to ensure consistent results across pipeline reruns.

**xxhash64**
: The hash function used for hash-based shard routing. Uses seed=42 (Spark's default) and an 8-byte little-endian key payload. `pmod(xxhash64(payload), num_dbs)` determines the `db_id`.
