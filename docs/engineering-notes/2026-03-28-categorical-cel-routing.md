# 2026-03-28 CEL Routing Simplification

- date: 2026-03-28
- commit before the change was introduced: `f7b100504383f1c78d9e1ea08f41a595db6c2b72`
- commit when the change was committed: `pending (not committed yet)`

## 1. What problem is being solved or functionality being added by the changes?

The existing CEL routing model originally supported two cases:

- direct mode, where the CEL expression returned the final dense integer shard id
- boundary mode, where the CEL expression returned an orderable value and `boundaries` converted it to a dense shard id with `bisect_right`

That covered hash-like and range-like routing, but it did not cover exact categorical routing well. A helper like `cel_sharding_by_columns("region", "tier", num_shards=None)` wants "one shard per distinct token" semantics, not interval semantics. With only `boundaries`, unseen values still fell into a neighboring interval, which is wrong for exact categories and a poor fit for shardyfusion's exact shard-ownership model.

This change introduces categorical CEL routing and then removes boundary routing entirely while keeping the internal storage model unchanged. CEL now supports only:

- direct mode, where the expression returns the dense shard id directly
- categorical mode, where the expression returns an exact token resolved through `routing_values`

## 2. What design decisions were considered with their pros and cons and trade offs?

### Option A: Keep only direct and boundary modes

Pros:

- no manifest format change
- minimal implementation complexity
- existing routing and reader behavior already covered these cases

Cons:

- exact categorical routing had to be approximated with interval semantics
- `cel_sharding_by_columns(..., num_shards=None)` could not naturally mean "one shard per distinct token"
- unseen categorical values were forced into some shard instead of becoming an exact-match miss

### Option B: Make shard ids themselves opaque, non-integer values

Pros:

- boundaries become unnecessary
- categorical routing tokens could become the public shard identity directly

Cons:

- large architectural change across manifest validation, shard path layout, worker orchestration, reader state, and many tests
- weakens the current dense-integer internal model that is efficient for storage layout and execution
- much larger backward-compatibility surface

### Option C: Keep dense internal shard ids, but generalize CEL to "token -> resolver -> db_id"

Pros:

- preserves the current internal dense integer `db_id` model
- adds exact categorical routing without disturbing hash routing or direct CEL
- keeps manifest routing metadata explicit and reconstructable by readers
- lets helpers infer categorical routing tables from data while leaving internal storage unchanged

Cons:

- requires a manifest schema extension
- adds a new resolver mode to validate and test
- requires defining unknown-token behavior for categorical snapshots

### Option D: Keep direct CEL and add categorical CEL, then remove boundary routing

Pros:

- matches shardyfusion's exact routing semantics better than interval routing
- avoids silently routing unseen categorical contexts into an existing shard
- keeps the public CEL model smaller and easier to explain
- true range bucketing remains possible by returning explicit integer shard ids

Cons:

- breaks unreleased boundary-based configs and manifests
- requires test and doc rewrites across all writer backends and readers

The chosen approach was Option D built on top of Option C's internal `token -> db_id` model.

## 3. What implementation was chosen and why?

The implementation adds `routing_values` as CEL manifest metadata and treats CEL routing as two resolver modes:

- direct integer mode: no `routing_values`
- categorical mode: `routing_values` present

Key design points:

- Internal shard identity remains a dense integer `db_id`. Categorical routing tokens are resolved to `db_id` by exact match against `routing_values`, using the token's position as the dense shard id.
- `cel_sharding()` now accepts explicit `routing_values`.
- `cel_sharding_by_columns(..., num_shards=None)` now produces an inferred categorical CEL spec. Writers evaluate tokens from the input, build sorted distinct `routing_values`, and publish the resolved categorical metadata in the manifest.
- Boundary metadata is removed from `ShardingSpec`, `ManifestShardingSpec`, writer paths, and reader/router logic. Manifests that still contain `boundaries` are now rejected.
- The manifest format version is bumped to `3` for snapshots that use `routing_values`. Older format-2 manifests remain valid as long as they do not rely on removed boundary routing.
- Readers reconstruct categorical routing from manifest metadata. For categorical snapshots, unknown tokens are treated as misses by `get()` and `multi_get()`, while routing/introspection APIs still raise because there is no matching shard.
- Spark, Dask, Ray, and Python writers all produce resolved categorical metadata before publish. Python parallel mode still rejects inferred categorical routing because worker counts must be known up front.
- Range bucketing is still available, but only when encoded directly in the CEL expression so the expression itself returns the dense shard id.

This keeps the storage and execution model stable, makes exact categorical routing a first-class concept, and removes interval semantics that were a poor fit for exact shard ownership.
