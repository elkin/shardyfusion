"""Opaque per-shard checkpoint identifier.

shardyfusion 0.x backed every published shard with a backend-specific
checkpoint id: SlateDB returned a UUID from ``Db::create_checkpoint``,
the SQLite adapters returned a SHA-256 of the materialized DB file,
and LanceDB returned a manifest hash. The reader pinned itself to that
value via ``DbReaderBuilder::with_checkpoint_id`` (SlateDB) or used it
as a content fingerprint (SQLite) for cache validation.

slatedb 0.12 removed checkpoint creation from the public uniffi API.
We could no longer obtain a SlateDB-managed checkpoint UUID at write
time, so the contract changed: every backend now stamps shards with an
opaque shardyfusion-generated UUID. The id is durable enough to act as
a stable manifest reference but is no longer interpretable by any
backend.

This works because shardyfusion enforces a single-writer-per-shard
invariant and only publishes a manifest after every shard writer has
flushed and closed successfully. With S3 strong read-after-write
consistency, a published shard URL is immediately readable, so reader
pinning by checkpoint id is unnecessary.

The helper lives in its own module to keep the call sites uniform —
every framework writer used to call ``adapter.checkpoint()`` in the
same shape, and now they all call :func:`generate_checkpoint_id`
instead.
"""

from __future__ import annotations

import uuid

__all__ = ["generate_checkpoint_id"]


def generate_checkpoint_id() -> str:
    """Return a fresh opaque shard checkpoint id.

    The format is the 32-character lowercase hex form of a random
    UUIDv4 (e.g. ``"5b2f1c0a4d3a4e6f9b8c7a6d5e4f3210"``). The value is
    treated as opaque by readers; do not parse it.
    """
    return uuid.uuid4().hex
