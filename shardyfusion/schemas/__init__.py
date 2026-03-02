"""JSON Schema resources for shardyfusion manifest and CURRENT pointer formats.

Schemas are generated from Pydantic models by ``scripts/generate_schemas.py``
and shipped as package data so downstream consumers can validate payloads::

    from shardyfusion.schemas import load_manifest_schema

    schema = load_manifest_schema()
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def load_manifest_schema() -> dict[str, Any]:
    """Return the parsed JSON Schema for the sharded manifest format."""
    return json.loads(
        files(__name__).joinpath("manifest.schema.json").read_text("utf-8")
    )


def load_current_pointer_schema() -> dict[str, Any]:
    """Return the parsed JSON Schema for the ``_CURRENT`` pointer format."""
    return json.loads(
        files(__name__).joinpath("current-pointer.schema.json").read_text("utf-8")
    )
