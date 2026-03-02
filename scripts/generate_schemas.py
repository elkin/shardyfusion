"""Generate JSON Schema files from Pydantic models.

Usage:
    python scripts/generate_schemas.py          # write to schemas/
    python scripts/generate_schemas.py --check  # verify schemas/ is up-to-date
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from shardyfusion.manifest import CurrentPointer, ParsedManifest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMAS_DIR = REPO_ROOT / "schemas"

SCHEMAS: dict[str, dict] = {
    "manifest.schema.json": {
        **ParsedManifest.model_json_schema(mode="serialization"),
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/slatedb/shardyfusion/schemas/manifest.schema.json",
        "title": "SlateDB Sharded Manifest",
        "description": "JSON manifest published to S3 by the sharded writer.",
    },
    "current-pointer.schema.json": {
        **CurrentPointer.model_json_schema(),
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/slatedb/shardyfusion/schemas/current-pointer.schema.json",
        "title": "SlateDB Sharded CURRENT Pointer",
        "description": "JSON pointer published to S3 at _CURRENT.",
    },
}


def _serialize(schema: dict) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def generate(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, schema in SCHEMAS.items():
        (target_dir / filename).write_text(_serialize(schema), encoding="utf-8")


def check() -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        generate(tmp_path)

        ok = True
        for filename in SCHEMAS:
            committed = SCHEMAS_DIR / filename
            generated = tmp_path / filename

            if not committed.exists():
                print(f"MISSING: {committed}")
                ok = False
                continue

            committed_text = committed.read_text(encoding="utf-8")
            generated_text = generated.read_text(encoding="utf-8")
            if committed_text != generated_text:
                print(f"DRIFT: {committed} differs from generated output")
                ok = False

        return ok


def main() -> None:
    if "--check" in sys.argv:
        if check():
            print("Schemas are up-to-date.")
        else:
            print(
                "\nSchemas are out of date. "
                "Run `python scripts/generate_schemas.py` to regenerate.",
                file=sys.stderr,
            )
            raise SystemExit(1)
    else:
        generate(SCHEMAS_DIR)
        print(f"Schemas written to {SCHEMAS_DIR}/")


if __name__ == "__main__":
    main()
