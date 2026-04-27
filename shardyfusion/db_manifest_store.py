"""Database-backed ManifestStore implementation (PostgreSQL).

Stores manifests in normalized ``_builds`` + ``_shards`` tables and tracks
the current pointer via an append-only ``_pointer`` table.

No external dependencies beyond a DB-API 2 driver (user-provided ``psycopg2``).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from .errors import ConfigValidationError, ManifestParseError, ManifestStoreError
from .logging import FailureSeverity, get_logger, log_failure
from .manifest import (
    ManifestRef,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from .manifest_store import _validate_manifest

_logger = get_logger(__name__)


def _json_col(value: object) -> str | None:
    """Serialize a value to a JSON string for a JSONB column, or None."""
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _parse_json_col(value: object) -> Any:
    """Parse a JSONB column value — psycopg2 returns dict, other drivers may return str."""
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    return json.loads(value)  # type: ignore[arg-type]


def _make_ref(run_id: object, timestamp: object) -> ManifestRef:
    """Build a ManifestRef from DB row values."""
    rid = str(run_id)
    return ManifestRef(
        ref=rid,
        run_id=rid,
        published_at=datetime.fromisoformat(str(timestamp)),
    )


class PostgresManifestStore:
    """PostgreSQL manifest store with normalized builds + shards tables."""

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        *,
        table_name: str = "shardyfusion_manifests",
        pointer_table_name: str = "shardyfusion_pointer",
        ensure_table: bool = True,
    ) -> None:
        self._connection_factory = connection_factory
        self._builds_table = f"{table_name}_builds"
        self._shards_table = f"{table_name}_shards"
        self._pointer_table_name = pointer_table_name
        if ensure_table:
            self._ensure_table()

    # -- helpers ------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[Any, None, None]:
        conn = self._connection_factory()
        try:
            yield conn
        finally:
            conn.close()

    # -- DDL ----------------------------------------------------------------

    @property
    def _create_builds_ddl(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self._builds_table} ("
            f"  run_id           TEXT PRIMARY KEY,"
            f"  created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
            f"  num_dbs          INTEGER NOT NULL,"
            f"  s3_prefix        TEXT NOT NULL,"
            f"  key_col          TEXT NOT NULL DEFAULT '_key',"
            f"  db_path_template TEXT NOT NULL DEFAULT 'db={{db_id:05d}}',"
            f"  shard_prefix     TEXT NOT NULL DEFAULT 'shards',"
            f"  key_encoding     TEXT NOT NULL DEFAULT 'u64be',"
            f"  sharding         JSONB NOT NULL,"
            f"  custom           JSONB NOT NULL DEFAULT '{{}}'"
            f")"
        )

    @property
    def _create_shards_ddl(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self._shards_table} ("
            f"  run_id        TEXT NOT NULL REFERENCES {self._builds_table}(run_id),"
            f"  db_id         INTEGER NOT NULL,"
            f"  db_url        TEXT,"
            f"  attempt       INTEGER NOT NULL DEFAULT 0,"
            f"  row_count     INTEGER NOT NULL DEFAULT 0,"
            f"  db_bytes      BIGINT  NOT NULL DEFAULT 0,"
            f"  checkpoint_id TEXT,"
            f"  min_key       JSONB,"
            f"  max_key       JSONB,"
            f"  writer_info   JSONB,"
            f"  PRIMARY KEY (run_id, db_id)"
            f")"
        )

    @property
    def _create_pointer_ddl(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self._pointer_table_name} ("
            f"  updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),"
            f"  manifest_ref  TEXT NOT NULL"
            f")"
        )

    @property
    def _create_pointer_index_ddl(self) -> str:
        return (
            f"CREATE INDEX IF NOT EXISTS idx_{self._pointer_table_name}_updated_at "
            f"ON {self._pointer_table_name} (updated_at DESC)"
        )

    def _ensure_table(self) -> None:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(self._create_builds_ddl)
                cursor.execute(self._create_shards_ddl)
                cursor.execute(self._create_pointer_ddl)
                cursor.execute(self._create_pointer_index_ddl)
                conn.commit()
        except Exception as exc:
            raise ConfigValidationError(
                f"Failed to create manifest tables: {exc}"
            ) from exc

    # -- ManifestStore protocol ---------------------------------------------

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        build = required_build.model_dump(mode="json")

        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"INSERT INTO {self._builds_table} "
                    f"  (run_id, created_at, num_dbs, s3_prefix, key_col,"
                    f"   db_path_template, shard_prefix, key_encoding,"
                    f"   sharding, custom) "
                    f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                    f"ON CONFLICT (run_id) DO UPDATE SET "
                    f"  created_at = EXCLUDED.created_at,"
                    f"  num_dbs = EXCLUDED.num_dbs,"
                    f"  s3_prefix = EXCLUDED.s3_prefix,"
                    f"  key_col = EXCLUDED.key_col,"
                    f"  db_path_template = EXCLUDED.db_path_template,"
                    f"  shard_prefix = EXCLUDED.shard_prefix,"
                    f"  key_encoding = EXCLUDED.key_encoding,"
                    f"  sharding = EXCLUDED.sharding,"
                    f"  custom = EXCLUDED.custom",
                    (
                        run_id,
                        build["created_at"],
                        build["num_dbs"],
                        build["s3_prefix"],
                        build["key_col"],
                        build["db_path_template"],
                        build["shard_prefix"],
                        build["key_encoding"],
                        _json_col(build["sharding"]),
                        _json_col(custom),
                    ),
                )

                cursor.execute(
                    f"DELETE FROM {self._shards_table} WHERE run_id = %s",
                    (run_id,),
                )

                if shards:
                    shard_rows = []
                    for shard in shards:
                        sd = shard.model_dump(mode="json")
                        shard_rows.append(
                            (
                                run_id,
                                sd["db_id"],
                                sd["db_url"],
                                sd["attempt"],
                                sd["row_count"],
                                sd["db_bytes"],
                                sd["checkpoint_id"],
                                _json_col(sd["min_key"]),
                                _json_col(sd["max_key"]),
                                _json_col(sd["writer_info"]),
                            )
                        )
                    cursor.executemany(
                        f"INSERT INTO {self._shards_table} "
                        f"  (run_id, db_id, db_url, attempt, row_count, db_bytes,"
                        f"   checkpoint_id, min_key, max_key, writer_info) "
                        f"VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        shard_rows,
                    )

                cursor.execute(
                    f"INSERT INTO {self._pointer_table_name} (manifest_ref) VALUES (%s)",
                    (run_id,),
                )

                conn.commit()
        except ManifestStoreError:
            raise
        except Exception as exc:
            log_failure(
                "db_manifest_publish_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                run_id=run_id,
            )
            raise ManifestStoreError(
                f"Failed to publish manifest for run_id={run_id}: {exc}"
            ) from exc

        return run_id

    def load_current(self) -> ManifestRef | None:
        pointer_sql = (
            f"SELECT manifest_ref, updated_at "
            f"FROM {self._pointer_table_name} "
            f"ORDER BY updated_at DESC LIMIT 1"
        )
        fallback_sql = (
            f"SELECT run_id, created_at "
            f"FROM {self._builds_table} "
            f"ORDER BY created_at DESC LIMIT 1"
        )

        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(pointer_sql)
                row = cursor.fetchone()
                if row is not None:
                    return _make_ref(row[0], row[1])
                cursor.execute(fallback_sql)
                row = cursor.fetchone()
        except Exception as exc:
            log_failure(
                "db_manifest_load_current_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
            )
            raise ManifestStoreError(f"Failed to load current manifest: {exc}") from exc

        if row is None:
            return None
        return _make_ref(row[0], row[1])

    def load_manifest(self, ref: str) -> ParsedManifest:
        try:
            with self._connect() as conn:
                cursor = conn.cursor()

                cursor.execute(
                    f"SELECT run_id, created_at, num_dbs, s3_prefix, key_col,"
                    f"       db_path_template, shard_prefix, key_encoding,"
                    f"       sharding, custom "
                    f"FROM {self._builds_table} WHERE run_id = %s",
                    (ref,),
                )
                build_row = cursor.fetchone()
                if build_row is None:
                    raise ManifestParseError(f"Manifest not found: ref={ref}")

                cursor.execute(
                    f"SELECT db_id, db_url, attempt, row_count, db_bytes, checkpoint_id,"
                    f"       min_key, max_key, writer_info "
                    f"FROM {self._shards_table} WHERE run_id = %s ORDER BY db_id",
                    (ref,),
                )
                shard_rows = cursor.fetchall()
        except ManifestParseError:
            raise
        except Exception as exc:
            log_failure(
                "db_manifest_load_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                manifest_ref=ref,
            )
            raise ManifestStoreError(
                f"Failed to load manifest ref={ref}: {exc}"
            ) from exc

        (
            run_id,
            created_at,
            num_dbs,
            s3_prefix,
            key_col,
            db_path_template,
            shard_prefix,
            key_encoding,
            sharding_raw,
            custom_raw,
        ) = build_row

        build_data = {
            "run_id": run_id,
            "created_at": str(created_at),
            "num_dbs": num_dbs,
            "s3_prefix": s3_prefix,
            "key_col": key_col,
            "db_path_template": db_path_template,
            "shard_prefix": shard_prefix,
            "key_encoding": key_encoding,
            "sharding": _parse_json_col(sharding_raw),
        }

        shards_data = []
        for (
            db_id,
            db_url,
            attempt,
            row_count,
            db_bytes,
            checkpoint_id,
            min_key_raw,
            max_key_raw,
            writer_info_raw,
        ) in shard_rows:
            shard_dict: dict[str, Any] = {
                "db_id": db_id,
                "db_url": db_url,
                "attempt": attempt,
                "row_count": row_count,
                "db_bytes": db_bytes,
                "checkpoint_id": checkpoint_id,
                "min_key": _parse_json_col(min_key_raw),
                "max_key": _parse_json_col(max_key_raw),
            }
            wi = _parse_json_col(writer_info_raw)
            if wi:
                shard_dict["writer_info"] = wi
            shards_data.append(shard_dict)

        custom = _parse_json_col(custom_raw) or {}
        data = {"required": build_data, "shards": shards_data, "custom": custom}
        try:
            parsed = ParsedManifest.model_validate(data)
        except ValidationError as exc:
            raise ManifestParseError(f"Manifest validation failed: {exc}") from exc
        _validate_manifest(parsed.required_build, parsed.shards)
        return parsed

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        sql = (
            f"SELECT run_id, created_at "
            f"FROM {self._builds_table} "
            f"ORDER BY created_at DESC LIMIT %s"
        )

        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (limit,))
                rows = cursor.fetchall()
        except Exception as exc:
            log_failure(
                "db_manifest_list_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
            )
            raise ManifestStoreError(f"Failed to list manifests: {exc}") from exc

        return [_make_ref(run_id, created_at) for run_id, created_at in rows]

    def set_current(self, ref: str) -> None:
        sql = f"INSERT INTO {self._pointer_table_name} (manifest_ref) VALUES (%s)"

        try:
            with self._connect() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (ref,))
                conn.commit()
        except Exception as exc:
            log_failure(
                "db_manifest_set_current_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                manifest_ref=ref,
            )
            raise ManifestStoreError(
                f"Failed to set current manifest ref={ref}: {exc}"
            ) from exc
