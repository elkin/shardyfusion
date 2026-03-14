"""Database-backed ManifestStore implementations (DB-API 2).

Provides PostgreSQL and Comdb2 subclasses that store manifests in a
``shardyfusion_manifests`` table and track the current pointer via an
append-only ``shardyfusion_pointer`` table.

No external dependencies are required beyond DB-API 2 drivers
(user-provided ``psycopg2`` / comdb2 drivers).
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any

from .errors import ConfigValidationError, ManifestParseError, ManifestStoreError
from .logging import FailureSeverity, get_logger, log_failure
from .manifest import (
    JsonManifestBuilder,
    ManifestRef,
    ParsedManifest,
    RequiredBuildMeta,
    RequiredShardMeta,
)
from .manifest_store import parse_json_manifest

_logger = get_logger(__name__)


class _DbManifestStoreBase(ABC):
    """Base class for DB-API 2 manifest stores.

    Subclasses provide dialect-specific DDL and parameter markers.
    """

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        *,
        table_name: str = "shardyfusion_manifests",
        pointer_table_name: str = "shardyfusion_pointer",
        ensure_table: bool = True,
    ) -> None:
        self._connection_factory = connection_factory
        self._table_name = table_name
        self._pointer_table_name = pointer_table_name
        if ensure_table:
            self._ensure_table()

    @property
    @abstractmethod
    def _create_table_ddl(self) -> str:
        """Return dialect-specific CREATE TABLE IF NOT EXISTS DDL."""
        ...

    @property
    @abstractmethod
    def _create_pointer_table_ddl(self) -> str:
        """Return dialect-specific CREATE TABLE DDL for the pointer table."""
        ...

    @property
    @abstractmethod
    def _create_pointer_index_ddl(self) -> str:
        """Return dialect-specific CREATE INDEX DDL for the pointer table."""
        ...

    @property
    @abstractmethod
    def _param_marker(self) -> str:
        """Return the DB-API 2 parameter marker (``%s`` or ``?``)."""
        ...

    def _ensure_table(self) -> None:
        try:
            conn = self._connection_factory()
            try:
                cursor = conn.cursor()
                cursor.execute(self._create_table_ddl)
                cursor.execute(self._create_pointer_table_ddl)
                cursor.execute(self._create_pointer_index_ddl)
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            raise ConfigValidationError(
                f"Failed to create manifest tables '{self._table_name}' / "
                f"'{self._pointer_table_name}': {exc}"
            ) from exc

    def publish(
        self,
        *,
        run_id: str,
        required_build: RequiredBuildMeta,
        shards: list[RequiredShardMeta],
        custom: dict[str, Any],
    ) -> str:
        builder = JsonManifestBuilder()
        artifact = builder.build(
            required_build=required_build,
            shards=shards,
            custom_fields=custom,
        )
        payload_str = artifact.payload.decode("utf-8")

        p = self._param_marker
        upsert_sql = self._upsert_sql(p)
        pointer_sql = self._pointer_insert_sql(p)

        try:
            conn = self._connection_factory()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    upsert_sql,
                    (
                        run_id,
                        "manifest",
                        payload_str,
                        artifact.content_type,
                    ),
                )
                cursor.execute(pointer_sql, (run_id,))
                conn.commit()
            finally:
                conn.close()
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

    @abstractmethod
    def _upsert_sql(self, param_marker: str) -> str:
        """Return dialect-specific INSERT/UPSERT SQL."""
        ...

    def load_current(self) -> ManifestRef | None:
        # Try pointer table first
        pointer_sql = (
            f"SELECT manifest_ref, updated_at "
            f"FROM {self._pointer_table_name} "
            f"ORDER BY updated_at DESC LIMIT 1"
        )
        # Fallback to manifests table
        fallback_sql = (
            f"SELECT run_id, created_at "
            f"FROM {self._table_name} "
            f"ORDER BY created_at DESC LIMIT 1"
        )

        try:
            conn = self._connection_factory()
            try:
                cursor = conn.cursor()
                cursor.execute(pointer_sql)
                row = cursor.fetchone()
                if row is not None:
                    manifest_ref, updated_at = row
                    return ManifestRef(
                        ref=str(manifest_ref),
                        run_id=str(manifest_ref),
                        published_at=datetime.fromisoformat(str(updated_at)),
                    )
                # Pointer table empty — fall back to manifests table
                cursor.execute(fallback_sql)
                row = cursor.fetchone()
            finally:
                conn.close()
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

        run_id, created_at = row
        return ManifestRef(
            ref=str(run_id),
            run_id=str(run_id),
            published_at=datetime.fromisoformat(str(created_at)),
        )

    def load_manifest(self, ref: str) -> ParsedManifest:
        p = self._param_marker
        sql = f"SELECT payload FROM {self._table_name} WHERE run_id = {p}"

        try:
            conn = self._connection_factory()
            try:
                cursor = conn.cursor()
                cursor.execute(sql, (ref,))
                row = cursor.fetchone()
            finally:
                conn.close()
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

        if row is None:
            raise ManifestParseError(f"Manifest not found: ref={ref}")

        payload_str = row[0]
        return parse_json_manifest(
            payload_str.encode("utf-8") if isinstance(payload_str, str) else payload_str
        )

    def list_manifests(self, *, limit: int = 10) -> list[ManifestRef]:
        p = self._param_marker
        sql = (
            f"SELECT run_id, created_at "
            f"FROM {self._table_name} "
            f"ORDER BY created_at DESC LIMIT {p}"
        )

        try:
            conn = self._connection_factory()
            try:
                cursor = conn.cursor()
                cursor.execute(sql, (limit,))
                rows = cursor.fetchall()
            finally:
                conn.close()
        except Exception as exc:
            log_failure(
                "db_manifest_list_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
            )
            raise ManifestStoreError(f"Failed to list manifests: {exc}") from exc

        return [
            ManifestRef(
                ref=str(run_id),
                run_id=str(run_id),
                published_at=datetime.fromisoformat(str(created_at)),
            )
            for run_id, created_at in rows
        ]

    def _pointer_insert_sql(self, param_marker: str) -> str:
        return f"INSERT INTO {self._pointer_table_name} (manifest_ref) VALUES ({param_marker})"

    def set_current(self, ref: str) -> None:
        sql = self._pointer_insert_sql(self._param_marker)

        try:
            conn = self._connection_factory()
            try:
                cursor = conn.cursor()
                cursor.execute(sql, (ref,))
                conn.commit()
            finally:
                conn.close()
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


class PostgresManifestStore(_DbManifestStoreBase):
    """PostgreSQL manifest store using JSONB and TIMESTAMPTZ."""

    @property
    def _param_marker(self) -> str:
        return "%s"

    @property
    def _create_table_ddl(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self._table_name} ("
            f"  run_id       TEXT PRIMARY KEY,"
            f"  name         TEXT NOT NULL,"
            f"  payload      JSONB NOT NULL,"
            f"  content_type TEXT NOT NULL DEFAULT 'application/json',"
            f"  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()"
            f")"
        )

    @property
    def _create_pointer_table_ddl(self) -> str:
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

    def _upsert_sql(self, param_marker: str) -> str:
        p = param_marker
        return (
            f"INSERT INTO {self._table_name} (run_id, name, payload, content_type) "
            f"VALUES ({p}, {p}, {p}, {p}) "
            f"ON CONFLICT (run_id) DO UPDATE SET "
            f"  name = EXCLUDED.name, "
            f"  payload = EXCLUDED.payload, "
            f"  content_type = EXCLUDED.content_type, "
            f"  created_at = NOW()"
        )


class Comdb2ManifestStore(_DbManifestStoreBase):
    """Comdb2 manifest store with dialect-specific SQL."""

    @property
    def _param_marker(self) -> str:
        return "%s"

    @property
    def _create_table_ddl(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self._table_name} ("
            f"  run_id       TEXT PRIMARY KEY,"
            f"  name         TEXT NOT NULL,"
            f"  payload      TEXT NOT NULL,"
            f"  content_type TEXT NOT NULL DEFAULT 'application/json',"
            f"  created_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
            f")"
        )

    @property
    def _create_pointer_table_ddl(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self._pointer_table_name} ("
            f"  updated_at    TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,"
            f"  manifest_ref  TEXT NOT NULL"
            f")"
        )

    @property
    def _create_pointer_index_ddl(self) -> str:
        return (
            f"CREATE INDEX IF NOT EXISTS idx_{self._pointer_table_name}_updated_at "
            f"ON {self._pointer_table_name} (updated_at DESC)"
        )

    def _upsert_sql(self, param_marker: str) -> str:
        p = param_marker
        return (
            f"INSERT INTO {self._table_name} (run_id, name, payload, content_type) "
            f"VALUES ({p}, {p}, {p}, {p}) "
            f"ON CONFLICT (run_id) DO UPDATE SET "
            f"  name = EXCLUDED.name, "
            f"  payload = EXCLUDED.payload, "
            f"  content_type = EXCLUDED.content_type, "
            f"  created_at = CURRENT_TIMESTAMP"
        )
