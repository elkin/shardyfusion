"""Run-scoped writer registry used by periodic cleanup workflows."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from enum import Enum
from types import TracebackType
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict

from .credentials import CredentialProvider
from .logging import FailureSeverity, get_logger, log_failure
from .storage import ObstoreBackend, StorageBackend, create_s3_store, join_s3
from .type_defs import S3ConnectionOptions

if TYPE_CHECKING:
    from .manifest_store import ManifestStore

_logger = get_logger(__name__)

RUN_REGISTRY_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
RUN_RECORD_NAME = "run.yaml"
RUN_RECORD_FORMAT_VERSION = 1
DEFAULT_RUN_LEASE_DURATION = timedelta(minutes=10)
DEFAULT_RUN_HEARTBEAT_INTERVAL = timedelta(minutes=1)


class RunStatus(str, Enum):
    """Terminal and in-flight writer states stored in the run registry."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RunRecord(BaseModel):
    """One driver-owned run record describing a writer invocation."""

    model_config = ConfigDict(use_enum_values=False)

    format_version: int = RUN_RECORD_FORMAT_VERSION
    run_id: str
    writer_type: str
    status: RunStatus
    started_at: datetime
    updated_at: datetime
    lease_expires_at: datetime
    s3_prefix: str
    shard_prefix: str
    db_path_template: str
    manifest_ref: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class RunRegistry(Protocol):
    """Persistence interface for run records."""

    def create(self, record: RunRecord) -> str:
        """Persist a new run record and return its stable reference."""
        ...

    def update(self, ref: str, record: RunRecord) -> None:
        """Overwrite an existing run record."""
        ...

    def load(self, ref: str) -> RunRecord:
        """Load a run record by reference."""
        ...


class _RunRecordOutputConfig(Protocol):
    @property
    def shard_prefix(self) -> str: ...

    @property
    def db_path_template(self) -> str: ...

    @property
    def run_registry_prefix(self) -> str: ...


class _RunRecordManifestConfig(Protocol):
    @property
    def store(self) -> ManifestStore | None: ...


class _RunRecordConfig(Protocol):
    @property
    def s3_prefix(self) -> str: ...

    @property
    def output(self) -> _RunRecordOutputConfig: ...

    @property
    def manifest(self) -> _RunRecordManifestConfig: ...

    @property
    def run_registry(self) -> RunRegistry | None: ...

    @property
    def credential_provider(self) -> CredentialProvider | None: ...

    @property
    def s3_connection_options(self) -> S3ConnectionOptions | None: ...


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _format_run_timestamp(dt: datetime) -> str:
    return dt.strftime(RUN_REGISTRY_TIMESTAMP_FMT)


def _build_run_payload(record: RunRecord) -> bytes:
    return yaml.safe_dump(
        record.model_dump(mode="json"),
        sort_keys=True,
        default_flow_style=False,
    ).encode("utf-8")


def parse_run_record(payload: bytes) -> RunRecord:
    """Parse a raw YAML run record payload."""
    data = yaml.safe_load(payload)
    if not isinstance(data, dict):
        raise ValueError("Run record payload must decode to a mapping")
    return RunRecord.model_validate(data)


class S3RunRegistry:
    """S3-backed run registry storing one YAML object per run."""

    def __init__(
        self,
        backend: StorageBackend,
        s3_prefix: str,
        *,
        run_registry_prefix: str = "runs",
    ) -> None:
        self._backend = backend
        self.s3_prefix = s3_prefix.rstrip("/")
        self.run_registry_prefix = run_registry_prefix

    def create(self, record: RunRecord) -> str:
        ref = join_s3(
            self.s3_prefix,
            self.run_registry_prefix,
            f"{_format_run_timestamp(_utc_now())}_run_id={record.run_id}_{uuid4().hex}",
            RUN_RECORD_NAME,
        )
        self.update(ref, record)
        return ref

    def update(self, ref: str, record: RunRecord) -> None:
        self._backend.put(
            ref,
            _build_run_payload(record),
            "application/x-yaml",
        )

    def load(self, ref: str) -> RunRecord:
        payload = self._backend.get(ref)
        return parse_run_record(payload)


class InMemoryRunRegistry:
    """In-memory run registry used by tests and local-only flows."""

    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}

    def create(self, record: RunRecord) -> str:
        ref = f"mem://runs/{record.run_id}/{uuid4().hex}/{RUN_RECORD_NAME}"
        self.update(ref, record)
        return ref

    def update(self, ref: str, record: RunRecord) -> None:
        self._records[ref] = record.model_copy(deep=True)

    def load(self, ref: str) -> RunRecord:
        return self._records[ref].model_copy(deep=True)


def resolve_run_registry(config: _RunRecordConfig) -> RunRegistry:
    """Resolve the run registry for this write configuration."""

    if config.run_registry is not None:
        return config.run_registry

    from .manifest_store import InMemoryManifestStore

    if isinstance(config.manifest.store, InMemoryManifestStore):
        return InMemoryRunRegistry()

    # The run registry is writer-scoped metadata, not manifest-store metadata.
    credentials = config.credential_provider
    conn_opts = config.s3_connection_options
    from .storage import parse_s3_url

    bucket, _ = parse_s3_url(config.s3_prefix)
    store = create_s3_store(
        bucket=bucket,
        credentials=credentials.resolve() if credentials else None,
        connection_options=conn_opts,
    )
    backend = ObstoreBackend(store)
    return S3RunRegistry(
        backend=backend,
        s3_prefix=config.s3_prefix,
        run_registry_prefix=config.output.run_registry_prefix,
    )


class RunRecordLifecycle:
    """Driver-owned lifecycle helper that keeps one run record current."""

    def __init__(
        self,
        *,
        registry: RunRegistry | None,
        record: RunRecord,
        ref: str | None,
        lease_duration: timedelta = DEFAULT_RUN_LEASE_DURATION,
        heartbeat_interval: timedelta = DEFAULT_RUN_HEARTBEAT_INTERVAL,
    ) -> None:
        self._registry = registry
        self._record = record
        self._ref = ref
        self._lease_duration = lease_duration
        self._heartbeat_interval = heartbeat_interval
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._terminal = False
        self._heartbeat_thread: threading.Thread | None = None

        if self._registry is not None and self._ref is not None:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"shardy-run-heartbeat-{record.run_id}",
                daemon=True,
            )
            self._heartbeat_thread.start()

    @property
    def run_record_ref(self) -> str | None:
        return self._ref

    def __enter__(self) -> RunRecordLifecycle:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        # The run registry is observability metadata: its own failures must never
        # mask the user's in-flight exception or leak the heartbeat thread.
        # Both branches swallow registry-side errors via log_failure(); returning
        # None re-raises any in-flight exception from the with-block.
        try:
            if exc is not None:
                self._safe_mark_failed(exc)
        finally:
            if not self._terminal:
                self._safe_close()

    def _safe_mark_failed(self, exc: BaseException) -> None:
        try:
            self.mark_failed(exc)
        except Exception as registry_exc:
            log_failure(
                "run_record_mark_failed_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=registry_exc,
                run_id=self._record.run_id,
                run_record_ref=self._ref,
                include_traceback=True,
            )

    def _safe_close(self) -> None:
        try:
            self.close()
        except Exception as registry_exc:
            log_failure(
                "run_record_close_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=registry_exc,
                run_id=self._record.run_id,
                run_record_ref=self._ref,
                include_traceback=True,
            )

    @classmethod
    def start(
        cls,
        *,
        config: _RunRecordConfig,
        run_id: str,
        writer_type: str,
        lease_duration: timedelta = DEFAULT_RUN_LEASE_DURATION,
        heartbeat_interval: timedelta = DEFAULT_RUN_HEARTBEAT_INTERVAL,
    ) -> RunRecordLifecycle:
        now = _utc_now()
        record = RunRecord(
            run_id=run_id,
            writer_type=writer_type,
            status=RunStatus.RUNNING,
            started_at=now,
            updated_at=now,
            lease_expires_at=now + lease_duration,
            s3_prefix=config.s3_prefix,
            shard_prefix=config.output.shard_prefix,
            db_path_template=config.output.db_path_template,
        )
        try:
            registry = resolve_run_registry(config)
            ref = registry.create(record)
        except Exception as exc:
            log_failure(
                "run_record_create_failed",
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                run_id=run_id,
                writer_type=writer_type,
                include_traceback=True,
            )
            registry = None
            ref = None
        return cls(
            registry=registry,
            record=record,
            ref=ref,
            lease_duration=lease_duration,
            heartbeat_interval=heartbeat_interval,
        )

    def set_manifest_ref(self, manifest_ref: str) -> None:
        with self._lock:
            self._record.manifest_ref = manifest_ref

    def mark_succeeded(self) -> None:
        self._mark_terminal(status=RunStatus.SUCCEEDED)

    def mark_failed(self, exc: BaseException) -> None:
        self._mark_terminal(
            status=RunStatus.FAILED,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    def close(self) -> None:
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(
                timeout=self._heartbeat_interval.total_seconds()
            )

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_interval.total_seconds()):
            with self._lock:
                if self._terminal:
                    return
                now = _utc_now()
                self._record.updated_at = now
                self._record.lease_expires_at = now + self._lease_duration
                record = self._record.model_copy(deep=True)
            self._safe_update("run_record_heartbeat_failed", record)

    def _mark_terminal(
        self,
        *,
        status: RunStatus,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._lock:
            if self._terminal:
                return
            self._terminal = True
            now = _utc_now()
            self._record.status = status
            self._record.updated_at = now
            self._record.lease_expires_at = now
            self._record.error_type = error_type
            self._record.error_message = error_message
            record = self._record.model_copy(deep=True)
        self.close()
        self._safe_update("run_record_terminal_update_failed", record)

    def _safe_update(self, event: str, record: RunRecord) -> None:
        if self._registry is None or self._ref is None:
            return
        try:
            self._registry.update(self._ref, record)
        except Exception as exc:
            log_failure(
                event,
                severity=FailureSeverity.ERROR,
                logger=_logger,
                error=exc,
                run_id=record.run_id,
                run_record_ref=self._ref,
                include_traceback=True,
            )
