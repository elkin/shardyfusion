"""LanceDB-based vector index adapter.

lancedb is imported lazily so the module can be imported without the
``vector-lancedb`` extra installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...errors import VectorIndexError, VectorSearchError
from ...logging import get_logger, log_event
from ...storage import ObstoreBackend, create_s3_store, parse_s3_url
from ...type_defs import Manifest
from ..types import (
    DistanceMetric,
    SearchResult,
)

_logger = get_logger(__name__)


def _import_lancedb() -> Any:
    try:
        import lancedb  # type: ignore[import-untyped]
        import pyarrow as pa  # type: ignore[import-untyped]
    except ImportError as exc:
        raise VectorIndexError(
            "lancedb and pyarrow are required for LanceDB adapter. "
            "Install with: pip install 'shardyfusion[vector-lancedb]'"
        ) from exc
    return lancedb, pa


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class LanceDbWriter:
    """Builds a LanceDB dataset locally and uploads to S3."""

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        dim: int,
        metric: DistanceMetric,
        quantization: str | None = None,
        index_params: dict[str, Any] | None = None,
        storage_options: dict[str, str] | None = None,
        credential_provider: Any | None = None,
        s3_connection_options: Any | None = None,
    ) -> None:
        lancedb, pa = _import_lancedb()
        self._db_url = db_url
        self._local_dir = local_dir
        self._storage_options = storage_options
        self._credential_provider = credential_provider
        self._s3_connection_options = s3_connection_options
        self._local_dir.mkdir(parents=True, exist_ok=True)

        self._metric = metric
        self._dim = dim
        self._quantization = quantization
        self._index_params = index_params or {}

        # LanceDB schema: id (string), vector (fixed size list), payload (string/JSON)
        # We store id as string to preserve both int and str IDs properly, handling
        # exact match queries on strings without type confusion.
        self._schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
                pa.field("payload", pa.string()),
            ]
        )

        self._db = lancedb.connect(str(self._local_dir))
        self._table = None
        self._closed = False
        self._count = 0
        self._db_bytes = 0
        self._table_name = "vector_index"

    def add_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._closed:
            raise VectorIndexError("Writer is closed")

        lancedb, pa = _import_lancedb()
        n = len(ids)

        str_ids = [str(i) for i in ids]

        # Serialize payloads to JSON strings or None
        serialized_payloads = []
        if payloads is not None:
            serialized_payloads = [
                json.dumps(p) if p is not None else None for p in payloads
            ]
        else:
            serialized_payloads = [None] * n

        data = [
            {
                "id": str_ids[i],
                "vector": vectors[i].tolist(),
                "payload": serialized_payloads[i],
            }
            for i in range(n)
        ]

        if self._table is None:
            self._table = self._db.create_table(
                self._table_name,
                data=data,
                schema=self._schema,
            )
        else:
            self._table.add(data)

        self._count += n

    def flush(self) -> None:
        if self._closed:
            return
        if self._count == 0 or self._table is None:
            return

        # Create the HNSW index on flush if we have enough records
        # Note: LanceDB IVF_PQ requires at least num_partitions * 256 records usually,
        # but for small datasets it can fail if num_partitions is too large.
        try:
            m = self._index_params.get("M", 20)
            ef = self._index_params.get("ef_construction", 300)

            # Map distance metric to LanceDB expected strings
            metric_map = {
                DistanceMetric.COSINE: "cosine",
                DistanceMetric.L2: "l2",
                DistanceMetric.DOT_PRODUCT: "dot",
            }
            metric_str = metric_map.get(self._metric, "cosine")

            # Determine num_partitions based on count (heuristic: ~256 records per partition)
            num_partitions = max(1, min(self._count // 256, 256))

            if self._quantization == "fp16":
                # Create IVF_HNSW_SQ
                from lancedb.index import (  # pyright: ignore[reportMissingImports]
                    HnswSq,
                )

                index_config = HnswSq(
                    distance_type=metric_str,  # type: ignore
                    m=m,
                    ef_construction=ef,
                    num_partitions=num_partitions,
                )
            else:
                # Create IVF_HNSW_PQ
                from lancedb.index import (  # pyright: ignore[reportMissingImports]
                    HnswPq,
                )

                num_sub_vectors = min(self._dim // 16, 1) if self._dim >= 16 else 1
                index_config = HnswPq(
                    distance_type=metric_str,  # type: ignore
                    m=m,
                    ef_construction=ef,
                    num_partitions=num_partitions,
                    num_sub_vectors=num_sub_vectors,
                )

            self._table.create_index("vector", config=index_config)
        except Exception as exc:
            _logger.warning(
                "Failed to create HNSW index during flush, likely too few records: %s",
                exc,
            )

    def checkpoint(self) -> str | None:
        self.flush()
        if self._table is None:
            return None
        # Sum the local .lance dataset directory size for db_bytes reporting.
        dataset_dir = self._local_dir / f"{self._table_name}.lance"
        if dataset_dir.exists():
            import os

            total = 0
            for root, _dirs, files in os.walk(dataset_dir):
                for f in files:
                    try:
                        total += (Path(root) / f).stat().st_size
                    except OSError:
                        pass
            self._db_bytes = total
        return (
            "checkpoint"  # LanceDB checkpoints could use version but we just use string
        )

    def db_bytes(self) -> int:
        return self._db_bytes

    def close(self) -> None:
        if self._closed:
            return
        self.flush()

        if self._table is not None:
            # Upload the entire .lance directory to S3
            dataset_dir = self._local_dir / f"{self._table_name}.lance"
            if dataset_dir.exists():
                self._upload_dir(
                    dataset_dir, f"{self._db_url}/{self._table_name}.lance"
                )

        self._closed = True

        log_event(
            "lancedb_shard_uploaded",
            logger=_logger,
            db_url=self._db_url,
            vector_count=self._count,
        )

    def _upload_dir(self, local_path: Path, remote_url: str) -> None:
        import os

        credentials = (
            self._credential_provider.resolve() if self._credential_provider else None
        )
        bucket, _ = parse_s3_url(remote_url)
        store = create_s3_store(
            bucket=bucket,
            credentials=credentials,
            connection_options=self._s3_connection_options,
        )
        backend = ObstoreBackend(store)

        for root, _dirs, files in os.walk(local_path):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(local_path)
                file_url = f"{remote_url}/{rel_path}"

                with open(file_path, "rb") as f:
                    content = f.read()
                    backend.put(
                        file_url,
                        content,
                        "application/octet-stream",
                    )

    def __enter__(self) -> LanceDbWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class LanceDbWriterFactory:
    """Factory that creates LanceDbWriter instances."""

    def __init__(
        self,
        s3_connection_options: Any | None = None,
        credential_provider: Any | None = None,
    ) -> None:
        self._s3_connection_options = s3_connection_options
        self._credential_provider = credential_provider

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
    ) -> LanceDbWriter:
        credentials = (
            self._credential_provider.resolve() if self._credential_provider else None
        )
        storage_options = _make_lancedb_storage_options(
            credentials, self._s3_connection_options
        )
        return LanceDbWriter(
            db_url=db_url,
            local_dir=local_dir,
            dim=index_config.dim,
            metric=index_config.metric,
            quantization=index_config.quantization,
            index_params=index_config.index_params,
            storage_options=storage_options,
            credential_provider=self._credential_provider,
            s3_connection_options=self._s3_connection_options,
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _make_lancedb_storage_options(
    credentials: Any | None = None,
    connection_options: Any | None = None,
) -> dict[str, str] | None:
    """Build LanceDB storage_options dict from shardyfusion config."""
    opts: dict[str, str] = {}
    conn = connection_options or {}

    endpoint = conn.get("endpoint_url")
    if endpoint:
        opts["aws_endpoint"] = endpoint
        if endpoint.startswith("http://"):
            opts["aws_allow_http"] = "true"

    region = conn.get("region_name")
    if region:
        opts["aws_region"] = region

    if credentials is not None:
        if getattr(credentials, "access_key_id", None):
            opts["aws_access_key_id"] = credentials.access_key_id
        if getattr(credentials, "secret_access_key", None):
            opts["aws_secret_access_key"] = credentials.secret_access_key
        if getattr(credentials, "session_token", None):
            opts["aws_session_token"] = credentials.session_token

    return opts if opts else None


class LanceDbShardReader:
    """Connects to remote LanceDB dataset on S3 and searches remotely."""

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
        storage_options: dict[str, str] | None = None,
    ) -> None:
        lancedb, _ = _import_lancedb()
        self._db_url = db_url
        self._table_name = "vector_index"

        # We connect directly to the db_url (assuming it's s3://...)
        self._db = lancedb.connect(db_url, storage_options=storage_options)
        self._closed = False

        try:
            self._table = self._db.open_table(self._table_name)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"FAILED TO OPEN TABLE {self._table_name} AT {db_url}: {exc}")
            self._table = None

    def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        if self._closed:
            raise VectorSearchError("Reader is closed")

        if self._table is None:
            return []

        try:
            # Search returns a pyarrow Table
            matches = (
                self._table.search(query.astype(np.float32)).limit(top_k).to_arrow()
            )
        except Exception as exc:
            raise VectorSearchError(f"Search failed: {exc}") from exc

        results: list[SearchResult] = []

        # Process PyArrow table
        for i in range(len(matches)):
            row = matches.take([i]).to_pylist()[0]

            orig_id: int | str = row["id"]

            score = float(row.get("_distance", 0.0))

            payload_str = row.get("payload")
            payload = None
            if payload_str is not None:
                try:
                    payload = json.loads(payload_str)
                except Exception:
                    pass

            results.append(
                SearchResult(
                    id=orig_id,
                    score=score,
                    payload=payload,
                )
            )

        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True


class LanceDbReaderFactory:
    """Factory that creates LanceDbShardReader instances."""

    def __init__(
        self,
        s3_connection_options: Any | None = None,
        credential_provider: Any | None = None,
    ) -> None:
        self._s3_connection_options = s3_connection_options
        self._credential_provider = credential_provider

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
        manifest: Manifest,
    ) -> LanceDbShardReader:
        del manifest  # unused in concrete factory
        credentials = (
            self._credential_provider.resolve() if self._credential_provider else None
        )
        storage_options = _make_lancedb_storage_options(
            credentials, self._s3_connection_options
        )
        return LanceDbShardReader(
            db_url=db_url,
            local_dir=local_dir,
            index_config=index_config,
            storage_options=storage_options,
        )


class AsyncLanceDbShardReader:
    """Connects to remote LanceDB dataset on S3 and searches asynchronously."""

    def __init__(self) -> None:
        # Private constructor: instantiate via open() or factory
        self._db: Any = None
        self._table: Any = None
        self._closed = False
        self._table_name = "vector_index"

    @classmethod
    async def open(
        cls,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
        storage_options: dict[str, str] | None = None,
    ) -> AsyncLanceDbShardReader:
        lancedb, _ = _import_lancedb()
        instance = cls()
        try:
            # We connect to LanceDB natively using async API
            instance._db = await lancedb.connect_async(
                db_url, storage_options=storage_options
            )
            instance._table = await instance._db.open_table(instance._table_name)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(
                f"FAILED TO OPEN ASYNC TABLE {instance._table_name} AT {db_url}: {exc}"
            )
            instance._table = None
        return instance

    async def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        if self._closed:
            raise VectorSearchError("Reader is closed")

        if self._table is None:
            return []

        try:
            query_builder = await self._table.search(query.astype(np.float32))
            query_builder = query_builder.limit(top_k)
            matches = await query_builder.to_arrow()
        except Exception as exc:
            raise VectorSearchError(f"Search failed: {exc}") from exc

        results: list[SearchResult] = []
        for i in range(len(matches)):
            row = matches.take([i]).to_pylist()[0]
            orig_id: int | str = row["id"]
            score = float(row.get("_distance", 0.0))

            payload_str = row.get("payload")
            payload = None
            if payload_str is not None:
                try:
                    payload = json.loads(payload_str)
                except Exception:
                    pass

            results.append(
                SearchResult(
                    id=orig_id,
                    score=score,
                    payload=payload,
                )
            )

        return results

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True


class AsyncLanceDbReaderFactory:
    """Factory that creates AsyncLanceDbShardReader instances."""

    def __init__(
        self,
        s3_connection_options: Any | None = None,
        credential_provider: Any | None = None,
    ) -> None:
        self._s3_connection_options = s3_connection_options
        self._credential_provider = credential_provider

    async def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
        manifest: Manifest,
    ) -> AsyncLanceDbShardReader:
        del manifest  # unused in concrete factory
        credentials = (
            self._credential_provider.resolve() if self._credential_provider else None
        )
        storage_options = _make_lancedb_storage_options(
            credentials, self._s3_connection_options
        )
        return await AsyncLanceDbShardReader.open(
            db_url=db_url,
            local_dir=local_dir,
            index_config=index_config,
            storage_options=storage_options,
        )
