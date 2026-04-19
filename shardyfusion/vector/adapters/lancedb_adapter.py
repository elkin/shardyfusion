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
from ...storage import put_bytes
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
        s3_client: Any | None = None,
    ) -> None:
        lancedb, pa = _import_lancedb()
        self._db_url = db_url
        self._local_dir = local_dir
        self._s3_client = s3_client
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
        return (
            "checkpoint"  # LanceDB checkpoints could use version but we just use string
        )

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

        for root, _dirs, files in os.walk(local_path):
            for file in files:
                file_path = Path(root) / file
                rel_path = file_path.relative_to(local_path)
                file_url = f"{remote_url}/{rel_path}"

                with open(file_path, "rb") as f:
                    content = f.read()
                    put_bytes(
                        file_url,
                        content,
                        "application/octet-stream",
                        s3_client=self._s3_client,
                    )

    def __enter__(self) -> LanceDbWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class LanceDbWriterFactory:
    """Factory that creates LanceDbWriter instances."""

    def __init__(self, s3_client: Any | None = None) -> None:
        self._s3_client = s3_client
        self._s3_config: dict[str, Any] | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_s3_client", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._s3_client = None

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
    ) -> LanceDbWriter:
        return LanceDbWriter(
            db_url=db_url,
            local_dir=local_dir,
            dim=index_config.dim,
            metric=index_config.metric,
            quantization=index_config.quantization,
            index_params=index_config.index_params,
            s3_client=self._s3_client,
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _extract_storage_options(s3_client: Any | None) -> dict[str, str] | None:
    if s3_client is None or not hasattr(s3_client, "meta"):
        return None

    options: dict[str, str] = {}
    if s3_client.meta.endpoint_url:
        options["aws_endpoint"] = s3_client.meta.endpoint_url
        if s3_client.meta.endpoint_url.startswith("http://"):
            options["aws_allow_http"] = "true"
    if s3_client.meta.region_name:
        options["aws_region"] = s3_client.meta.region_name

    # Try to extract credentials
    if hasattr(s3_client, "_request_signer") and hasattr(
        s3_client._request_signer, "_credentials"
    ):
        creds = s3_client._request_signer._credentials
        if creds:
            if getattr(creds, "access_key", None):
                options["aws_access_key_id"] = creds.access_key
            if getattr(creds, "secret_key", None):
                options["aws_secret_access_key"] = creds.secret_key
            if getattr(creds, "token", None):
                options["aws_session_token"] = creds.token

    return options if options else None


class LanceDbShardReader:
    """Connects to remote LanceDB dataset on S3 and searches remotely."""

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
        s3_client: Any | None = None,
    ) -> None:
        lancedb, _ = _import_lancedb()
        self._db_url = db_url
        self._table_name = "vector_index"

        # Determine S3 endpoint options if we're using a custom s3_client
        # In actual prod we should configure LanceDB storage options
        # based on shardyfusion config, but for simplicity here we pass
        # the url. LanceDB uses `object_store` which picks up AWS_ params.

        # We connect directly to the db_url (assuming it's s3://...)
        storage_options = _extract_storage_options(s3_client)
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

    def __init__(self, s3_client: Any | None = None) -> None:
        self._s3_client = s3_client

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
    ) -> LanceDbShardReader:
        return LanceDbShardReader(
            db_url=db_url,
            local_dir=local_dir,
            index_config=index_config,
            s3_client=self._s3_client,
        )
