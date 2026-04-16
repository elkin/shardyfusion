"""USearch-based vector index adapter with SQLite payload sidecar.

usearch is imported lazily so the module can be imported without the
``vector-usearch`` extra installed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from ...errors import VectorIndexError, VectorSearchError
from ...logging import get_logger, log_event
from ...storage import get_bytes, put_bytes
from ..types import (
    DistanceMetric,
    SearchResult,
)

_logger = get_logger(__name__)

INDEX_FILENAME = "index.usearch"
PAYLOADS_FILENAME = "payloads.db"

_USEARCH_METRIC_MAP = {
    DistanceMetric.COSINE: "cos",
    DistanceMetric.L2: "l2sq",
    DistanceMetric.DOT_PRODUCT: "ip",
}

_USEARCH_DTYPE_MAP = {
    None: "f32",
    "fp16": "f16",
    "i8": "i8",
}


def _import_usearch() -> Any:
    try:
        from usearch import index as usearch_index  # type: ignore[import-untyped]
    except ImportError as exc:
        raise VectorIndexError(
            "usearch is required for USearch adapter. "
            "Install with: pip install 'shardyfusion[vector-usearch]'"
        ) from exc
    return usearch_index


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class USearchWriter:
    """Builds a usearch HNSW index + SQLite payload sidecar locally."""

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
        usearch_index = _import_usearch()
        self._db_url = db_url
        self._local_dir = local_dir
        self._s3_client = s3_client
        self._local_dir.mkdir(parents=True, exist_ok=True)

        params = index_params or {}
        metric_str = _USEARCH_METRIC_MAP.get(metric, "cos")
        dtype_str = _USEARCH_DTYPE_MAP.get(quantization, "f32")
        if quantization is not None and quantization not in _USEARCH_DTYPE_MAP:
            raise VectorIndexError(
                f"Unsupported quantization '{quantization}' for USearch adapter. "
                f"Supported: {list(_USEARCH_DTYPE_MAP.keys())}"
            )

        self._index = usearch_index.Index(
            ndim=dim,
            metric=metric_str,
            dtype=dtype_str,
            connectivity=params.get("M", 16),
            expansion_add=params.get("ef_construction", 128),
            expansion_search=params.get("ef_search", 64),
        )
        self._index_path = self._local_dir / INDEX_FILENAME
        self._payloads_path = self._local_dir / PAYLOADS_FILENAME

        self._payload_conn = sqlite3.connect(str(self._payloads_path))
        self._payload_conn.execute(
            "CREATE TABLE IF NOT EXISTS payloads "
            "(id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
        )
        # Map internal int64 keys → original string IDs (only for non-int IDs)
        self._payload_conn.execute(
            "CREATE TABLE IF NOT EXISTS id_map "
            "(internal_id INTEGER PRIMARY KEY, original_id TEXT NOT NULL)"
        )
        self._payload_conn.execute("PRAGMA journal_mode=WAL")
        self._closed = False
        self._count = 0
        self._next_internal_id = 0

    def add_batch(
        self,
        ids: np.ndarray,
        vectors: np.ndarray,
        payloads: list[dict[str, Any]] | None = None,
    ) -> None:
        if self._closed:
            raise VectorIndexError("Writer is closed")

        # USearch only supports int64 keys internally.  We always use
        # monotonically increasing synthetic internal IDs to prevent
        # collisions between batches (regardless of whether caller IDs
        # are ints or strings), and store the original IDs in id_map to
        # preserve type fidelity on read.
        n = len(ids)
        start = self._next_internal_id
        internal_ids = np.arange(start, start + n, dtype=np.int64)
        self._next_internal_id = start + n
        normalized_ids: list[int | str] = [
            int(raw_id) if isinstance(raw_id, (int, np.integer)) else str(raw_id)
            for raw_id in ids
        ]

        # Store the mapping from internal → original, preserving the
        # original type representation (int stays as str(int) in SQLite
        # TEXT, but we tag ints so the reader can reconstruct them).
        id_rows = [
            (
                int(internal_ids[i]),
                json.dumps({"v": normalized_ids[i], "t": "int"})
                if isinstance(normalized_ids[i], int)
                else json.dumps({"v": normalized_ids[i], "t": "str"}),
            )
            for i in range(n)
        ]
        self._payload_conn.executemany(
            "INSERT OR REPLACE INTO id_map (internal_id, original_id) VALUES (?, ?)",
            id_rows,
        )

        try:
            self._index.add(internal_ids, vectors.astype(np.float32))
        except Exception as exc:
            raise VectorIndexError(f"Failed to add vectors to index: {exc}") from exc

        if payloads is not None:
            rows = [
                (str(int(internal_ids[i])), json.dumps(payloads[i], default=str))
                for i in range(len(ids))
            ]
            self._payload_conn.executemany(
                "INSERT OR REPLACE INTO payloads (id, payload) VALUES (?, ?)",
                rows,
            )

        self._count += len(ids)

    def flush(self) -> None:
        if self._closed:
            return
        self._index.save(str(self._index_path))
        self._payload_conn.commit()

    def checkpoint(self) -> str | None:
        self.flush()
        if not self._index_path.exists():
            return None
        h = hashlib.sha256(self._index_path.read_bytes()).hexdigest()
        return h[:16]

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._payload_conn.close()

        # Upload to S3
        index_url = f"{self._db_url}/{INDEX_FILENAME}"
        payloads_url = f"{self._db_url}/{PAYLOADS_FILENAME}"

        try:
            put_bytes(
                index_url,
                self._index_path.read_bytes(),
                "application/octet-stream",
                s3_client=self._s3_client,
            )
            put_bytes(
                payloads_url,
                self._payloads_path.read_bytes(),
                "application/x-sqlite3",
                s3_client=self._s3_client,
            )
        except Exception as exc:
            raise VectorIndexError(
                f"Failed to upload index/payloads to S3: {exc}"
            ) from exc

        self._closed = True

        log_event(
            "usearch_shard_uploaded",
            logger=_logger,
            db_url=self._db_url,
            vector_count=self._count,
        )

    def __enter__(self) -> USearchWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class USearchWriterFactory:
    """Factory that creates USearchWriter instances."""

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
    ) -> USearchWriter:
        return USearchWriter(
            db_url=db_url,
            local_dir=local_dir,
            dim=index_config.dim,
            metric=index_config.metric,
            quantization=index_config.quantization,
            index_params=index_config.index_params,
            s3_client=self._s3_client,
        )


def _decode_id_map_value(raw: str) -> int | str:
    """Decode an id_map original_id value, handling both new typed JSON
    format (``{"v": ..., "t": "int"|"str"}``) and legacy plain-string format.
    """
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "v" in obj:
            if obj.get("t") == "int":
                return int(obj["v"])
            return str(obj["v"])
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # Legacy format: plain string
    return raw


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class USearchShardReader:
    """Downloads index + payloads from S3, searches locally."""

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
        s3_client: Any | None = None,
    ) -> None:
        usearch_index = _import_usearch()
        self._db_url = db_url
        self._local_dir = local_dir
        self._local_dir.mkdir(parents=True, exist_ok=True)

        index_path = self._local_dir / INDEX_FILENAME
        payloads_path = self._local_dir / PAYLOADS_FILENAME

        # Download from S3
        index_url = f"{db_url}/{INDEX_FILENAME}"
        payloads_url = f"{db_url}/{PAYLOADS_FILENAME}"

        index_bytes = get_bytes(index_url, s3_client=s3_client)
        index_path.write_bytes(index_bytes)

        payloads_bytes = get_bytes(payloads_url, s3_client=s3_client)
        payloads_path.write_bytes(payloads_bytes)

        # Load index
        metric_str = _USEARCH_METRIC_MAP.get(index_config.metric, "cos")
        dtype_str = _USEARCH_DTYPE_MAP.get(index_config.quantization, "f32")
        self._index = usearch_index.Index(
            ndim=index_config.dim,
            metric=metric_str,
            dtype=dtype_str,
        )
        self._index.load(str(index_path))

        # Open payloads DB
        self._payload_conn = sqlite3.connect(
            str(payloads_path), check_same_thread=False
        )
        self._closed = False

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        ef: int = 50,
    ) -> list[SearchResult]:
        if self._closed:
            raise VectorSearchError("Reader is closed")
        try:
            matches = self._index.search(query.astype(np.float32), top_k, exact=False)
        except Exception as exc:
            raise VectorSearchError(f"Search failed: {exc}") from exc

        results: list[SearchResult] = []
        keys = matches.keys
        distances = matches.distances

        internal_ids = [int(k) for k in keys]

        # Resolve internal int64 keys → original IDs via id_map (if present)
        id_map: dict[int, int | str] = {}
        if internal_ids:
            placeholders = ",".join("?" for _ in internal_ids)
            try:
                cursor = self._payload_conn.execute(
                    f"SELECT internal_id, original_id FROM id_map WHERE internal_id IN ({placeholders})",  # noqa: S608
                    internal_ids,
                )
                for row in cursor:
                    id_map[int(row[0])] = _decode_id_map_value(row[1])
            except sqlite3.OperationalError:
                pass  # id_map table doesn't exist (old index format)

        # Look up payloads by the internal IDs to avoid collisions between
        # distinct original IDs like ``1`` and ``"1"``.
        original_ids: list[int | str] = [id_map.get(iid, iid) for iid in internal_ids]
        payload_map: dict[str, dict[str, Any]] = {}
        str_iids = [str(iid) for iid in internal_ids]
        if str_iids:
            placeholders = ",".join("?" for _ in str_iids)
            cursor = self._payload_conn.execute(
                f"SELECT id, payload FROM payloads WHERE id IN ({placeholders})",  # noqa: S608
                str_iids,
            )
            for row in cursor:
                payload_map[row[0]] = json.loads(row[1])

        for i in range(len(keys)):
            orig_id = original_ids[i]
            payload = payload_map.get(str(internal_ids[i]))
            results.append(
                SearchResult(
                    id=orig_id,
                    score=float(distances[i]),
                    payload=payload,
                )
            )

        return results

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._payload_conn.close()


class USearchReaderFactory:
    """Factory that creates USearchShardReader instances."""

    def __init__(self, s3_client: Any | None = None) -> None:
        self._s3_client = s3_client

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: Any,
    ) -> USearchShardReader:
        return USearchShardReader(
            db_url=db_url,
            local_dir=local_dir,
            index_config=index_config,
            s3_client=self._s3_client,
        )
