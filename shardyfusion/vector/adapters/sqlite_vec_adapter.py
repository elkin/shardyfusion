"""Vector-only sqlite-vec adapter.

Thin shim over :mod:`shardyfusion.sqlite_vec_adapter` (the unified KV+vector
adapter) that adapts its factory signatures to the vector-only
:class:`VectorShardReaderFactory` / :class:`VectorIndexWriterFactory`
protocols.

The on-disk layout is the same (``shard.db`` containing ``vec_index``,
``vec_id_map``, ``vec_payloads`` tables); only the kwargs and the
``VectorSpec`` construction differ.

Requires the ``vector-sqlite`` extra (``sqlite-vec``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ...credentials import CredentialProvider
from ...errors import ConfigValidationError
from ...type_defs import S3ConnectionOptions
from ..types import SearchResult

if TYPE_CHECKING:
    from ...sqlite_vec_adapter import SqliteVecAdapter, SqliteVecShardReader
    from ..config import VectorIndexConfig


_BACKEND_NAME = "sqlite-vec"
_INSTALL_HINT = (
    "sqlite-vec is required for the vector-only sqlite-vec backend. "
    "Install with: pip install 'shardyfusion[vector-sqlite]'"
)


def _build_vector_spec(index_config: VectorIndexConfig) -> Any:
    """Construct a :class:`VectorSpec` from an :class:`VectorIndexConfig`.

    The unified adapter internally requires a ``VectorSpec`` (because it
    also supports KV writes).  For vector-only users we synthesize one
    with no ``vector_col`` — the KV path is unused.
    """
    from ...config import VectorSpec

    return VectorSpec(
        dim=index_config.dim,
        metric=index_config.metric.value,
        index_type=index_config.index_type,
        index_params=dict(index_config.index_params),
        quantization=index_config.quantization,
    )


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SqliteVecVectorWriterFactory:
    """Vector-only writer factory backed by sqlite-vec.

    Constructs a :class:`VectorSpec` on the fly from the
    :class:`VectorIndexConfig` passed to ``__call__``, so vector-only
    users never have to import the unified :class:`VectorSpec` type.
    """

    page_size: int = 4096
    cache_size_pages: int = -2000
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None
    backend_name: str = _BACKEND_NAME

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: VectorIndexConfig,
    ) -> SqliteVecAdapter:
        try:
            from ...sqlite_vec_adapter import SqliteVecAdapter
        except ImportError as exc:  # pragma: no cover - defensive
            raise ConfigValidationError(_INSTALL_HINT) from exc

        return SqliteVecAdapter(
            db_url=db_url,
            local_dir=local_dir,
            vector_spec=_build_vector_spec(index_config),
            page_size=self.page_size,
            cache_size_pages=self.cache_size_pages,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
        )


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class SqliteVecVectorShardReader:
    """Vector-only shard reader backed by the unified sqlite-vec reader.

    Thin adapter that delegates ``search()`` to
    :class:`shardyfusion.sqlite_vec_adapter.SqliteVecShardReader` and
    exposes the :class:`VectorShardReader` protocol (``search`` + ``close``).

    Note: ``index_config`` is accepted for
    :class:`VectorShardReaderFactory` protocol parity (distance metric and
    dimensionality are already baked into the serialized ``shard.db``), so
    no per-call configuration is needed.  Similarly, the ``ef`` argument to
    ``search()`` is ignored — sqlite-vec uses a brute-force scan and has no
    equivalent tunable.
    """

    def __init__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: VectorIndexConfig,  # noqa: ARG002 (kept for protocol parity)
        mmap_size: int = 268_435_456,
        s3_connection_options: S3ConnectionOptions | None = None,
        credential_provider: CredentialProvider | None = None,
    ) -> None:
        try:
            from ...sqlite_vec_adapter import SqliteVecShardReader
        except ImportError as exc:  # pragma: no cover - defensive
            raise ConfigValidationError(_INSTALL_HINT) from exc

        self._inner: SqliteVecShardReader = SqliteVecShardReader(
            db_url=db_url,
            local_dir=local_dir,
            checkpoint_id=None,
            mmap_size=mmap_size,
            s3_connection_options=s3_connection_options,
            credential_provider=credential_provider,
        )

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        ef: int = 50,  # noqa: ARG002 (sqlite-vec ignores ef)
    ) -> list[SearchResult]:
        return self._inner.search(query, top_k)

    def close(self) -> None:
        self._inner.close()

    def __enter__(self) -> SqliteVecVectorShardReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(slots=True)
class SqliteVecVectorReaderFactory:
    """Vector-only reader factory backed by sqlite-vec."""

    mmap_size: int = 268_435_456
    s3_connection_options: S3ConnectionOptions | None = None
    credential_provider: CredentialProvider | None = None
    backend_name: str = _BACKEND_NAME

    def __call__(
        self,
        *,
        db_url: str,
        local_dir: Path,
        index_config: VectorIndexConfig,
    ) -> SqliteVecVectorShardReader:
        return SqliteVecVectorShardReader(
            db_url=db_url,
            local_dir=local_dir,
            index_config=index_config,
            mmap_size=self.mmap_size,
            s3_connection_options=self.s3_connection_options,
            credential_provider=self.credential_provider,
        )


__all__ = [
    "SqliteVecVectorReaderFactory",
    "SqliteVecVectorShardReader",
    "SqliteVecVectorWriterFactory",
]
