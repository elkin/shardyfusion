"""Tests for Dask vector sharding."""

from __future__ import annotations

import numpy as np
import pytest

from shardyfusion.vector._distributed import ResolvedVectorRouting
from shardyfusion.vector.types import DistanceMetric, VectorShardingStrategy

pytest.importorskip("dask", reason="requires writer-dask extra")

import dask.dataframe as dd
import pandas as pd

from shardyfusion.writer.dask.sharding import (  # noqa: E402
    VECTOR_DB_ID_COL,
    add_vector_db_id_column,
)


class TestAddVectorDbIdColumnDask:
    def test_explicit_adds_column(self) -> None:
        """EXPLICIT: uses existing shard_id column."""
        pdf = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "shard_id": [0, 1, 2, 3],
            }
        )
        ddf = dd.from_pandas(pdf, npartitions=2)
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        ddf_with_id, num_dbs = add_vector_db_id_column(
            ddf,
            vector_col="embedding",
            routing=routing,
            shard_id_col="shard_id",
        )

        assert VECTOR_DB_ID_COL in ddf_with_id.columns
        result = ddf_with_id.compute()
        ids = result[VECTOR_DB_ID_COL].tolist()
        assert ids == [0, 1, 2, 3]
        assert num_dbs == 4

    def test_explicit_missing_shard_id_col_raises(self) -> None:
        """EXPLICIT: requires shard_id_col."""
        pdf = pd.DataFrame(
            {
                "id": [1],
                "embedding": [[0.1] * 128],
            }
        )
        ddf = dd.from_pandas(pdf, npartitions=1)
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.EXPLICIT,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
        )

        with pytest.raises(AssertionError):
            add_vector_db_id_column(
                ddf,
                vector_col="embedding",
                routing=routing,
            )

    def test_cluster_returns_correct_columns(self) -> None:
        """CLUSTER: returns correct column."""
        centroids = np.random.rand(4, 128).astype(np.float32)
        pdf = pd.DataFrame(
            {
                "id": [1],
                "embedding": [[0.5] * 128],
            }
        )
        ddf = dd.from_pandas(pdf, npartitions=1)
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            centroids=centroids,
        )

        ddf_with_id, num_dbs = add_vector_db_id_column(
            ddf,
            vector_col="embedding",
            routing=routing,
        )

        assert VECTOR_DB_ID_COL in ddf_with_id.columns
        assert num_dbs == 4

    def test_cluster_accepts_stringified_vectors(self) -> None:
        """CLUSTER: handles vectors serialized to strings by Dask ops."""
        centroids = np.random.rand(2, 4).astype(np.float32)
        pdf = pd.DataFrame(
            {
                "id": [1, 2],
                "embedding": ["[0.1, 0.2, 0.3, 0.4]", "[0.4, 0.3, 0.2, 0.1]"],
            }
        )
        ddf = dd.from_pandas(pdf, npartitions=1)
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.CLUSTER,
            num_dbs=2,
            metric=DistanceMetric.COSINE,
            centroids=centroids,
        )

        ddf_with_id, num_dbs = add_vector_db_id_column(
            ddf,
            vector_col="embedding",
            routing=routing,
        )

        result = ddf_with_id.compute()
        ids = result[VECTOR_DB_ID_COL].tolist()
        assert num_dbs == 2
        assert all(db_id in {0, 1} for db_id in ids)

    def test_lsh_returns_correct_columns(self) -> None:
        """LSH: returns correct column."""
        hyperplanes = np.random.rand(8, 128).astype(np.float32)
        pdf = pd.DataFrame(
            {
                "id": [1],
                "embedding": [[0.5] * 128],
            }
        )
        ddf = dd.from_pandas(pdf, npartitions=1)
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.LSH,
            num_dbs=8,
            metric=DistanceMetric.COSINE,
            hyperplanes=hyperplanes,
        )

        ddf_with_id, num_dbs = add_vector_db_id_column(
            ddf,
            vector_col="embedding",
            routing=routing,
        )

        assert VECTOR_DB_ID_COL in ddf_with_id.columns
        assert num_dbs == 8

    def test_lsh_accepts_stringified_vectors(self) -> None:
        """LSH: handles vectors serialized to strings by Dask ops."""
        hyperplanes = np.random.rand(2, 4).astype(np.float32)
        pdf = pd.DataFrame(
            {
                "id": [1, 2],
                "embedding": ["[0.1, 0.2, 0.3, 0.4]", "[0.4, 0.3, 0.2, 0.1]"],
            }
        )
        ddf = dd.from_pandas(pdf, npartitions=1)
        routing = ResolvedVectorRouting(
            strategy=VectorShardingStrategy.LSH,
            num_dbs=4,
            metric=DistanceMetric.COSINE,
            hyperplanes=hyperplanes,
        )

        ddf_with_id, num_dbs = add_vector_db_id_column(
            ddf,
            vector_col="embedding",
            routing=routing,
        )

        result = ddf_with_id.compute()
        ids = result[VECTOR_DB_ID_COL].tolist()
        assert num_dbs == 4
        assert all(0 <= db_id < 4 for db_id in ids)
