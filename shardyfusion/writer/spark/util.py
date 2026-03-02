import logging
import types
from typing import Self

from pyspark import StorageLevel
from pyspark.sql import DataFrame, SparkSession

from shardyfusion.logging import log_event


class DataFrameCacheContext:
    """Cache a DataFrame for the lifetime of the context and unpersist on exit."""

    def __init__(
        self,
        df: DataFrame,
        storage_level: StorageLevel | None = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._df = df
        self._storage_level = storage_level
        self._enabled = enabled
        self._cached_df: DataFrame | None = None

    def __enter__(self) -> DataFrame:
        if not self._enabled:
            return self._df
        try:
            if self._storage_level is None:
                self._cached_df = self._df.persist()
            else:
                self._cached_df = self._df.persist(self._storage_level)
        except Exception as exc:  # pragma: no cover - Spark environment dependent
            log_event(
                "dataframe_cache_failed",
                level=logging.WARNING,
                error=str(exc),
            )
            self._cached_df = self._df
        return self._cached_df

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        if not self._enabled or self._cached_df is None:
            return
        try:
            self._cached_df.unpersist(blocking=False)
        except Exception as unpersist_exc:  # pragma: no cover - Spark env dependent
            log_event(
                "dataframe_unpersist_failed",
                level=logging.WARNING,
                error=str(unpersist_exc),
            )


class SparkConfOverrideContext:
    """Temporarily override Spark configuration values and restore them on exit."""

    def __init__(self, spark: SparkSession, overrides: dict[str, str] | None) -> None:
        self._spark = spark
        self._overrides = dict(overrides or {})
        self._original_values: dict[str, str | None] = {}

    def __enter__(self) -> Self:
        for key, value in self._overrides.items():
            self._original_values[key] = self._get_conf_or_none(key)
            try:
                self._spark.conf.set(key, value)
            except Exception as exc:  # pragma: no cover - Spark environment dependent
                log_event(
                    "spark_conf_override_failed",
                    level=logging.WARNING,
                    key=key,
                    value=value,
                    error=str(exc),
                )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        for key in reversed(list(self._overrides.keys())):
            original = self._original_values.get(key)
            try:
                if original is None:
                    self._unset_conf_if_supported(key)
                else:
                    self._spark.conf.set(key, original)
            except (
                Exception
            ) as restore_exc:  # pragma: no cover - Spark environment dependent
                log_event(
                    "spark_conf_restore_failed",
                    level=logging.WARNING,
                    key=key,
                    original_value=original,
                    error=str(restore_exc),
                )

    def _get_conf_or_none(self, key: str) -> str | None:
        try:
            return self._spark.conf.get(key, None)
        except Exception:  # pragma: no cover - Spark environment dependent
            log_event(
                "spark_conf_get_failed",
                level=logging.WARNING,
                key=key,
            )
            return None

    def _unset_conf_if_supported(self, key: str) -> None:
        self._spark.conf.unset(key)
