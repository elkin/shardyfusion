"""SQLite schema types and inference helpers.

Provides lightweight dataclasses for describing SQLite table schemas and
utilities for inferring schemas from DataFrames, Arrow tables, or explicit
column definitions.  Schemas are serializable to/from plain dicts for
storage in manifest ``custom_manifest_fields``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Schema types
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ColumnDef:
    """One column in a SQLite table."""

    name: str
    type: str = "BLOB"
    primary_key: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "type": self.type}
        if self.primary_key:
            d["primary_key"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ColumnDef:
        return cls(
            name=d["name"],
            type=d.get("type", "BLOB"),
            primary_key=d.get("primary_key", False),
        )


@dataclass(slots=True, frozen=True)
class SqliteSchema:
    """Full schema for a SQLite shard table."""

    table_name: str
    columns: tuple[ColumnDef, ...]
    indexes: tuple[tuple[str, ...], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "table_name": self.table_name,
            "columns": [c.to_dict() for c in self.columns],
            "indexes": [list(idx) for idx in self.indexes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SqliteSchema:
        return cls(
            table_name=d["table_name"],
            columns=tuple(ColumnDef.from_dict(c) for c in d["columns"]),
            indexes=tuple(tuple(idx) for idx in d.get("indexes", [])),
        )

    def create_table_sql(self, *, without_rowid: bool = True) -> str:
        """Return a ``CREATE TABLE`` statement for this schema."""
        parts: list[str] = []
        pk_cols: list[str] = []
        for col in self.columns:
            parts.append(f"{_quote_ident(col.name)} {col.type}")
            if col.primary_key:
                pk_cols.append(_quote_ident(col.name))

        body = ", ".join(parts)
        if pk_cols:
            body += f", PRIMARY KEY ({', '.join(pk_cols)})"

        suffix = " WITHOUT ROWID" if without_rowid and pk_cols else ""
        return f"CREATE TABLE {_quote_ident(self.table_name)} ({body}){suffix}"

    def create_index_sqls(self) -> list[str]:
        """Return ``CREATE INDEX`` statements for secondary indexes."""
        stmts: list[str] = []
        for i, idx_cols in enumerate(self.indexes):
            cols_str = ", ".join(_quote_ident(c) for c in idx_cols)
            idx_name = f"idx_{self.table_name}_{i}"
            stmts.append(
                f"CREATE INDEX {_quote_ident(idx_name)} "
                f"ON {_quote_ident(self.table_name)} ({cols_str})"
            )
        return stmts

    def insert_sql(self) -> str:
        """Return a parameterized ``INSERT`` statement."""
        col_names = ", ".join(_quote_ident(c.name) for c in self.columns)
        placeholders = ", ".join("?" for _ in self.columns)
        return f"INSERT OR REPLACE INTO {_quote_ident(self.table_name)} ({col_names}) VALUES ({placeholders})"


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

# Mapping from common type systems to SQLite type affinity.
_ARROW_TO_SQLITE: dict[str, str] = {
    "int8": "INTEGER",
    "int16": "INTEGER",
    "int32": "INTEGER",
    "int64": "INTEGER",
    "uint8": "INTEGER",
    "uint16": "INTEGER",
    "uint32": "INTEGER",
    "uint64": "INTEGER",
    "float16": "REAL",
    "float32": "REAL",
    "float64": "REAL",
    "double": "REAL",
    "bool": "INTEGER",
    "string": "TEXT",
    "large_string": "TEXT",
    "utf8": "TEXT",
    "large_utf8": "TEXT",
    "binary": "BLOB",
    "large_binary": "BLOB",
}

_PANDAS_TO_SQLITE: dict[str, str] = {
    "int64": "INTEGER",
    "int32": "INTEGER",
    "int16": "INTEGER",
    "int8": "INTEGER",
    "uint64": "INTEGER",
    "uint32": "INTEGER",
    "uint16": "INTEGER",
    "uint8": "INTEGER",
    "float64": "REAL",
    "float32": "REAL",
    "float16": "REAL",
    "bool": "INTEGER",
    "object": "TEXT",
    "string": "TEXT",
    "bytes": "BLOB",
}


def infer_schema_from_arrow(
    arrow_schema: Any,
    *,
    table_name: str = "data",
    key_col: str,
    indexes: list[list[str]] | None = None,
) -> SqliteSchema:
    """Infer a :class:`SqliteSchema` from a PyArrow schema object."""
    columns: list[ColumnDef] = []
    for f in arrow_schema:
        type_str = str(f.type)
        sqlite_type = _ARROW_TO_SQLITE.get(type_str, "BLOB")
        columns.append(
            ColumnDef(name=f.name, type=sqlite_type, primary_key=(f.name == key_col))
        )
    return SqliteSchema(
        table_name=table_name,
        columns=tuple(columns),
        indexes=tuple(tuple(idx) for idx in (indexes or [])),
    )


def infer_schema_from_pandas(
    df: Any,
    *,
    table_name: str = "data",
    key_col: str,
    indexes: list[list[str]] | None = None,
) -> SqliteSchema:
    """Infer a :class:`SqliteSchema` from a pandas DataFrame."""
    columns: list[ColumnDef] = []
    for col_name, dtype in df.dtypes.items():
        dtype_str = str(dtype)
        sqlite_type = _PANDAS_TO_SQLITE.get(dtype_str, "TEXT")
        columns.append(
            ColumnDef(
                name=str(col_name),
                type=sqlite_type,
                primary_key=(str(col_name) == key_col),
            )
        )
    return SqliteSchema(
        table_name=table_name,
        columns=tuple(columns),
        indexes=tuple(tuple(idx) for idx in (indexes or [])),
    )


def schema_from_columns(
    columns: list[tuple[str, str]],
    *,
    table_name: str = "data",
    key_col: str,
    indexes: list[list[str]] | None = None,
) -> SqliteSchema:
    """Build a :class:`SqliteSchema` from explicit ``(name, type)`` pairs."""
    return SqliteSchema(
        table_name=table_name,
        columns=tuple(
            ColumnDef(name=name, type=typ, primary_key=(name == key_col))
            for name, typ in columns
        ),
        indexes=tuple(tuple(idx) for idx in (indexes or [])),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quote_ident(name: str) -> str:
    """Quote a SQLite identifier to prevent SQL injection."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'
