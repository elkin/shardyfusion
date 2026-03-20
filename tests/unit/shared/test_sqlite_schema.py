"""Unit tests for sqlite_schema module."""

from shardyfusion.sqlite_schema import (
    ColumnDef,
    SqliteSchema,
    schema_from_columns,
)


class TestColumnDef:
    def test_round_trip_dict(self) -> None:
        col = ColumnDef(name="user_id", type="INTEGER", primary_key=True)
        d = col.to_dict()
        restored = ColumnDef.from_dict(d)
        assert restored == col

    def test_defaults(self) -> None:
        col = ColumnDef(name="data")
        assert col.type == "BLOB"
        assert col.primary_key is False


class TestSqliteSchema:
    def test_create_table_sql(self) -> None:
        schema = SqliteSchema(
            table_name="users",
            columns=(
                ColumnDef(name="id", type="INTEGER", primary_key=True),
                ColumnDef(name="name", type="TEXT"),
            ),
        )
        sql = schema.create_table_sql()
        assert "CREATE TABLE" in sql
        assert '"users"' in sql
        assert "WITHOUT ROWID" in sql
        assert "PRIMARY KEY" in sql

    def test_create_table_no_pk_no_without_rowid(self) -> None:
        schema = SqliteSchema(
            table_name="t",
            columns=(ColumnDef(name="a", type="TEXT"),),
        )
        sql = schema.create_table_sql()
        assert "WITHOUT ROWID" not in sql

    def test_create_index_sqls(self) -> None:
        schema = SqliteSchema(
            table_name="users",
            columns=(
                ColumnDef(name="id", type="INTEGER", primary_key=True),
                ColumnDef(name="email", type="TEXT"),
            ),
            indexes=(("email",),),
        )
        sqls = schema.create_index_sqls()
        assert len(sqls) == 1
        assert "CREATE INDEX" in sqls[0]
        assert '"email"' in sqls[0]

    def test_insert_sql(self) -> None:
        schema = SqliteSchema(
            table_name="t",
            columns=(
                ColumnDef(name="a", type="INTEGER"),
                ColumnDef(name="b", type="TEXT"),
            ),
        )
        sql = schema.insert_sql()
        assert "INSERT OR REPLACE" in sql
        assert "?, ?" in sql

    def test_round_trip_dict(self) -> None:
        schema = SqliteSchema(
            table_name="data",
            columns=(
                ColumnDef(name="k", type="BLOB", primary_key=True),
                ColumnDef(name="v", type="TEXT"),
            ),
            indexes=(("v",),),
        )
        d = schema.to_dict()
        restored = SqliteSchema.from_dict(d)
        assert restored == schema

    def test_quote_ident_escapes_quotes(self) -> None:
        schema = SqliteSchema(
            table_name='my"table',
            columns=(ColumnDef(name="id", type="INTEGER", primary_key=True),),
        )
        sql = schema.create_table_sql()
        assert '"my""table"' in sql


class TestSchemaFromColumns:
    def test_basic(self) -> None:
        schema = schema_from_columns(
            [("id", "INTEGER"), ("name", "TEXT")],
            table_name="users",
            key_col="id",
        )
        assert schema.table_name == "users"
        assert len(schema.columns) == 2
        assert schema.columns[0].primary_key is True
        assert schema.columns[1].primary_key is False

    def test_with_indexes(self) -> None:
        schema = schema_from_columns(
            [("id", "INTEGER"), ("email", "TEXT")],
            table_name="users",
            key_col="id",
            indexes=[["email"]],
        )
        assert schema.indexes == (("email",),)
