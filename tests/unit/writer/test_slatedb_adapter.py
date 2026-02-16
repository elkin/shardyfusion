from __future__ import annotations

import sys
import types

import pytest

from slatedb_spark_sharded.errors import SlateDbApiError
from slatedb_spark_sharded.slatedb_adapter import DefaultSlateDbAdapter


def test_open_uses_official_constructor_signature(monkeypatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    sentinel = object()

    def fake_slatedb_ctor(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    fake_module = types.ModuleType("slatedb")
    fake_module.SlateDB = fake_slatedb_ctor
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    adapter = DefaultSlateDbAdapter()
    result = adapter.open(
        local_dir="/tmp/local",
        db_url="s3://bucket/path",
        env_file="slatedb.env",
        settings={"durability": "strict"},
    )

    assert result is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("/tmp/local",)
    assert kwargs["url"] == "s3://bucket/path"
    assert kwargs["env_file"] == "slatedb.env"
    assert kwargs["settings"] == '{"durability":"strict"}'


def test_open_raises_when_binding_signature_is_not_official(monkeypatch) -> None:
    def fake_slatedb_ctor(*args, **kwargs):
        _ = (args, kwargs)
        raise TypeError("unexpected kwargs")

    fake_module = types.ModuleType("slatedb")
    fake_module.SlateDB = fake_slatedb_ctor
    monkeypatch.setitem(sys.modules, "slatedb", fake_module)

    adapter = DefaultSlateDbAdapter()
    with pytest.raises(
        SlateDbApiError,
        match="official Python binding signature",
    ):
        adapter.open(
            local_dir="/tmp/local",
            db_url="s3://bucket/path",
            env_file=None,
            settings={"durability": "strict"},
        )
