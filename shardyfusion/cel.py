"""CEL (Common Expression Language) integration for flexible sharding.

Provides compile/evaluate/routing functions shared by all writer backends
and the reader routing layer. Requires the ``cel`` extra:
``pip install shardyfusion[cel]``.

CEL routing supports three resolver modes:

- direct integer mode: the expression returns the dense internal shard id
- categorical mode: the expression returns a discrete token and
  ``routing_values`` resolves the shard id by exact match

A built-in ``shard_hash()`` function wrapping xxh3_64 is registered
automatically.

Uses ``cel-expr-python`` (Google's C++ CEL wrapper) via:
    ``from cel_expr_python.cel import NewEnv, Type``
"""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import ConfigValidationError
from .sharding_types import CelShardingSpec, RoutingValue

# ---------------------------------------------------------------------------
# Lazy imports — fail fast with a clear message
# ---------------------------------------------------------------------------

_CEL_IMPORT_ERROR = (
    "CEL sharding requires the 'cel' extra. "
    "Install it with: pip install shardyfusion[cel]"
)


def _import_cel_module() -> Any:
    try:
        from cel_expr_python import cel  # type: ignore[import-not-found]

        return cel
    except ImportError as exc:
        raise ImportError(_CEL_IMPORT_ERROR) from exc


# ---------------------------------------------------------------------------
# CEL type mapping
# ---------------------------------------------------------------------------

# Maps user-declared column types to CEL Type attribute names.
CEL_TYPE_MAP: dict[str, str] = {
    "int": "INT",
    "string": "STRING",
    "bytes": "BYTES",
    "double": "DOUBLE",
    "bool": "BOOL",
    "uint": "UINT",
}


# ---------------------------------------------------------------------------
# shard_hash() custom function
# ---------------------------------------------------------------------------

_SHARD_HASH_EXTENSION: Any = None


def _shard_hash_impl(x: Any) -> int:
    """Compute xxh3_64 hash of a value — used as CEL custom function."""
    return _xxh3_digest(x)


# Lazy-bound reference to routing.xxh3_digest — avoids per-call import overhead.
_xxh3_digest: Any = None


def _ensure_xxh3_digest() -> None:
    global _xxh3_digest
    if _xxh3_digest is None:
        from .routing import xxh3_digest

        _xxh3_digest = xxh3_digest


def _get_shard_hash_extension() -> Any:
    """Lazily build and cache the CelExtension with shard_hash() overloads."""
    global _SHARD_HASH_EXTENSION
    if _SHARD_HASH_EXTENSION is not None:
        return _SHARD_HASH_EXTENSION

    _ensure_xxh3_digest()

    cel = _import_cel_module()
    _SHARD_HASH_EXTENSION = cel.CelExtension(
        "shardyfusion",
        [
            cel.FunctionDecl(
                "shard_hash",
                [
                    cel.Overload(
                        "shard_hash_int",
                        cel.Type.UINT,
                        [cel.Type.INT],
                        impl=_shard_hash_impl,
                    ),
                    cel.Overload(
                        "shard_hash_uint",
                        cel.Type.UINT,
                        [cel.Type.UINT],
                        impl=_shard_hash_impl,
                    ),
                    cel.Overload(
                        "shard_hash_string",
                        cel.Type.UINT,
                        [cel.Type.STRING],
                        impl=_shard_hash_impl,
                    ),
                    cel.Overload(
                        "shard_hash_bytes",
                        cel.Type.UINT,
                        [cel.Type.BYTES],
                        impl=_shard_hash_impl,
                    ),
                ],
            )
        ],
    )
    return _SHARD_HASH_EXTENSION


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


class CompiledCel:
    """Wrapper around a compiled CEL expression with its environment."""

    __slots__ = ("_env", "_expr", "_columns")

    def __init__(self, env: Any, expr: Any, columns: dict[str, str]) -> None:
        self._env = env
        self._expr = expr
        self._columns = columns

    @property
    def columns(self) -> dict[str, str]:
        return self._columns

    def evaluate(self, context: dict[str, Any]) -> int | str | bytes:
        """Evaluate the CEL expression with the given variable bindings.

        Uses ``cel_expr_python``'s ``Expression.eval(data=...)`` method
        (a sandboxed CEL evaluator, NOT Python's built-in eval).
        """
        # S307: This is cel_expr_python's sandboxed evaluator, not Python's eval()
        cel_result = self._expr.eval(data=context)  # noqa: S307
        value = cel_result.plain_value()
        if isinstance(value, bytearray):
            value = bytes(value)
        if isinstance(value, bool) or not isinstance(value, (int, str, bytes)):
            raise ConfigValidationError(
                f"CEL expression must return int, str, or bytes; got {type(value).__name__}"
            )
        return value


class UnknownRoutingTokenError(ValueError):
    """Raised when categorical CEL routing produces a token not in the snapshot."""


def compile_cel(expr: str, columns: dict[str, str]) -> CompiledCel:
    """Compile a CEL expression with declared column types.

    The ``shard_hash()`` custom function is automatically registered,
    providing overloads for int, uint, string, and bytes arguments.

    Args:
        expr: CEL expression string (e.g., ``"shard_hash(key) % 100u"``).
        columns: Map of variable name -> type string
            (e.g., ``{"key": "int", "region": "string"}``).

    Returns:
        A compiled expression object that can be evaluated with variable bindings.

    Raises:
        ImportError: If ``cel-expr-python`` is not installed.
        ConfigValidationError: If the expression fails to compile.
    """
    cel = _import_cel_module()

    variables: dict[str, Any] = {}
    for name, type_str in columns.items():
        cel_type_attr = CEL_TYPE_MAP.get(type_str)
        if cel_type_attr is None:
            raise ConfigValidationError(
                f"Unsupported CEL column type: {type_str!r} for column {name!r}. "
                f"Allowed: {sorted(CEL_TYPE_MAP)}"
            )
        variables[name] = getattr(cel.Type, cel_type_attr)

    try:
        ext = _get_shard_hash_extension()
        env = cel.NewEnv(variables=variables, extensions=[ext])
        compiled_expr = env.compile(expr)
    except Exception as exc:
        raise ConfigValidationError(
            f"Failed to compile CEL expression {expr!r}: {exc}"
        ) from exc

    return CompiledCel(env=env, expr=compiled_expr, columns=columns)


@functools.lru_cache(maxsize=16)
def _compile_cel_cached(
    expr: str, columns_key: tuple[tuple[str, str], ...]
) -> CompiledCel:
    """Cached wrapper around compile_cel for hashable arguments."""
    return compile_cel(expr, dict(columns_key))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_cel_arrow_batch(
    compiled: CompiledCel,
    batch: Any,
) -> list[int | str | bytes]:
    """Evaluate CEL on each row of a PyArrow RecordBatch/Table.

    Used by the Spark writer via ``mapInArrow`` (Arrow batches are
    the natural format).  Dask and Ray writers route through
    ``route_key()`` → ``route_cel()`` per row instead.

    Args:
        compiled: Compiled CEL expression.
        batch: PyArrow RecordBatch or Table.

    Returns:
        List of routing key values, one per row.
    """
    col_names = list(compiled.columns)
    return [compiled.evaluate(row) for row in batch.select(col_names).to_pylist()]


def route_cel_batch(
    compiled: CompiledCel,
    batch: Any,
    routing_values: Sequence[RoutingValue] | None = None,
) -> list[int]:
    """Evaluate CEL and route each row to a shard.

    If ``routing_values`` is provided, uses categorical exact-match lookup.
    Otherwise, the CEL output is used directly as the shard ID.
    """
    routing_keys = evaluate_cel_arrow_batch(compiled, batch)
    if routing_values is not None:
        lookup = build_categorical_routing_lookup(routing_values)
        return [
            resolve_cel_routing_key(rk, routing_values=routing_values, lookup=lookup)
            for rk in routing_keys
        ]
    return [resolve_cel_routing_key(rk) for rk in routing_keys]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_cel(
    compiled: CompiledCel,
    context: dict[str, Any],
    routing_values: Sequence[RoutingValue] | None = None,
    lookup: dict[RoutingValue, int] | None = None,
) -> int:
    """Evaluate CEL expression and route to a shard.

    Args:
        compiled: Compiled CEL expression.
        context: Variable bindings for the expression.
        routing_values: Optional categorical routing values. If provided,
            exact-match lookup gives the shard ID.

    Returns:
        Shard db_id (0-based).
    """
    routing_key = compiled.evaluate(context)
    return resolve_cel_routing_key(
        routing_key,
        routing_values=routing_values,
        lookup=lookup,
    )


def build_categorical_routing_lookup(
    routing_values: Sequence[RoutingValue],
) -> dict[RoutingValue, int]:
    """Build exact-match lookup for categorical CEL routing."""

    return {value: idx for idx, value in enumerate(routing_values)}


def resolve_cel_routing_key(
    routing_key: int | str | bytes,
    *,
    routing_values: Sequence[RoutingValue] | None = None,
    lookup: dict[RoutingValue, int] | None = None,
) -> int:
    """Resolve a CEL routing key to a dense internal shard id."""

    if routing_values is not None:
        value_to_db_id = (
            lookup
            if lookup is not None
            else build_categorical_routing_lookup(routing_values)
        )
        try:
            return value_to_db_id[routing_key]
        except KeyError as exc:
            raise UnknownRoutingTokenError(
                f"Categorical CEL routing token {routing_key!r} is not present "
                "in this snapshot."
            ) from exc
    return int(routing_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CelType(str, Enum):
    """Supported CEL column types for :func:`cel_sharding_by_columns`."""

    INT = "int"
    STRING = "string"
    BYTES = "bytes"
    DOUBLE = "double"
    BOOL = "bool"
    UINT = "uint"


@dataclass(frozen=True, slots=True)
class CelColumn:
    """Column specification for :func:`cel_sharding_by_columns`.

    Args:
        name: Column name as it appears in the DataFrame / record.
        type: CEL type of the column (default :attr:`CelType.STRING`).
    """

    name: str
    type: CelType = CelType.STRING


def cel_sharding(
    expr: str,
    columns: dict[str, str | CelType],
    *,
    routing_values: Sequence[RoutingValue] | None = None,
) -> CelShardingSpec:
    """Build a :class:`CelShardingSpec` for an arbitrary CEL expression.

    This is the general-purpose companion to :func:`cel_sharding_by_columns`.
    Use it when you want raw CEL control but do not want to construct
    ``CelShardingSpec(...)`` manually.

    Direct routing example::

        cel_sharding("key % 4", {"key": "int"})

    Args:
        expr: CEL expression to evaluate for routing.
        columns: Mapping of column name to CEL type string or :class:`CelType`.
        routing_values: Optional categorical routing values for exact-match mode.

    Returns:
        A :class:`CelShardingSpec`.

    Raises:
        ConfigValidationError: If no columns are given or an unsupported CEL type
        is declared.
    """
    if not columns:
        raise ConfigValidationError("cel_sharding requires at least one column")

    normalized_columns: dict[str, str] = {}
    for name, type_value in columns.items():
        normalized_type = (
            type_value.value if isinstance(type_value, CelType) else type_value
        )
        if normalized_type not in CEL_TYPE_MAP:
            raise ConfigValidationError(
                f"Unsupported CEL column type: {normalized_type!r} for column {name!r}. "
                f"Allowed: {sorted(CEL_TYPE_MAP)}"
            )
        normalized_columns[name] = normalized_type

    return CelShardingSpec(
        routing_values=list(routing_values) if routing_values is not None else None,
        cel_expr=expr,
        cel_columns=normalized_columns,
    )


def cel_sharding_by_columns(
    *columns: str | CelColumn,
    num_shards: int | None = None,
    separator: str = ":",
) -> CelShardingSpec:
    """Build a :class:`CelShardingSpec` for CEL-based partitioning by column values.

    Each column is either a bare name (``str``, defaults to
    :attr:`CelType.STRING`) or a :class:`CelColumn` instance.  The generated
    CEL expression either hashes the column value(s) and takes modulo
    ``num_shards`` to produce a 0-based shard ID, or when ``num_shards`` is
    omitted, returns a categorical routing token whose distinct values are
    discovered from data at write time.

    Single column example::

        cel_sharding_by_columns("region", num_shards=10)
        # → cel_expr="shard_hash(region) % 10u"
        #   cel_columns={"region": "string"}

    Typed column example::

        cel_sharding_by_columns(CelColumn("tier", CelType.INT), num_shards=4)
        # → cel_expr="shard_hash(tier) % 4u"
        #   cel_columns={"tier": "int"}

    Multiple columns example::

        cel_sharding_by_columns("region", CelColumn("tier", CelType.INT), num_shards=8)
        # → cel_expr='shard_hash(region + ":" + string(tier)) % 8u'
        #   cel_columns={"region": "string", "tier": "int"}

    Args:
        *columns: One or more column specifications.
        num_shards: Target number of shards (embedded as ``% <N>u``). If omitted,
            categorical routing values are inferred from data at write time.
        separator: Delimiter for multi-column concatenation (default ``":"``).

    Returns:
        A :class:`CelShardingSpec`.

    Raises:
        ConfigValidationError: If no columns given or ``num_shards < 1``.
    """
    if not columns:
        raise ConfigValidationError(
            "cel_sharding_by_columns requires at least one column"
        )
    if num_shards is not None and num_shards < 1:
        raise ConfigValidationError(f"num_shards must be >= 1; got {num_shards}")

    parsed: list[CelColumn] = []
    for col in columns:
        if isinstance(col, str):
            parsed.append(CelColumn(col))
        elif isinstance(col, CelColumn):
            parsed.append(col)
        else:
            raise ConfigValidationError(
                f"Column must be a str or CelColumn; got {col!r}"
            )

    cel_columns: dict[str, str] = {c.name: c.type.value for c in parsed}

    # Build the inner expression (what gets hashed).
    if len(parsed) == 1:
        column = parsed[0]
        # shard_hash() has direct overloads for int/uint/string/bytes only.
        inner = (
            column.name
            if column.type in (CelType.INT, CelType.UINT, CelType.STRING, CelType.BYTES)
            else f"string({column.name})"
        )
    else:
        separator_literal = json.dumps(separator)
        parts: list[str] = []
        for c in parsed:
            if c.type == CelType.STRING:
                parts.append(c.name)
            else:
                parts.append(f"string({c.name})")
        inner = f" + {separator_literal} + ".join(parts)

    if num_shards is None:
        return CelShardingSpec(
            cel_expr=inner,
            cel_columns=cel_columns,
            infer_routing_values_from_data=True,
        )

    cel_expr = f"shard_hash({inner}) % {num_shards}u"
    return cel_sharding(cel_expr, cel_columns)


def pandas_rows_to_contexts(
    pdf: Any,
    cel_columns: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert pandas DataFrame rows to CEL context dicts.

    Handles numpy scalar coercion. Shared by Spark and Dask writers.
    """
    contexts: list[dict[str, Any]] = []
    for _, row in pdf.iterrows():
        ctx = {col: row[col] for col in cel_columns}
        for col in ctx:
            if hasattr(ctx[col], "item"):
                ctx[col] = ctx[col].item()
        contexts.append(ctx)
    return contexts


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Lazy-initialized Arrow type compatibility map (populated on first call).
_ARROW_TO_CEL: dict[str, set[str]] | None = None


def _get_arrow_to_cel() -> dict[str, set[str]]:
    global _ARROW_TO_CEL
    if _ARROW_TO_CEL is None:
        import pyarrow as pa  # type: ignore[import-not-found]

        _ARROW_TO_CEL = {
            "int": {str(pa.int8()), str(pa.int16()), str(pa.int32()), str(pa.int64())},
            "uint": {
                str(pa.uint8()),
                str(pa.uint16()),
                str(pa.uint32()),
                str(pa.uint64()),
            },
            "double": {str(pa.float16()), str(pa.float32()), str(pa.float64())},
            "string": {
                str(pa.string()),
                str(pa.large_string()),
                str(pa.utf8()),
                str(pa.large_utf8()),
            },
            "bytes": {str(pa.binary()), str(pa.large_binary())},
            "bool": {str(pa.bool_())},
        }
    return _ARROW_TO_CEL


def validate_cel_columns(
    arrow_schema: Any,
    cel_columns: dict[str, str],
) -> None:
    """Validate that declared CEL columns exist in an Arrow schema."""
    schema_names = set(arrow_schema.names)
    for col_name in cel_columns:
        if col_name not in schema_names:
            raise ConfigValidationError(
                f"CEL column {col_name!r} not found in Arrow schema. "
                f"Available: {sorted(schema_names)}"
            )

    arrow_to_cel = _get_arrow_to_cel()
    for col_name, cel_type in cel_columns.items():
        field = arrow_schema.field(col_name)
        allowed = arrow_to_cel.get(cel_type, set())
        if allowed and str(field.type) not in allowed:
            raise ConfigValidationError(
                f"CEL column {col_name!r} declared as {cel_type!r} but Arrow type "
                f"is {field.type}. Compatible Arrow types: {sorted(allowed)}"
            )
