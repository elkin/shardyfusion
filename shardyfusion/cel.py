"""CEL (Common Expression Language) integration for flexible sharding.

Provides compile/evaluate/boundary functions shared by all writer backends
and the reader routing layer.  Requires the ``cel`` extra:
``pip install shardyfusion[cel]``.

Uses ``cel-expr-python`` (Google's C++ CEL wrapper) via:
    ``from cel_expr_python.cel import NewEnv, Type``
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from typing import Any

from .errors import ConfigValidationError
from .routing import _XXHASH64_SEED
from .sharding_types import BoundaryValue, ShardingSpec, ShardingStrategy

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


def _import_fastdigest() -> Any:
    try:
        import fastdigest  # type: ignore[import-not-found]

        return fastdigest
    except ImportError as exc:
        raise ImportError(
            "CEL boundary computation requires the 'cel' extra. "
            "Install it with: pip install shardyfusion[cel]"
        ) from exc


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
        cel_result = self._expr.eval(data=context)  # noqa: S307
        value = cel_result.plain_value()
        if not isinstance(value, (int, str, bytes)):
            raise ConfigValidationError(
                f"CEL expression must return int, str, or bytes; got {type(value).__name__}"
            )
        return value


def compile_cel(expr: str, columns: dict[str, str]) -> CompiledCel:
    """Compile a CEL expression with declared column types.

    Args:
        expr: CEL expression string (e.g., ``"key % 1000"``).
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
        env = cel.NewEnv(variables=variables)
        compiled_expr = env.compile(expr)
    except Exception as exc:
        raise ConfigValidationError(
            f"Failed to compile CEL expression {expr!r}: {exc}"
        ) from exc

    return CompiledCel(env=env, expr=compiled_expr, columns=columns)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_cel_arrow_batch(
    compiled: CompiledCel,
    batch: Any,
) -> list[int | str | bytes]:
    """Evaluate CEL on each row of a PyArrow RecordBatch/Table.

    Shared by Spark (mapInArrow), Ray (map_batches), and Dask writers.

    Args:
        compiled: Compiled CEL expression.
        batch: PyArrow RecordBatch or Table.

    Returns:
        List of routing key values, one per row.
    """
    col_names = list(compiled.columns)
    col_arrays = {name: batch.column(name).to_pylist() for name in col_names}
    num_rows = len(batch)

    # Reuse a single mutable dict to avoid per-row allocation.
    context: dict[str, Any] = {}
    results: list[int | str | bytes] = []
    for i in range(num_rows):
        for name in col_names:
            context[name] = col_arrays[name][i]
        results.append(compiled.evaluate(context))

    return results


def route_cel_batch(
    compiled: CompiledCel,
    batch: Any,
    boundaries: list[BoundaryValue],
) -> list[int]:
    """Evaluate CEL and route each row to a shard via bisect_right.

    Combines ``evaluate_cel_arrow_batch`` + ``bisect_right`` in one pass
    to keep routing logic in one place.
    """
    routing_keys = evaluate_cel_arrow_batch(compiled, batch)
    return [bisect_right(boundaries, rk) for rk in routing_keys]


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_cel(
    compiled: CompiledCel,
    context: dict[str, Any],
    boundaries: list[BoundaryValue],
) -> int:
    """Evaluate CEL expression and route to a shard via bisect_right.

    Args:
        compiled: Compiled CEL expression.
        context: Variable bindings for the expression.
        boundaries: Sorted boundary values.

    Returns:
        Shard db_id (0-based).
    """
    routing_key = compiled.evaluate(context)
    return bisect_right(boundaries, routing_key)


# ---------------------------------------------------------------------------
# Boundary computation
# ---------------------------------------------------------------------------


def compute_boundaries_tdigest(
    routing_keys: Iterable[int | str | bytes],
    num_dbs: int,
) -> list[BoundaryValue]:
    """Compute quantile boundaries from routing keys using t-digest.

    For non-numeric keys (str, bytes), keys are hashed to int before
    feeding t-digest so the quantile structure works uniformly.

    Args:
        routing_keys: Iterable of routing key values.
        num_dbs: Target number of shards.

    Returns:
        Sorted list of ``num_dbs - 1`` boundary values.
    """
    import xxhash

    fd = _import_fastdigest()

    expected = max(num_dbs - 1, 0)
    if expected == 0:
        return []

    digest = fd.TDigest()

    for key in routing_keys:
        if isinstance(key, int):
            digest.update(float(key))
        elif isinstance(key, str):
            digest.update(
                float(xxhash.xxh64_intdigest(key.encode("utf-8"), seed=_XXHASH64_SEED))
            )
        elif isinstance(key, bytes):
            digest.update(float(xxhash.xxh64_intdigest(key, seed=_XXHASH64_SEED)))
        else:
            raise ConfigValidationError(
                f"Unsupported routing key type for t-digest: {type(key)!r}"
            )

    probabilities = [idx / num_dbs for idx in range(1, num_dbs)]
    boundaries: list[BoundaryValue] = []
    for p in probabilities:
        q = digest.quantile(p)
        boundaries.append(int(q))

    return boundaries


def compute_boundaries_distinct(
    routing_keys: Iterable[int | str | bytes],
) -> tuple[int, list[BoundaryValue]]:
    """Compute boundaries from distinct values (data-driven mode).

    One shard per distinct routing key value.

    Args:
        routing_keys: Iterable of routing key values.

    Returns:
        Tuple of (num_dbs, sorted boundary list).
        num_dbs = number of distinct values.
        boundaries has num_dbs - 1 elements.
    """
    distinct: set[int | str | bytes] = set()
    for key in routing_keys:
        distinct.add(key)

    sorted_values = sorted(distinct, key=_boundary_sort_key)
    num_dbs = len(sorted_values)

    if num_dbs <= 1:
        return max(num_dbs, 1), []

    boundaries: list[BoundaryValue] = list(sorted_values[1:])
    return num_dbs, boundaries


def resolve_cel_boundaries(
    compiled: CompiledCel,
    routing_keys: Iterable[dict[str, Any]],
    num_dbs: int,
    sharding: ShardingSpec,
) -> ShardingSpec:
    """Evaluate CEL on sampled rows and compute t-digest boundaries.

    Shared helper for Spark, Dask, and Ray writers. Each framework
    samples rows in its own way and passes them as dicts here.

    Args:
        compiled: Pre-compiled CEL expression.
        routing_keys: Iterable of column-value dicts (one per sampled row).
        num_dbs: Target number of shards.
        sharding: Original ShardingSpec (cel_expr/cel_columns carried over).

    Returns:
        New ShardingSpec with computed boundaries.
    """
    evaluated = [compiled.evaluate(ctx) for ctx in routing_keys]
    boundaries = compute_boundaries_tdigest(evaluated, num_dbs)
    return ShardingSpec(
        strategy=ShardingStrategy.CEL,
        boundaries=boundaries,
        cel_expr=sharding.cel_expr,
        cel_columns=sharding.cel_columns,
    )


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


def _boundary_sort_key(value: int | str | bytes) -> tuple[int, Any]:
    """Sort key that groups by type to avoid cross-type comparison."""
    if isinstance(value, int):
        return (0, value)
    if isinstance(value, str):
        return (1, value)
    if isinstance(value, bytes):
        return (2, value)
    return (3, value)


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
