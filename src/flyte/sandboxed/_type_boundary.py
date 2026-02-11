from __future__ import annotations

import typing
from typing import Any, Dict, Tuple, Type

from flyte.io import DataFrame, Dir, File

# Types that Monty can handle natively
_MONTY_PRIMITIVE_TYPES = frozenset({
    int,
    float,
    str,
    bool,
    bytes,
    type(None),
})

# Flyte IO types passed as opaque handles through Monty
_MONTY_COLLECTION_TYPES = frozenset({
    list,
    dict,
    tuple,
    set,
    frozenset,
})

_FLYTE_IO_TYPES = frozenset({
    File,
    Dir,
    DataFrame,
})

_ALLOWED_TYPES = _MONTY_PRIMITIVE_TYPES | _MONTY_COLLECTION_TYPES | _FLYTE_IO_TYPES


def _is_allowed_type(tp: Type) -> bool:
    """Check whether *tp* is a Monty-compatible type."""
    if tp is Any or tp is type(None):
        return True

    # Unwrap Optional / Union
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        return all(_is_allowed_type(arg) for arg in typing.get_args(tp))

    # Generic collections: list[int], dict[str, int], tuple[int, ...], set[int]
    if origin in (list, dict, tuple, set, frozenset):
        args = typing.get_args(tp)
        if not args:
            return True  # bare list, dict, etc.
        return all(_is_allowed_type(arg) for arg in args if arg is not Ellipsis)

    return tp in _ALLOWED_TYPES


def validate_sandboxed_interface(
    inputs: Dict[str, Tuple[Type, Any]],
    outputs: Dict[str, Type],
) -> None:
    """Validate that all input and output types are Monty-compatible.

    Raises ``TypeError`` if any type is unsupported.
    """
    import inspect

    for name, (tp, _default) in inputs.items():
        if tp is inspect.Parameter.empty:
            continue  # untyped — will be pickled, let it through
        if not _is_allowed_type(tp):
            raise TypeError(
                f"Sandboxed task input '{name}' has unsupported type {tp!r}. "
                f"Supported types: primitives (int, float, str, bool, bytes, None), "
                f"collections (list, dict, tuple, set), and flyte.io types (File, Dir, DataFrame)."
            )

    for name, tp in outputs.items():
        if tp is inspect.Parameter.empty:
            continue
        if not _is_allowed_type(tp):
            raise TypeError(
                f"Sandboxed task output '{name}' has unsupported type {tp!r}. "
                f"Supported types: primitives (int, float, str, bool, bytes, None), "
                f"collections (list, dict, tuple, set), and flyte.io types (File, Dir, DataFrame)."
            )
