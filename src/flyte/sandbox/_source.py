"""Source extraction for Monty sandbox execution.

Monty (Pydantic's Rust-based sandboxed Python interpreter) natively supports
``return`` statements inside function definitions and returns the value of the
last expression.  The functions here prepare Python source for Monty without
any AST rewriting:

- ``extract_source``: takes a decorated function, strips decorators, and
  appends a trailing call so Monty executes the function and returns its result.
- ``prepare_code_source``: takes a raw code string, dedents/strips it, and
  passes it through as-is.  Monty returns the last expression natively.
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Callable, List, Tuple


def extract_source(func: Callable) -> Tuple[str, List[str]]:
    """Extract the source of *func* for Monty execution.

    Returns ``(code, input_names)`` where *code* contains the full function
    definition (with ``return`` statements preserved) followed by a trailing
    call ``func_name(param1, param2, ...)`` so Monty executes the function
    and returns its result.  For ``async def`` functions the trailing call
    is wrapped in ``await``.

    Raises ``TypeError`` for generator functions.
    """
    if inspect.isgeneratorfunction(func):
        raise TypeError(f"Sandboxed tasks cannot be generators: {func.__qualname__}")
    if inspect.isasyncgenfunction(func):
        raise TypeError(f"Sandboxed tasks cannot be async generators: {func.__qualname__}")

    source = inspect.getsource(func)
    dedented = textwrap.dedent(source)

    # Strip decorator lines: find the first line starting with 'def ' or 'async def '
    lines = dedented.splitlines(keepends=True)
    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "async def")):
            start_idx = i
            break
    func_source = "".join(lines[start_idx:])

    # Get parameter names via inspect.signature
    sig = inspect.signature(func)
    input_names = [
        name
        for name, param in sig.parameters.items()
        if param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
    ]

    # Append trailing call: func_name(param1, param2, ...)
    # For async functions, wrap in ``await`` so Monty drives the coroutine.
    func_name = func.__name__
    args_str = ", ".join(input_names)
    call = f"{func_name}({args_str})"
    if inspect.iscoroutinefunction(func):
        call = f"await {call}"
    code = f"{func_source.rstrip()}\n{call}"

    return code, input_names


def prepare_code_source(source: str) -> str:
    """Dedent and strip a user code string for Monty execution.

    Monty returns the value of the last expression natively, so no
    rewriting is needed.  Returns ``"None"`` for empty input.
    """
    source = textwrap.dedent(source).strip()
    if not source:
        return "None"
    return source
