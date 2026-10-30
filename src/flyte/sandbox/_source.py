"""Source extraction for Monty sandbox execution.

Monty (Pydantic's Rust-based sandboxed Python interpreter) natively supports
`return` statements inside function definitions and returns the value of the
last expression.  The functions here prepare Python source for Monty without
any AST rewriting:

- `extract_source`: takes a decorated function, strips decorators, and
  appends a trailing call so Monty executes the function and returns its result.
- `prepare_code_source`: takes a raw code string, dedents/strips it, and
  passes it through as-is.  Monty returns the last expression natively.
- `module_imports`: the imports of standard modules the function uses through
  its module's globals, which the extracted source would otherwise lack.
"""

from __future__ import annotations

import inspect
import textwrap
import types
from typing import Callable, List, Set, Tuple

# The standard modules Monty implements, which the source can import.
MONTY_MODULES = frozenset(
    {
        "asyncio",
        "base64",
        "binascii",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "functools",
        "itertools",
        "json",
        "math",
        "os",
        "pathlib",
        "random",
        "re",
        "sys",
        "time",
        "typing",
        "unicodedata",
    }
)


def extract_source(func: Callable) -> Tuple[str, List[str]]:
    """Extract the source of *func* for Monty execution.

    Returns `(code, input_names)` where *code* contains the full function
    definition (with `return` statements preserved) followed by a trailing
    call `func_name(param1, param2, ...)` so Monty executes the function
    and returns its result.  For `async def` functions the trailing call
    is wrapped in `await`.

    Raises `TypeError` for generator functions.
    """
    if inspect.isgeneratorfunction(func):
        raise TypeError(f"Sandboxed tasks cannot be generators: {getattr(func, '__qualname__')}")
    if inspect.isasyncgenfunction(func):
        raise TypeError(f"Sandboxed tasks cannot be async generators: {getattr(func, '__qualname__')}")

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
    # For async functions, wrap in `await` so Monty drives the coroutine.
    func_name = getattr(func, "__name__")
    args_str = ", ".join(input_names)
    call = f"{func_name}({args_str})"
    if inspect.iscoroutinefunction(func):
        call = f"await {call}"
    code = f"{func_source.rstrip()}\n{call}"

    return code, input_names


def prepare_code_source(source: str) -> str:
    """Dedent and strip a user code string for Monty execution.

    Monty returns the value of the last expression natively, so no
    rewriting is needed.  Returns `"None"` for empty input.
    """
    source = textwrap.dedent(source).strip()
    if not source:
        return "None"
    return source


def referenced_names(func: Callable) -> Set[str]:
    """The global and free names *func* uses, including inside the functions,
    lambdas and comprehensions it defines (which before Python 3.12 are
    functions of their own, with their own code objects)."""
    names: Set[str] = set()
    pending = [getattr(func, "__code__")]
    while pending:
        code = pending.pop()
        names.update(code.co_names, code.co_freevars)
        pending.extend(const for const in code.co_consts if isinstance(const, types.CodeType))
    return names


def module_imports(func: Callable) -> str:
    """Imports for the standard modules *func* reaches through its globals.

    Only the function's source is shipped, so a module imported at the top of
    its file (`import asyncio`, `import math as m`) is undefined in the sandbox
    unless it is imported again. Modules Monty does not implement are left out.
    """
    func_globals = getattr(func, "__globals__", {})
    lines = []
    for name in sorted(referenced_names(func)):
        module = func_globals.get(name)
        if not isinstance(module, types.ModuleType) or module.__name__ not in MONTY_MODULES:
            continue
        lines.append(f"import {module.__name__}" if name == module.__name__ else f"import {module.__name__} as {name}")
    return "".join(f"{line}\n" for line in lines)
