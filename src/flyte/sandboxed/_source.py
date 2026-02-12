"""AST rewriting for Monty sandbox execution.

Monty (Pydantic's Rust-based sandboxed Python interpreter) doesn't support
``return`` statements — it returns the value of the *last expression* in the
code.  The functions here use ``ast`` to transform normal Python source into
Monty-compatible form:

- ``extract_source``: takes a decorated function, extracts its body, and
  rewrites every ``return X`` into ``__result__ = X``.
- ``prepare_code_source``: takes a raw code string and rewrites the last
  statement so its value is captured in ``__result__``.

Both append ``__result__`` as the final expression so Monty returns it.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable, List, Tuple


def _make_result_assign(value: ast.expr, *, lineno: int = 0, col_offset: int = 0) -> ast.Assign:
    """Create ``__result__ = <value>`` and fix locations."""
    node = ast.Assign(
        targets=[ast.Name(id="__result__", ctx=ast.Store())],
        value=value,
        lineno=lineno,
        col_offset=col_offset,
    )
    return ast.fix_missing_locations(node)


def _unparse_with_result(tree: ast.Module) -> str:
    """Unparse *tree* and append ``__result__`` as the final expression."""
    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    code += "\n__result__"
    return code


class _ReturnRewriter(ast.NodeTransformer):
    """Rewrite ``return X`` to ``__result__ = X`` so Monty can capture the result."""

    def visit_Return(self, node: ast.Return) -> ast.AST:
        value = node.value if node.value is not None else ast.Constant(value=None)
        return _make_result_assign(value, lineno=node.lineno, col_offset=node.col_offset)

    # Don't recurse into nested function/class definitions
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return node


def extract_source(func: Callable) -> Tuple[str, List[str]]:
    """Extract the body source of *func* for Monty execution.

    Returns ``(code, input_names)`` where *code* has ``return`` statements
    rewritten to ``__result__ = ...`` assignments, with ``__result__``
    appended as the final expression so Monty returns it.

    Raises ``TypeError`` for async, generator, or context-manager functions.
    """
    if inspect.iscoroutinefunction(func):
        raise TypeError(f"Sandboxed tasks cannot be async: {func.__qualname__}")
    if inspect.isgeneratorfunction(func):
        raise TypeError(f"Sandboxed tasks cannot be generators: {func.__qualname__}")
    if inspect.isasyncgenfunction(func):
        raise TypeError(f"Sandboxed tasks cannot be async generators: {func.__qualname__}")

    source = inspect.getsource(func)
    dedented = textwrap.dedent(source)
    tree = ast.parse(dedented)

    # The top-level node should be a Module containing a single FunctionDef
    func_def: ast.FunctionDef | None = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break

    if func_def is None:
        raise TypeError(f"Could not find function definition in source of {func.__qualname__}")

    # Validate: reject async def at the AST level too (belt-and-suspenders)
    if isinstance(func_def, ast.AsyncFunctionDef):
        raise TypeError(f"Sandboxed tasks cannot be async: {func.__qualname__}")

    # Check for yield / yield from in the body
    for node in ast.walk(ast.Module(body=func_def.body, type_ignores=[])):
        if isinstance(node, (ast.Yield, ast.YieldFrom)):
            raise TypeError(f"Sandboxed tasks cannot be generators: {func.__qualname__}")

    # Extract input parameter names
    input_names = [arg.arg for arg in func_def.args.args]

    # Build a new Module from just the function body, with returns rewritten
    body_module = ast.Module(body=func_def.body, type_ignores=[])
    rewritten = _ReturnRewriter().visit(body_module)

    return _unparse_with_result(rewritten), input_names


def prepare_code_source(source: str) -> str:
    """Transform user code so Monty returns the value of the last expression.

    - If the last statement is an expression: assigns it to ``__result__``
    - If the last statement is a simple assignment ``x = ...``: appends ``__result__ = x``
    - Appends ``__result__`` as the final expression for Monty to return.

    This mirrors the ``return`` → ``__result__`` rewriting that
    ``extract_source`` does for decorated functions.
    """
    source = textwrap.dedent(source).strip()
    if not source:
        return "__result__ = None\n__result__"

    tree = ast.parse(source)
    if not tree.body:
        return "__result__ = None\n__result__"

    last = tree.body[-1]

    if isinstance(last, ast.Expr):
        # Expression statement → replace with ``__result__ = expr``
        tree.body[-1] = _make_result_assign(last.value, lineno=last.lineno, col_offset=last.col_offset)
    elif isinstance(last, ast.Assign) and len(last.targets) == 1 and isinstance(last.targets[0], ast.Name):
        # Simple assignment ``x = expr`` → append ``__result__ = x``
        var_name = last.targets[0].id
        tree.body.append(_make_result_assign(ast.Name(id=var_name, ctx=ast.Load())))

    return _unparse_with_result(tree)
