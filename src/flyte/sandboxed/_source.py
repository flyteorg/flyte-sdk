from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Callable, List, Tuple


class _ReturnRewriter(ast.NodeTransformer):
    """Rewrite ``return X`` to ``__result__ = X`` so Monty can capture the result."""

    def visit_Return(self, node: ast.Return) -> ast.AST:
        value = node.value if node.value is not None else ast.Constant(value=None)
        assign = ast.Assign(
            targets=[ast.Name(id="__result__", ctx=ast.Store())],
            value=value,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )
        return ast.fix_missing_locations(assign)

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
    ast.fix_missing_locations(rewritten)

    code = ast.unparse(rewritten)

    # Append __result__ as the final expression so Monty returns it.
    # Monty returns the value of the last expression in the code.
    code += "\n__result__"

    return code, input_names
