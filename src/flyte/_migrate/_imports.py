"""Final import bookkeeping: add the imports generated code needs and drop flytekit imports that became unused."""

from __future__ import annotations

import libcst as cst
from libcst.codemod import CodemodContext
from libcst.codemod.visitors import AddImportsVisitor, RemoveImportsVisitor

from ._context import MigrationContext

_COLLECT_HELPER = '''async def _collect(results):
    """Gather the results of `flyte.map.aio` into a list, so the map can run inside `asyncio.gather`."""
    return [r async for r in results]
'''


def finalize_imports(module: cst.Module, ctx: MigrationContext) -> cst.Module:
    context = CodemodContext()
    for module_name, obj in sorted(ctx.imports, key=lambda i: (i[0], i[1] or "")):
        AddImportsVisitor.add_needed_import(context, module_name, obj)
    module = AddImportsVisitor(context).transform_module(module)

    context = CodemodContext()
    for statement in module.body:
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for small in statement.body:
            if isinstance(small, cst.Import):
                for alias in small.names:
                    name = cst.Module([]).code_for_node(alias.name)
                    if _is_flytekit(name):
                        RemoveImportsVisitor.remove_unused_import(context, name, asname=_asname(alias))
            elif isinstance(small, cst.ImportFrom) and small.module is not None and not small.relative:
                name = cst.Module([]).code_for_node(small.module)
                if _is_flytekit(name) and not isinstance(small.names, cst.ImportStar):
                    for alias in small.names:
                        obj = cst.Module([]).code_for_node(alias.name)
                        RemoveImportsVisitor.remove_unused_import(context, name, obj, asname=_asname(alias))
    module = RemoveImportsVisitor(context).transform_module(module)

    if ctx.needs_collect_helper:
        helper = [
            statement.with_changes(leading_lines=[cst.EmptyLine(), cst.EmptyLine()])
            for statement in cst.parse_module(_COLLECT_HELPER).body
        ]
        module = _insert_after_imports(module, helper)
    return module


def _is_flytekit(module_name: str) -> bool:
    return module_name == "flytekit" or module_name.startswith("flytekit.")


def _asname(alias: cst.ImportAlias):
    if alias.asname is None:
        return None
    return cst.Module([]).code_for_node(alias.asname.name)


def _insert_after_imports(module: cst.Module, statements) -> cst.Module:
    body = list(module.body)
    index = 0
    for i, statement in enumerate(body):
        if isinstance(statement, cst.SimpleStatementLine) and all(
            isinstance(s, (cst.Import, cst.ImportFrom)) for s in statement.body
        ):
            index = i + 1
    body[index:index] = statements
    return module.with_changes(body=body)
