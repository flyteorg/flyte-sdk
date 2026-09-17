"""LibCST helpers shared by the migration rules."""

from __future__ import annotations

import ast
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import libcst as cst
from libcst.metadata import QualifiedNameProvider

from ._context import MigrationContext
from ._result import TODO_MARKER

_STATEMENTS = (cst.SimpleStatementLine, cst.BaseCompoundStatement)
_EMPTY_MODULE = cst.Module(body=[])

#: Returned by `literal()` when a node is not a Python literal.
NOT_LITERAL: Any = object()


def code(node: cst.CSTNode) -> str:
    """Render a node back to source code."""
    return _EMPTY_MODULE.code_for_node(node)


def literal(node: Optional[cst.CSTNode]) -> Any:
    """Evaluate a node as a Python literal, or return `NOT_LITERAL`."""
    if node is None:
        return NOT_LITERAL
    try:
        return ast.literal_eval(code(node))
    except (ValueError, SyntaxError, TypeError):
        return NOT_LITERAL


def call_arguments(
    call: cst.Call, positional_names: Sequence[str] = ()
) -> Tuple[Dict[str, cst.BaseExpression], List[cst.BaseExpression]]:
    """
    Split a call's arguments into keyword arguments and leftover positional arguments.

    Positional arguments are named after `positional_names`, in order; any beyond that are returned as leftovers.
    Star-arguments (`*args`, `**kwargs`) are returned as leftovers too.
    """
    kwargs: Dict[str, cst.BaseExpression] = {}
    leftovers: List[cst.BaseExpression] = []
    index = 0
    for arg in call.args:
        if arg.star:
            leftovers.append(arg.value)
        elif arg.keyword is not None:
            kwargs[arg.keyword.value] = arg.value
        elif index < len(positional_names):
            kwargs[positional_names[index]] = arg.value
            index += 1
        else:
            leftovers.append(arg.value)
    return kwargs, leftovers


def arguments_code(call: cst.Call) -> str:
    """Render a call's arguments without the surrounding parentheses."""
    return ", ".join(code(arg.with_changes(comma=cst.MaybeSentinel.DEFAULT)) for arg in call.args)


def kwargs_code(pairs: Sequence[Tuple[str, str]]) -> str:
    return ", ".join(f"{k}={v}" for k, v in pairs)


def todo_comment(message: str) -> cst.EmptyLine:
    return cst.EmptyLine(comment=cst.Comment(f"# {TODO_MARKER} {message}"))


def with_todos(statement: cst.CSTNode, messages: Sequence[str]) -> cst.CSTNode:
    """Prepend TODO comments to a statement, skipping ones that are already there."""
    existing = {line.comment.value for line in statement.leading_lines if line.comment}  # type: ignore[attr-defined]
    new_lines = [todo_comment(m) for m in dict.fromkeys(messages)]
    new_lines = [line for line in new_lines if line.comment and line.comment.value not in existing]
    if not new_lines:
        return statement
    return statement.with_changes(leading_lines=[*statement.leading_lines, *new_lines])  # type: ignore[attr-defined]


class Rule(cst.CSTTransformer):
    """
    Base class for a migration rule.

    Rules run one after another, each on a fresh metadata wrapper, so qualified names are always resolved against the
    output of the previous rule. `todo()` attaches a `TODO(flyte migrate)` comment to the statement being visited.
    """

    id: ClassVar[str]
    description: ClassVar[str]
    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self, ctx: MigrationContext):
        super().__init__()
        self.ctx = ctx
        self._todo_stack: List[List[str]] = []

    def run(self, module: cst.Module) -> cst.Module:
        return cst.MetadataWrapper(module).visit(self)

    def applied(self, what: str) -> None:
        self.ctx.applied[f"{self.id}.{what}"] += 1

    def todo(self, message: str) -> None:
        if self._todo_stack:
            self._todo_stack[-1].append(message)
        else:
            self.ctx.module_todos.append(message)

    def on_visit(self, node: cst.CSTNode) -> bool:
        if isinstance(node, _STATEMENTS):
            self._todo_stack.append([])
        return super().on_visit(node)

    def on_leave(  # type: ignore[override]
        self, original_node: cst.CSTNode, updated_node: cst.CSTNode
    ) -> Union[cst.CSTNode, cst.RemovalSentinel, cst.FlattenSentinel]:
        result = super().on_leave(original_node, updated_node)
        if isinstance(original_node, _STATEMENTS):
            messages = self._todo_stack.pop()
            if messages:
                result = self._attach(result, messages)
        return result

    def _attach(self, result: Any, messages: List[str]) -> Any:
        if isinstance(result, _STATEMENTS):
            return with_todos(result, messages)
        if isinstance(result, cst.FlattenSentinel) and result.nodes:
            first, *rest = result.nodes
            return cst.FlattenSentinel([with_todos(first, messages), *rest])
        # The statement was removed: hand the messages to the enclosing statement.
        for message in messages:
            self.todo(message)
        return result

    # -- qualified names ------------------------------------------------------------------------------------------

    def qualified_name(self, node: cst.CSTNode) -> Optional[str]:
        try:
            names = self.get_metadata(QualifiedNameProvider, node, set())
        except KeyError:
            return None
        for q in names:
            return q.name
        return None

    def flytekit_name(self, node: cst.CSTNode) -> Optional[str]:
        """The fully-qualified flytekit name a node refers to, e.g. `flytekit.types.file.FlyteFile`, or None."""
        try:
            names = self.get_metadata(QualifiedNameProvider, node, set())
        except KeyError:
            return None
        for q in names:
            if q.name == "flytekit" or q.name.startswith("flytekit."):
                return q.name
        return None

    def flytekit_symbol(self, node: cst.CSTNode) -> Optional[str]:
        """The last segment of the flytekit name a node refers to, e.g. `FlyteFile`, or None."""
        name = self.flytekit_name(node)
        return name.rsplit(".", 1)[-1] if name else None

    def is_flytekit(self, node: cst.CSTNode, *symbols: str) -> bool:
        return self.flytekit_symbol(node) in symbols
