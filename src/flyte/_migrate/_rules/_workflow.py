from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import libcst as cst
import libcst.matchers as m

from .._conversions import convert_map, override_arguments
from .._cst import Rule, arguments_code, call_arguments, code

_SEQUENTIAL_TODO = (
    "this workflow body could not be analyzed, so its calls run sequentially in v2; "
    "run independent calls concurrently with asyncio.gather"
)


@dataclass
class _Node:
    """One call in a workflow body."""

    index: int
    statement: cst.SimpleStatementLine
    target: Optional[cst.BaseExpression]
    #: Source of an awaitable (for single calls) and, for maps, of the async generator.
    awaitable: str
    is_map: bool = False
    is_override: bool = False
    is_return: bool = False
    uses: Set[str] = field(default_factory=set)
    after: Set[int] = field(default_factory=set)
    level: int = 0


class _Unsupported(Exception):
    pass


class WorkflowRule(Rule):
    """
    Turn `@workflow` bodies whose DAG has independent calls into async code that preserves v1 parallelism.

    v1 runs independent workflow nodes concurrently. Plain v2 Python runs calls one after another, so bodies with
    independent calls become `async def` with `asyncio.gather`. Bodies without independent calls stay synchronous.
    """

    id = "workflow"
    description = "@workflow bodies with independent calls -> async + asyncio.gather"

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        entity = self.ctx.entities.get(original_node.name.value)
        if entity is None or entity.kind != "workflow" or original_node.asynchronous is not None:
            return updated_node
        if not isinstance(original_node.body, cst.IndentedBlock):
            return updated_node

        # Conversions record TODOs while planning; keep them only if the body is actually rewritten.
        self._todo_stack.append([])
        try:
            docstring, nodes, returns = self._plan(original_node.body)
        except _Unsupported:
            self._todo_stack.pop()
            # Bodies with other flytekit constructs (e.g. conditional) get a more specific TODO from LeftoversRule.
            uses_flytekit = any(self.flytekit_name(n) for n in m.findall(original_node.body, m.Name() | m.Attribute()))
            if not uses_flytekit and len(m.findall(original_node.body, m.Call())) > 1:
                self.todo(_SEQUENTIAL_TODO)
            return updated_node
        planning_todos = self._todo_stack.pop()

        levels = _assign_levels(nodes)
        if all(len(level) < 2 for level in levels):
            return updated_node
        for message in planning_todos:
            self.todo(message)

        body: List[cst.BaseStatement] = list(docstring)
        for level in levels:
            body.append(self._emit_level(level))
        if returns is not None:
            body.append(returns)
        self.ctx.require_import("asyncio")
        self.applied("parallel")
        return updated_node.with_changes(
            asynchronous=cst.Asynchronous(),
            body=updated_node.body.with_changes(body=body),
        )

    # -- analysis ---------------------------------------------------------------------------------------------------

    def _plan(self, block: cst.IndentedBlock):
        docstring: List[cst.BaseStatement] = []
        nodes: List[_Node] = []
        producers: Dict[str, int] = {}
        returns: Optional[cst.BaseStatement] = None

        for position, statement in enumerate(block.body):
            if returns is not None or not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
                raise _Unsupported
            small = statement.body[0]
            if position == 0 and m.matches(small, m.Expr(m.SimpleString() | m.ConcatenatedString())):
                docstring.append(statement)
            elif isinstance(small, cst.Pass):
                continue
            elif isinstance(small, cst.Assign):
                if len(small.targets) != 1 or not isinstance(small.value, cst.Call):
                    raise _Unsupported
                target = small.targets[0].target
                names = _target_names(target)
                node = self._node(len(nodes), statement, target, small.value)
                for name in names:
                    if name in producers:
                        raise _Unsupported
                    producers[name] = node.index
                nodes.append(node)
            elif isinstance(small, cst.Expr) and isinstance(small.value, cst.Call):
                nodes.append(self._node(len(nodes), statement, None, small.value))
            elif isinstance(small, cst.Expr) and isinstance(small.value, cst.BinaryOperation):
                self._ordering(small.value, producers, nodes)
            elif isinstance(small, cst.Return):
                if isinstance(small.value, cst.Call):
                    node = self._node(len(nodes), statement, None, small.value)
                    node.is_return = True
                    nodes.append(node)
                else:
                    returns = statement
            else:
                raise _Unsupported

        for node in nodes:
            node.after |= {producers[name] for name in node.uses if name in producers}
        return docstring, nodes, returns

    def _node(
        self, index: int, statement: cst.SimpleStatementLine, target: Optional[cst.BaseExpression], call: cst.Call
    ) -> _Node:
        func = call.func
        if isinstance(func, cst.Attribute) and func.attr.value == "with_overrides" and isinstance(func.value, cst.Call):
            # Override values are configuration (e.g. `requests=Resources(...)`), not data dependencies.
            inner = func.value
            overrides, positional = call_arguments(call)
            if positional or self.flytekit_name(inner.func) is not None:
                raise _Unsupported
            return _Node(
                index,
                statement,
                target,
                f"{code(inner.func)}.override({override_arguments(self, overrides)}).aio({arguments_code(inner)})",
                is_override=True,
                uses=_data_names(inner),
            )

        mapped = convert_map(self, call, is_async=True)
        if mapped is not None:
            assert isinstance(func, cst.Call)
            return _Node(index, statement, target, mapped, is_map=True, uses=_data_names(call) | _data_names(func))

        if self.flytekit_name(func) is not None or not isinstance(func, (cst.Name, cst.Attribute)):
            raise _Unsupported
        return _Node(index, statement, target, f"{code(func)}.aio({arguments_code(call)})", uses=_data_names(call))

    def _ordering(self, operation: cst.BinaryOperation, producers: Dict[str, int], nodes: List[_Node]) -> None:
        """Record `a >> b >> c` as explicit ordering edges."""
        operands: List[cst.BaseExpression] = []

        def flatten(expr: cst.BaseExpression) -> None:
            if isinstance(expr, cst.BinaryOperation) and isinstance(expr.operator, cst.RightShift):
                flatten(expr.left)
                flatten(expr.right)
            else:
                operands.append(expr)

        flatten(operation)
        if len(operands) < 2 or not all(isinstance(o, cst.Name) and o.value in producers for o in operands):
            raise _Unsupported
        indexes = [producers[o.value] for o in operands]  # type: ignore[attr-defined]
        for before, after in itertools.pairwise(indexes):
            nodes[after].after.add(before)
        self.applied("ordering")

    # -- emission ---------------------------------------------------------------------------------------------------

    def _emit_level(self, level: List[_Node]) -> cst.BaseStatement:
        for node in level:
            if node.is_map:
                self.applied("map_task")
            if node.is_override:
                self.applied("with_overrides")
        leading = [line for node in level for line in node.statement.leading_lines]
        if len(level) == 1:
            node = level[0]
            if node.is_map:
                value = f"[r async for r in {node.awaitable}]"
            else:
                value = f"await {node.awaitable}"
            source = _assignment(node, value)
        else:
            awaitables = []
            for node in level:
                if node.is_map:
                    self.ctx.needs_collect_helper = True
                    awaitables.append(f"_collect({node.awaitable})")
                else:
                    awaitables.append(node.awaitable)
            targets = [code(node.target) if node.target is not None else "_" for node in level]
            value = f"await asyncio.gather({', '.join(awaitables)})"
            if all(t == "_" for t in targets):
                source = value
            else:
                source = f"{', '.join(_parenthesize(t) for t in targets)} = {value}"
        return cst.parse_statement(source + "\n").with_changes(leading_lines=leading)


def _data_names(call: cst.Call) -> Set[str]:
    """Names a call's arguments read. Nested calls are not modeled, so they make the body unsupported."""
    names: Set[str] = set()
    for arg in call.args:
        if m.findall(arg.value, m.Call() | m.Lambda() | m.Await()) and not _is_partial(arg.value):
            raise _Unsupported
        names |= {n.value for n in m.findall(arg.value, m.Name()) if isinstance(n, cst.Name)}
    return names


def _is_partial(expr: cst.BaseExpression) -> bool:
    return isinstance(expr, cst.Call) and code(expr.func) in ("partial", "functools.partial")


def _assign_levels(nodes: List[_Node]) -> List[List[_Node]]:
    for node in nodes:
        node.level = max((nodes[i].level + 1 for i in node.after), default=0)
    for node in nodes:
        if node.is_return:
            node.level = max((n.level for n in nodes if n is not node), default=-1) + 1
    levels: Dict[int, List[_Node]] = {}
    for node in nodes:
        levels.setdefault(node.level, []).append(node)
    return [levels[k] for k in sorted(levels)]


def _assignment(node: _Node, value: str) -> str:
    if node.is_return:
        return f"return {value}"
    if node.target is None:
        return value
    return f"{code(node.target)} = {value}"


def _parenthesize(target: str) -> str:
    return f"({target})" if "," in target and not target.startswith("(") else target


def _target_names(target: cst.BaseExpression) -> List[str]:
    if isinstance(target, cst.Name):
        return [target.value]
    if isinstance(target, (cst.Tuple, cst.List)):
        names: List[str] = []
        for element in target.elements:
            names.extend(_target_names(element.value))
        return names
    raise _Unsupported
