"""CodeTaskTemplate — a sandboxed task created from a code string."""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flyte._task import AsyncFunctionTaskTemplate

from ._config import SandboxedConfig
from ._task import SandboxedTaskTemplate, _lazy_import_monty
from ._type_boundary import validate_sandboxed_interface


def _classify_refs(functions: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Classify user-provided functions into task/trace/durable ref buckets.

    Same classification logic as ``_discover_external_refs`` but from an
    explicit dict instead of scanning function globals.
    """
    from flyte._task import TaskTemplate

    task_refs: Dict[str, Any] = {}
    trace_refs: Dict[str, Any] = {}
    durable_refs: Dict[str, Any] = {}

    for name, obj in functions.items():
        if isinstance(obj, TaskTemplate):
            task_refs[name] = obj
        elif hasattr(obj, "__wrapped__") and hasattr(obj, "aio"):
            # @trace or @syncify wrapped functions
            trace_refs[name] = obj
        elif hasattr(obj, "__module__") and obj.__module__ and "durable" in obj.__module__:
            durable_refs[name] = obj
        else:
            # Default: treat unrecognized callables as trace refs
            trace_refs[name] = obj

    return {
        "task_refs": task_refs,
        "trace_refs": trace_refs,
        "durable_refs": durable_refs,
    }


def _prepare_code_source(source: str) -> str:
    """Transform user code so Monty returns the value of the last expression.

    - If the last statement is an expression: assigns it to ``__result__``
    - If the last statement is a simple assignment ``x = ...``: appends ``__result__ = x``
    - Appends ``__result__`` as the final expression for Monty to return.

    This mirrors the ``return`` → ``__result__`` rewriting that
    ``_source.extract_source`` does for decorated functions.
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
        assign = ast.Assign(
            targets=[ast.Name(id="__result__", ctx=ast.Store())],
            value=last.value,
            lineno=last.lineno,
            col_offset=last.col_offset,
        )
        tree.body[-1] = ast.fix_missing_locations(assign)
    elif isinstance(last, ast.Assign) and len(last.targets) == 1 and isinstance(last.targets[0], ast.Name):
        # Simple assignment ``x = expr`` → append ``__result__ = x``
        var_name = last.targets[0].id
        result_node = ast.Assign(
            targets=[ast.Name(id="__result__", ctx=ast.Store())],
            value=ast.Name(id=var_name, ctx=ast.Load()),
            lineno=0,
            col_offset=0,
        )
        tree.body.append(ast.fix_missing_locations(result_node))

    ast.fix_missing_locations(tree)
    code = ast.unparse(tree)
    code += "\n__result__"
    return code


@dataclass(kw_only=True)
class CodeTaskTemplate(SandboxedTaskTemplate):
    """A sandboxed task created from a code string rather than a decorated function.

    Unlike ``SandboxedTaskTemplate`` (which extracts source from a Python
    function), this class accepts pre-transformed source code and an explicit
    dict of external functions.  It is constructed via :func:`flyte.sandboxed.code`.
    """

    # Init fields specific to CodeTaskTemplate
    _user_source: str = field(default="", repr=False)
    _user_input_names: List[str] = field(default_factory=list, repr=False)
    _user_functions: Dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Skip SandboxedTaskTemplate.__post_init__ which would try to
        # inspect.getsource(self.func).  Go directly to
        # AsyncFunctionTaskTemplate.__post_init__ (handles
        # _call_as_synchronous flag) → TaskTemplate.__post_init__
        # (handles image/cache/retries/short_name).
        AsyncFunctionTaskTemplate.__post_init__(self)

        if self.plugin_config is None:
            self.plugin_config = SandboxedConfig()

        validate_sandboxed_interface(self.interface.inputs, self.interface.outputs)

        # Populate the inherited internal fields from our init fields
        self._source_code = self._user_source
        self._input_names = list(self._user_input_names)

        # Build _external_refs from the explicit functions dict
        self._external_refs = _classify_refs(self._user_functions)
        self._has_external_refs = bool(self._user_functions)

    def forward(self, *args, **kwargs) -> Any:
        """Not supported — there is no Python function to call directly."""
        raise NotImplementedError(
            "CodeTaskTemplate does not support forward(). "
            "Use flyte.run() to execute through the sandbox."
        )
