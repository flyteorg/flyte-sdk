from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from flyte._task import AsyncFunctionTaskTemplate

from ._config import SandboxedConfig
from ._source import extract_source
from ._type_boundary import validate_sandboxed_interface


def _lazy_import_monty():
    """Lazy import of pydantic_monty.Monty with a helpful error message."""
    try:
        from pydantic_monty import Monty

        return Monty
    except ImportError:
        raise ImportError(
            "pydantic-monty is required for sandboxed tasks. "
            "Install it with: pip install 'flyte[sandbox]' or pip install pydantic-monty"
        ) from None


def _discover_external_refs(func) -> Dict[str, Dict[str, Any]]:
    """Discover external references (tasks, traces, durables) in the function's globals.

    Returns a dict with keys ``task_refs``, ``trace_refs``, ``durable_refs``.
    """
    from flyte._task import TaskTemplate
    from flyte.remote._task import LazyEntity

    task_refs: Dict[str, Any] = {}
    trace_refs: Dict[str, Any] = {}
    durable_refs: Dict[str, Any] = {}

    # Get the code object's free variables and global references
    func_globals = getattr(func, "__globals__", {})
    code = func.__code__

    # Only look at names actually referenced in the function
    referenced_names = set(code.co_names) | set(code.co_freevars)

    for name in referenced_names:
        obj = func_globals.get(name)
        if obj is None:
            continue

        if isinstance(obj, (TaskTemplate, LazyEntity)):
            task_refs[name] = obj
        elif hasattr(obj, "__wrapped__") and hasattr(obj, "aio"):
            # @trace or @syncify wrapped functions
            trace_refs[name] = obj
        # Check for durable module references
        elif hasattr(obj, "__module__") and obj.__module__ and "durable" in obj.__module__:
            durable_refs[name] = obj

    return {
        "task_refs": task_refs,
        "trace_refs": trace_refs,
        "durable_refs": durable_refs,
    }


@dataclass(kw_only=True)
class SandboxedTaskTemplate(AsyncFunctionTaskTemplate):
    """A task template that executes the function body in a Monty sandbox.

    For pure Python functions (no external calls), Monty executes the
    entire body without pausing. For functions that call other tasks or
    durable operations, ``run_monty_async`` handles async dispatch.
    """

    task_type: str = "sandboxed-python"
    plugin_config: Optional[SandboxedConfig] = None

    # Internal fields populated by __post_init__
    _source_code: str = field(default="", init=False, repr=False)
    _input_names: List[str] = field(default_factory=list, init=False, repr=False)
    _external_refs: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _has_external_refs: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        super().__post_init__()

        if self.plugin_config is None:
            self.plugin_config = SandboxedConfig()

        # Validate type annotations
        validate_sandboxed_interface(
            self.interface.inputs,
            self.interface.outputs,
        )

        # Extract source code
        self._source_code, self._input_names = extract_source(self.func)

        # Discover external references
        self._external_refs = _discover_external_refs(self.func)
        self._has_external_refs = bool(
            self._external_refs["task_refs"] or self._external_refs["trace_refs"] or self._external_refs["durable_refs"]
        )

    async def execute(self, *args, **kwargs) -> Any:
        """Execute the function body in a Monty sandbox."""
        from flyte._context import internal_ctx

        ctx = internal_ctx()
        assert ctx.data.task_context is not None, "Function should have already returned if not in a task context"

        ctx_data = await self.pre(*args, **kwargs)
        tctx = ctx.data.task_context.replace(data=ctx_data)
        with ctx.replace_task_context(tctx):
            result = await self._run_sandboxed(*args, **kwargs)
            await self.post(result)
        return result

    async def _run_sandboxed(self, *args, **kwargs) -> Any:
        """Core sandbox execution logic."""
        Monty = _lazy_import_monty()

        # Build the inputs dict from args/kwargs
        inputs = self._build_inputs(*args, **kwargs)

        if not self._has_external_refs:
            # Fast path: pure Python, no external calls
            monty = Monty(self._source_code, inputs=self._input_names)
            return monty.run(inputs=inputs)
        else:
            # External calls: use run_monty_async for async dispatch
            from ._bridge import ExternalFunctionBridge

            bridge = ExternalFunctionBridge(**self._external_refs)
            return await bridge.execute_monty(Monty, self._source_code, self._input_names, inputs)

    def _build_inputs(self, *args, **kwargs) -> Dict[str, Any]:
        """Build an inputs dict mapping parameter names to values."""
        inputs: Dict[str, Any] = {}
        for i, name in enumerate(self._input_names):
            if i < len(args):
                inputs[name] = args[i]
            elif name in kwargs:
                inputs[name] = kwargs[name]
        return inputs

    def forward(self, *args, **kwargs) -> Any:
        """Bypass Monty and call the function directly (for local/debug execution)."""
        return self.func(*args, **kwargs)
