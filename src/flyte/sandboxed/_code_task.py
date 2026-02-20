"""CodeTaskTemplate — a sandboxed task created from a code string."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from flyte._task import AsyncFunctionTaskTemplate

from ._config import SandboxedConfig
from ._task import SandboxedTaskTemplate
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

        # Populate the inherited internal fields from our init fields.
        # Monty requires at least one input, so inject a dummy when the
        # user didn't provide any.  The dummy is internal-only — it never
        # appears in the task's NativeInterface, so callers don't see it.
        self._source_code = self._user_source
        self._needs_dummy_input = not self._user_input_names
        if self._needs_dummy_input:
            self._input_names = ["_unused"]
        else:
            self._input_names = list(self._user_input_names)

        # Build _external_refs from the explicit functions dict
        self._external_refs = _classify_refs(self._user_functions)
        self._has_external_refs = bool(self._user_functions)

    def _build_inputs(self, *args, **kwargs) -> Dict[str, Any]:
        """Build inputs dict, injecting the dummy when no user inputs exist."""
        if self._needs_dummy_input:
            return {"_unused": 0}
        return super()._build_inputs(*args, **kwargs)

    def forward(self, *args, **kwargs) -> Any:
        """Not supported — there is no Python function to call directly."""
        raise NotImplementedError(
            "CodeTaskTemplate does not support forward(). Use flyte.run() to execute through the sandbox."
        )
