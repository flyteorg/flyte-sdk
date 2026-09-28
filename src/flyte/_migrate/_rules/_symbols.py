from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import libcst as cst

from .._cst import Rule, call_arguments, code, kwargs_code
from ._launch_plan import interval_minutes

#: flytekit symbol -> (module to import from, replacement name). A module of None means `import flyte` + dotted name.
_RENAMES: Dict[str, Tuple[Optional[str], str]] = {
    "FlyteFile": ("flyte.io", "File"),
    "FlyteDirectory": ("flyte.io", "Dir"),
    "StructuredDataset": ("flyte.io", "DataFrame"),
    "HashMethod": ("flyte.io", "HashFunction"),
    "FlytePickle": ("flyte.types", "FlytePickle"),
    "Resources": (None, "flyte.Resources"),
    "Secret": (None, "flyte.Secret"),
    "PodTemplate": (None, "flyte.PodTemplate"),
    "CronSchedule": (None, "flyte.Cron"),
    "FixedRate": (None, "flyte.FixedRate"),
    "current_context": (None, "flyte.ctx"),
    "logger": (None, "flyte.logger"),
}

_CONTEXT_TODO = (
    "flyte.ctx() returns a TaskContext whose fields differ from flytekit's ExecutionParameters; "
    "secrets are exposed as environment variables in v2"
)


class SymbolsRule(Rule):
    """One-to-one renames of flytekit classes and functions, including their argument names."""

    id = "symbols"
    description = "FlyteFile/FlyteDirectory/StructuredDataset/Resources/Secret/... -> flyte equivalents"

    def visit_Import(self, node: cst.Import) -> bool:
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        return False

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        # `flytekit.types.file.FlyteFile` is replaced as a whole; do not visit its parts.
        return self.flytekit_symbol(node) not in _RENAMES

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
        return self._rename(original_node) or updated_node

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
        return self._rename(original_node) or updated_node

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        symbol = self.flytekit_symbol(original_node.func)
        if symbol == "Resources":
            return self._call(updated_node, "flyte.Resources", ("cpu", "mem", "gpu", "ephemeral_storage"), _resources)
        if symbol == "Secret":
            return self._call(
                updated_node,
                "flyte.Secret",
                ("group", "key", "group_version", "mount_requirement", "env_var"),
                self._secret,
            )
        if symbol == "CronSchedule":
            return self._call(
                updated_node,
                "flyte.Cron",
                ("cron_expression", "schedule", "offset", "kickoff_time_input_arg"),
                self._cron,
            )
        if symbol == "FixedRate":
            return self._call(updated_node, "flyte.FixedRate", ("duration", "kickoff_time_input_arg"), self._fixed_rate)
        if symbol in ("FlyteFile", "FlyteDirectory"):
            return self._local_path(updated_node, "File" if symbol == "FlyteFile" else "Dir")
        if symbol == "current_context":
            self.todo(_CONTEXT_TODO)
        return updated_node

    def _rename(self, node: Union[cst.Name, cst.Attribute]) -> Optional[cst.BaseExpression]:
        symbol = self.flytekit_symbol(node)
        if symbol not in _RENAMES:
            return None
        module, replacement = _RENAMES[symbol]
        if module is None:
            self.ctx.require_import("flyte")
        else:
            self.ctx.require_import(module, replacement)
        self.applied(symbol)
        return cst.parse_expression(replacement)

    def _call(self, call: cst.Call, func: str, positional: Tuple[str, ...], convert) -> cst.BaseExpression:
        kwargs, leftovers = call_arguments(call, positional)
        if leftovers:
            self.todo(f"review the arguments of `{func}`; positional or star arguments were not converted")
            return call
        pairs: List[Tuple[str, str]] = convert({k: code(v) for k, v in kwargs.items()})
        return cst.parse_expression(f"{func}({kwargs_code(pairs)})")

    def _local_path(self, updated: cst.Call, cls: str) -> cst.BaseExpression:
        """`FlyteFile("model.pkl")` -> `File.from_local_sync("model.pkl")`: v2 uploads local paths explicitly."""
        kwargs, leftovers = call_arguments(updated, ("path", "downloader", "remote_path"))
        if leftovers or "path" not in kwargs or set(kwargs) - {"path", "remote_path"}:
            self.todo(
                f"review this {cls}: construct it with {cls}.from_local_sync(...) or {cls}.from_existing_remote(...)"
            )
            return updated
        remote = f", remote_destination={code(kwargs['remote_path'])}" if "remote_path" in kwargs else ""
        return cst.parse_expression(f"{cls}.from_local_sync({code(kwargs['path'])}{remote})")

    def _secret(self, kwargs: Dict[str, str]) -> List[Tuple[str, str]]:
        pairs = [(k, kwargs[k]) for k in ("key", "group") if k in kwargs]
        if "env_var" in kwargs:
            pairs.append(("as_env_var", kwargs["env_var"]))
        for key in ("group_version", "mount_requirement"):
            if key in kwargs:
                self.todo(f"Secret `{key}` has no v2 equivalent and was dropped; use `mount=` for file secrets")
        return pairs

    def _cron(self, kwargs: Dict[str, str]) -> List[Tuple[str, str]]:
        expression = kwargs.get("schedule", kwargs.get("cron_expression"))
        pairs = [("expression", expression)] if expression is not None else []
        if "offset" in kwargs:
            self.todo("CronSchedule `offset` has no v2 equivalent and was dropped")
        if "kickoff_time_input_arg" in kwargs:
            self.todo("bind the kickoff time with `inputs={<arg>: flyte.TriggerTime}` on the flyte.Trigger")
        return pairs

    def _fixed_rate(self, kwargs: Dict[str, str]) -> List[Tuple[str, str]]:
        pairs = []
        if "duration" in kwargs:
            pairs.append(("interval_minutes", interval_minutes(cst.parse_expression(kwargs["duration"]))))
        if "kickoff_time_input_arg" in kwargs:
            self.todo("bind the kickoff time with `inputs={<arg>: flyte.TriggerTime}` on the flyte.Trigger")
        return pairs


def _resources(kwargs: Dict[str, str]) -> List[Tuple[str, str]]:
    renames = {"cpu": "cpu", "mem": "memory", "gpu": "gpu", "ephemeral_storage": "disk"}
    return [(renames[k], v) for k, v in kwargs.items() if k in renames]
