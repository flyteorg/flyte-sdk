from __future__ import annotations

import re
from typing import List, Optional

import libcst as cst

from .._cst import Rule, arguments_code, call_arguments, code, literal

_SPREAD_METHODS = {"with_packages": "with_pip_packages", "with_apt_packages": "with_apt_packages"}
_HANDLED = {
    "name",
    "registry",
    "python_version",
    "base_image",
    "packages",
    "apt_packages",
    "requirements",
    "env",
    "commands",
    "pip_index",
    "pip_extra_index_url",
    "platform",
}


class ImageRule(Rule):
    """Convert `ImageSpec(...)` to a `flyte.Image` builder chain."""

    id = "image"
    description = "ImageSpec -> flyte.Image"

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        # `ImageSpec(...).with_packages([...])` chains: rewrite the method once the receiver is converted.
        func = original_node.func
        if (
            isinstance(func, cst.Attribute)
            and func.attr.value in (*_SPREAD_METHODS, "with_commands")
            and isinstance(func.value, cst.Call)
            and self.is_flytekit(func.value.func, "ImageSpec")
        ):
            receiver = updated_node.func.value  # type: ignore[attr-defined]
            if func.attr.value == "with_commands":
                return cst.parse_expression(f"{code(receiver)}.with_commands({arguments_code(original_node)})")
            if len(original_node.args) != 1:
                self.todo(f"review `{func.attr.value}(...)`; flyte.Image takes packages as separate arguments")
                return updated_node
            method = _SPREAD_METHODS[func.attr.value]
            return cst.parse_expression(f"{code(receiver)}.{method}({_spread(original_node.args[0].value)})")

        if not self.is_flytekit(original_node.func, "ImageSpec"):
            return updated_node
        converted = self._convert(original_node)
        if converted is None:
            return updated_node
        self.applied("ImageSpec")
        self.ctx.require_import("flyte")
        return cst.parse_expression(converted)

    def _convert(self, call: cst.Call) -> Optional[str]:
        args, positional = call_arguments(call, ("name", "python_version", "builder", "source_root", "env", "registry"))
        if positional:
            return None

        base = args.get("base_image")
        if base is not None:
            if not isinstance(literal(base), str):
                self.todo("only string `base_image` values are converted; nested ImageSpec bases are not")
                return None
            chain = f"flyte.Image.from_base({code(base)})"
            clone = [(k, code(args[k])) for k in ("registry", "name") if k in args]
            if clone:
                chain += f".clone({', '.join(f'{k}={v}' for k, v in clone)})"
            if "python_version" in args:
                self.todo("`python_version` is ignored for custom base images; the base image's Python is used")
        else:
            options = [(k, code(args[k])) for k in ("registry", "name") if k in args]
            if "python_version" in args:
                version = _python_version(args["python_version"])
                if version is None:
                    self.todo(f"could not convert python_version={code(args['python_version'])}; pass a (major, minor)")
                else:
                    options.insert(0, ("python_version", version))
            if "platform" in args:
                platform = _platform(args["platform"])
                if platform is None:
                    self.todo('could not convert `platform`; pass platform=("linux/amd64", ...)')
                else:
                    options.append(("platform", platform))
            chain = f"flyte.Image.from_debian_base({', '.join(f'{k}={v}' for k, v in options)})"

        if "apt_packages" in args:
            chain += f".with_apt_packages({_spread(args['apt_packages'])})"
        if "packages" in args:
            pip_options = ""
            if "pip_index" in args:
                pip_options += f", index_url={code(args['pip_index'])}"
            if "pip_extra_index_url" in args:
                pip_options += f", extra_index_urls={code(args['pip_extra_index_url'])}"
            chain += f".with_pip_packages({_spread(args['packages'])}{pip_options})"
        if "requirements" in args:
            chain += f".with_requirements({code(args['requirements'])})"
        if "env" in args:
            chain += f".with_env_vars({code(args['env'])})"
        if "commands" in args:
            chain += f".with_commands({code(args['commands'])})"

        for key in args:
            if key not in _HANDLED:
                self.todo(f"ImageSpec `{key}` was not converted; see flyte.Image for the v2 equivalent")
        return chain


def _spread(node: cst.BaseExpression) -> str:
    if isinstance(node, (cst.List, cst.Tuple)):
        return ", ".join(code(e.value) for e in node.elements)
    return f"*{code(node)}"


def _python_version(node: cst.BaseExpression) -> Optional[str]:
    value = literal(node)
    match = re.fullmatch(r"(\d+)\.(\d+)(?:\.\d+)?", value) if isinstance(value, str) else None
    return f"({match.group(1)}, {match.group(2)})" if match else None


def _platform(node: cst.BaseExpression) -> Optional[str]:
    value = literal(node)
    if not isinstance(value, str):
        return None
    platforms: List[str] = [p.strip() for p in value.split(",") if p.strip()]
    return "(" + ", ".join(repr(p) for p in platforms) + ",)"
