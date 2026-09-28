"""Argument conversions shared by several rules. Each function returns generated v2 source code."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import libcst as cst

from ._cst import Rule, call_arguments, code, kwargs_code, literal

#: Positional parameter order of `flytekit.Resources`.
_V1_RESOURCE_FIELDS = ("cpu", "mem", "gpu", "ephemeral_storage")
_V1_TO_V2_RESOURCE_FIELD = {"cpu": "cpu", "mem": "memory", "gpu": "gpu", "ephemeral_storage": "disk"}

#: flytekit accelerator constants (`flytekit.extras.accelerators`) to v2 GPU type names.
_ACCELERATORS = {
    "A10G": "A10G",
    "A100": "A100",
    "A100_80GB": "A100 80G",
    "H100": "H100",
    "H200": "H200",
    "L4": "L4",
    "L40S": "L40s",
    "RTX_PRO_6000": "RTX PRO 6000",
    "T4": "T4",
    "V100": "V100",
}


def v1_resource_fields(rule: Rule, node: cst.BaseExpression) -> Optional[Dict[str, cst.BaseExpression]]:
    """The v2-named fields of a literal `flytekit.Resources(...)` call, or None if `node` is not one."""
    if not isinstance(node, cst.Call) or not rule.is_flytekit(node.func, "Resources"):
        return None
    kwargs, _ = call_arguments(node, _V1_RESOURCE_FIELDS)
    return {_V1_TO_V2_RESOURCE_FIELD[k]: v for k, v in kwargs.items() if k in _V1_TO_V2_RESOURCE_FIELD}


def merge_resources(rule: Rule, args: Dict[str, cst.BaseExpression]) -> Optional[str]:
    """
    Merge flytekit's `requests`, `limits`, `resources`, `accelerator` and `shared_memory` into one `flyte.Resources`.

    Returns None when none of those arguments is present.
    """
    requests = args.get("requests", args.get("resources"))
    limits = args.get("limits")
    accelerator = args.get("accelerator")
    shared_memory = args.get("shared_memory")
    if requests is None and limits is None and accelerator is None and shared_memory is None:
        return None

    request_fields = v1_resource_fields(rule, requests) if requests is not None else {}
    limit_fields = v1_resource_fields(rule, limits) if limits is not None else {}
    if request_fields is None or limit_fields is None:
        opaque = requests if request_fields is None else limits
        if limits is None or requests is None:
            # A single non-literal value (e.g. a module-level `Resources` variable) is converted where it is defined.
            if accelerator is not None or shared_memory is not None:
                rule.todo("could not merge accelerator/shared_memory into non-literal resources; review `resources=`")
            return code(opaque)  # type: ignore[arg-type]
        rule.todo("could not merge non-literal requests and limits; review `resources=`")
        return code(opaque)  # type: ignore[arg-type]

    rule.ctx.require_import("flyte")
    merged: List[Tuple[str, str]] = []
    for field in ("cpu", "memory"):
        req, lim = request_fields.get(field), limit_fields.get(field)
        if req is not None and lim is not None and code(req) != code(lim):
            merged.append((field, f"({code(req)}, {code(lim)})"))
        elif req is not None or lim is not None:
            merged.append((field, code(req if req is not None else lim)))  # type: ignore[arg-type]

    gpu = limit_fields.get("gpu", request_fields.get("gpu"))
    count = _gpu_count(gpu) if gpu is not None else 1
    if accelerator is not None:
        gpu_type = _ACCELERATORS.get(rule.flytekit_symbol(accelerator) or "")
        if gpu_type is None or count is None:
            rule.todo(f"could not convert accelerator `{code(accelerator)}`; set `gpu=` on flyte.Resources")
            if gpu is not None:
                merged.append(("gpu", code(gpu)))
        else:
            merged.append(("gpu", repr(f"{gpu_type}:{count}")))
    elif gpu is not None:
        merged.append(("gpu", repr(count) if count is not None else code(gpu)))

    disk = limit_fields.get("disk", request_fields.get("disk"))
    if disk is not None:
        merged.append(("disk", code(disk)))

    if shared_memory is not None:
        value = literal(shared_memory)
        merged.append(("shm", '"auto"' if value is True else code(shared_memory)))

    return f"flyte.Resources({kwargs_code(merged)})"


def _gpu_count(node: cst.BaseExpression) -> Optional[int]:
    """flytekit accepts GPU counts as ints or digit strings; v2 wants an int."""
    value = literal(node)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def convert_cache(rule: Rule, args: Dict[str, cst.BaseExpression]) -> Optional[str]:
    """Convert flytekit's `cache`, `cache_version`, `cache_serialize` and `cache_ignore_input_vars`."""
    cache = args.get("cache")
    if cache is None:
        return None
    value = literal(cache)
    if value is False:
        return None

    fields: List[Tuple[str, str]] = []
    if value is True:
        if "cache_version" in args:
            fields.append(("version_override", code(args["cache_version"])))
        if "cache_serialize" in args:
            fields.append(("serialize", code(args["cache_serialize"])))
        if "cache_ignore_input_vars" in args:
            fields.append(("ignored_inputs", code(args["cache_ignore_input_vars"])))
    elif isinstance(cache, cst.Call) and rule.is_flytekit(cache.func, "Cache"):
        kwargs, _ = call_arguments(cache, ("version", "serialize", "ignored_inputs", "salt", "policies"))
        renames = {"version": "version_override", "serialize": "serialize", "ignored_inputs": "ignored_inputs"}
        for key, node in kwargs.items():
            if key in renames:
                fields.append((renames[key], code(node)))
            elif key == "salt":
                fields.append(("salt", code(node)))
            else:
                rule.todo(f"cache `{key}` is not converted; v2 cache policies use flyte.CachePolicy")
    else:
        rule.todo(f"could not convert `cache={code(cache)}`; use flyte.Cache(...)")
        return None

    if not fields:
        return '"auto"'
    rule.ctx.require_import("flyte")
    return f'flyte.Cache(behavior="auto", {kwargs_code(fields)})'


#: `with_overrides` arguments that keep their name in `TaskTemplate.override`.
_OVERRIDE_PASSTHROUGH = ("retries", "timeout", "interruptible", "pod_template")
_OVERRIDE_RESOURCES = ("requests", "limits", "resources", "accelerator", "shared_memory")


def override_arguments(rule: Rule, args: Dict[str, cst.BaseExpression]) -> str:
    """Convert `with_overrides(...)` keyword arguments to `TaskTemplate.override(...)` keyword arguments."""
    converted: List[Tuple[str, str]] = []
    resources = merge_resources(rule, args)
    if resources is not None:
        converted.append(("resources", resources))
    for key, node in args.items():
        if key in _OVERRIDE_RESOURCES:
            continue
        if key in _OVERRIDE_PASSTHROUGH:
            converted.append((key, code(node)))
        elif key == "cache":
            value = literal(node)
            if value is True or value is False:
                converted.append(("cache", '"auto"' if value else '"disable"'))
            else:
                cache = convert_cache(rule, {"cache": node})
                if cache is not None:
                    converted.append(("cache", cache))
        elif key in ("name", "node_name"):
            converted.append(("short_name", code(node)))
        elif key == "task_config":
            converted.append(("plugin_config", code(node)))
            rule.todo("plugin configuration types differ in v2; replace with the flyteplugins equivalent")
        elif key == "container_image":
            rule.todo("v2 cannot override the image per call; call a task from an environment with that image")
        else:
            rule.todo(f"`with_overrides({key}=...)` has no v2 equivalent and was dropped")
    return kwargs_code(converted)


def convert_map(rule: Rule, call: cst.Call, *, is_async: bool) -> Optional[str]:
    """
    Convert `map_task(target, ...)(x=xs, ...)` to `flyte.map`.

    Returns `list(flyte.map(...))` in sync code, or `flyte.map.aio(...)` (an async generator) in async code.
    Returns None if `call` is not a map task call.
    """
    inner = call.func
    if not isinstance(inner, cst.Call) or not rule.is_flytekit(inner.func, "map_task"):
        return None
    options, _ = call_arguments(inner, ("target", "concurrency", "min_successes", "min_success_ratio"))
    target = options.pop("target", None)
    mapped, positional = call_arguments(call)
    if target is None or positional:
        return None

    params, bound = _target_params(rule, target)
    if params is None:
        order = list(mapped)
        if len(order) > 1:
            rule.todo("verify the iterables passed to flyte.map follow the target task's parameter order")
    else:
        order = [p for p in params if p in mapped and p not in bound]
        unknown = [k for k in mapped if k not in order]
        if unknown:
            rule.todo(f"mapped inputs {unknown} do not match the target's parameters")
            order.extend(unknown)

    extra: List[Tuple[str, str]] = []
    if "concurrency" in options:
        extra.append(("concurrency", code(options.pop("concurrency"))))
    tolerant = "min_successes" in options or "min_success_ratio" in options
    options.pop("min_successes", None)
    options.pop("min_success_ratio", None)
    if tolerant:
        rule.todo("failed map items are returned as exceptions in v2; filter them to honor min_success(es/_ratio)")
    extra.append(("return_exceptions", "True" if tolerant else "False"))
    for key in options:
        rule.todo(f"map_task `{key}` has no v2 equivalent and was dropped")

    rule.ctx.require_import("flyte")
    arguments = ", ".join([code(target), *(code(mapped[k]) for k in order), kwargs_code(extra)])
    if is_async:
        return f"flyte.map.aio({arguments})"
    return f"list(flyte.map({arguments}))"


def _target_params(rule: Rule, target: cst.BaseExpression) -> Tuple[Optional[List[str]], List[str]]:
    """The parameter names of a map target and the names bound by `functools.partial`."""
    bound: List[str] = []
    if isinstance(target, cst.Call) and rule.qualified_name(target.func) == "functools.partial" and target.args:
        bound = [a.keyword.value for a in target.args[1:] if a.keyword is not None]
        target = target.args[0].value
    if isinstance(target, cst.Name) and target.value in rule.ctx.entities:
        return rule.ctx.entities[target.value].params, bound
    return None, bound
