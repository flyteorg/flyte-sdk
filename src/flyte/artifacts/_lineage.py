"""
Deploy-time extraction of lineage declarations.

Reads `produces_artifacts=`, `consumes_artifacts=` and `labels=` off tasks (and apps), validates them against
the signature, and compiles them to the reserved labels the platform stores on the deployed entity:

- `lineage.produces`: comma-separated node ids the entity writes
- `lineage.consumes`: comma-separated node ids the entity reads
- `lineage.bindings`: JSON payload (version 1) describing each parameter's binding and each handle record

Everything here is pure: no backend, no protos. `get_proto_task` and the app serializer call it, `flyte.deploy`
calls `check_conflicts` across the whole deploy, and the CLI prints `summarize`.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import typing
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from flyte._logging import logger
from flyte.errors import LineageDeclarationError

from ._handle import (
    LINEAGE_IDENT_RULE,
    MAX_DESCRIPTION_BYTES,
    MAX_SHORT_TEXT_BYTES,
    MAX_SRC_FILE_BYTES,
    MAX_TYPE_BYTES,
    PROJECT_NAME_RE,
    Artifact,
    ArtifactMapping,
    OutputPartition,
    PartitionValue,
    RequiredParam,
    _Granularity,
    dimension_kind,
    is_handle,
    is_lineage_ident,
    record_source_path,
    relative_source_path,
    type_name,
    valid_node_id,
)

if TYPE_CHECKING:
    from flyte._task import TaskTemplate

__all__ = [
    "BINDINGS_LABEL",
    "CONSUMES_LABEL",
    "MANAGED_LABELS_KEY",
    "PRODUCES_LABEL",
    "HandleDeclaration",
    "LineageSummary",
    "TaskLineage",
    "app_env_lineage_labels",
    "app_lineage_labels",
    "check_conflicts",
    "declared_output_metadata",
    "extract_task_lineage",
    "iter_tasks",
    "merge_labels",
    "summarize",
    "task_lineage_tags",
    "validate_deploy_labels",
    "validate_labels",
]

PRODUCES_LABEL = "lineage.produces"
#: Label listing the keys the SDK wrote on an app, so a redeploy drops the ones it no longer declares while
#: keeping labels set outside the SDK.
MANAGED_LABELS_KEY = "flyte.io/managed-labels"
CONSUMES_LABEL = "lineage.consumes"
BINDINGS_LABEL = "lineage.bindings"
LINEAGE_PREFIX = "lineage."
BINDINGS_VERSION = 1
#: Backend limits (past them the backend stores the entity without its lineage; the deploy itself succeeds, so
#: the SDK checks them up front): `lineage.bindings` at most 64 KiB (the SDK sheds
#: detail past a soft limit with headroom), `lineage.produces`/`lineage.consumes` values at most 4 KiB, and per
#: entity at most 256 produced ids, 256 consumed ids, and consumes x produces at most 4096.
BINDINGS_MAX_BYTES = 65536
BINDINGS_SOFT_LIMIT = 60 * 1024
MAX_EDGE_LABEL_BYTES = 4096
MAX_EDGE_IDS = 256
MAX_EDGE_PAIRS = 4096
#: Backend limits on every label of a deployed entity (user-authored and SDK-written alike; checked up front, as
#: the backend would drop the entity's lineage or labels past them): keys at most 256 bytes,
#: values at most 4 KiB (except `lineage.bindings`, capped separately above), at most 64 labels per entity.
MAX_LABEL_KEY_BYTES = 256
MAX_LABEL_VALUE_BYTES = 4096
MAX_LABELS = 64

#: How much a task declares, recorded as `level` in its bindings: 0 nothing typed (publish-only), 3 produces a
#: handle, 4 consumes a handle, 5 a produced handle carries expected values.
LEVEL_PUBLISH_ONLY = 0


# --------------------------------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------------------------------


def split_node_ids(value: str) -> List[str]:
    """`"a, b,c"` -> `["a", "b", "c"]`; empty entries dropped."""
    return [v.strip() for v in value.split(",") if v.strip()]


def _union(*groups: Iterable[str]) -> List[str]:
    out: Dict[str, None] = {}
    for g in groups:
        out.update(dict.fromkeys(g))
    return list(out)


def validate_labels(labels: Optional[Mapping[str, str]], *, entity: str, where: str) -> None:
    """
    Enforce the reserved-namespace rule on user-authored labels.

    Only `lineage.consumes` (and, on a task, `lineage.produces`) may be hand-written inside `lineage.`; their
    values are comma-separated node ids (`[A-Za-z0-9_.:-]+`).

    Args:
        labels: The labels to check.
        entity: `"task"` or `"app"`.
        where: Name of the entity, for the error message.
    """
    allowed = (CONSUMES_LABEL, PRODUCES_LABEL) if entity == "task" else (CONSUMES_LABEL,)
    for key, value in (labels or {}).items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise LineageDeclarationError(f"{where}: labels must be str to str, got {key!r}: {value!r}")
        _check_label_size(key, value, where)
        if not key.startswith(LINEAGE_PREFIX):
            continue
        if key == PRODUCES_LABEL and entity == "app":
            raise LineageDeclarationError(
                f"{where}: an app may not set '{PRODUCES_LABEL}'; deploy derives it as 'app:<app name>' (the "
                "app's endpoint)."
            )
        if key not in allowed:
            raise LineageDeclarationError(
                f"{where}: label {key!r} is in the reserved 'lineage.' namespace; only "
                f"{' and '.join(repr(a) for a in allowed)} may be written by hand."
            )
        for node in split_node_ids(value):
            if not valid_node_id(node):
                raise LineageDeclarationError(
                    f"{where}: {key}={value!r} contains an invalid node id {node!r}; node ids are letters, digits, "
                    "'_', '.', ':' and '-' (an 'app:', 'task:' or 'trigger:' id may also hold '/', after an optional "
                    "'<project>/'; 'hidden:' is reserved), separated by commas."
                )
        check_edge_limits({key: value}, where)  # value size and id count


def _check_label_size(key: str, value: str, where: str) -> None:
    if not key:
        raise LineageDeclarationError(f"{where}: label keys must be non-empty")
    ksize = len(key.encode("utf-8"))
    if ksize > MAX_LABEL_KEY_BYTES:
        raise LineageDeclarationError(
            f"{where}: label key {key[:40]!r}... is {ksize} bytes; the limit is {MAX_LABEL_KEY_BYTES}."
        )
    vsize = len(value.encode("utf-8"))
    if key != BINDINGS_LABEL and vsize > MAX_LABEL_VALUE_BYTES:
        raise LineageDeclarationError(
            f"{where}: the value of label {key!r} is {vsize} bytes; the limit is {MAX_LABEL_VALUE_BYTES}."
        )


def check_label_count(labels: Mapping[str, str], where: str, *, reserved: int = 0) -> None:
    """
    At most `MAX_LABELS` labels per entity, counting the ones the SDK writes (`lineage.*`, and `reserved` more
    it adds later, such as an app's managed-labels key).
    """
    n = len(labels) + reserved
    if n > MAX_LABELS:
        raise LineageDeclarationError(
            f"{where}: {n} labels (including {reserved + sum(1 for k in labels if k.startswith(LINEAGE_PREFIX))} "
            f"written by flyte); at most {MAX_LABELS} are allowed per entity. Remove some labels."
        )


def validate_deploy_labels(labels: Optional[Mapping[str, str]]) -> None:
    """
    Deploy-wide labels (`flyte deploy --label`, `flyte.deploy(labels=)`) apply to every task and app, so they
    may not set `lineage.produces`: that would claim every entity writes the same node.
    """
    if labels and PRODUCES_LABEL in labels:
        raise LineageDeclarationError(
            f"'{PRODUCES_LABEL}' cannot be a deploy-wide label: it would apply to every task and app in the deploy. "
            f"Set it on the one task that writes the node, e.g. @env.task(labels={{'{PRODUCES_LABEL}': ...}})."
        )
    validate_labels(labels, entity="task", where="flyte deploy --label")


def merge_labels(*layers: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """Later layers win per key, except `lineage.consumes` / `lineage.produces`, whose node ids are unioned."""
    out: Dict[str, str] = {}
    for layer in layers:
        for k, v in (layer or {}).items():
            if k in (CONSUMES_LABEL, PRODUCES_LABEL) and k in out:
                out[k] = ",".join(_union(split_node_ids(out[k]), split_node_ids(v)))
            else:
                out[k] = v
    return out


# --------------------------------------------------------------------------------------------------
# Task extraction
# --------------------------------------------------------------------------------------------------


@dataclass
class TaskLineage:
    """
    The lineage a task declares, as deploy writes it.

    Attributes:
        task: The task's fully-qualified name.
        src_file: The file that declares the task, relative to the deploy root.
        src_line: The line of the task's decorator.
        produces: Node ids written (`lineage.produces`).
        consumes: Node ids read (`lineage.consumes`).
        bindings: The `lineage.bindings` payload, or None for a task without typed declarations.
        labels: Every tag written, lineage and user labels alike.
        level: How much the task declares: 0 nothing typed, 3 produces a handle, 4 consumes one, 5 a produced
            handle carries expected values.
        pullable: Whether the planner can build this task's outputs from bindings and defaults alone.
        unpullable_reason: Why not, when `pullable` is False.
        handles: The typed handles the declarations reference, keyed by name.
    """

    task: str
    src_file: str = ""
    src_line: int = 0
    produces: List[str] = field(default_factory=list)
    consumes: List[str] = field(default_factory=list)
    bindings: Optional[Dict[str, Any]] = None
    labels: Dict[str, str] = field(default_factory=dict)
    level: int = 0
    pullable: bool = False
    unpullable_reason: str = ""
    unpullable_params: List[str] = field(default_factory=list)
    unpullable_messages: List[str] = field(default_factory=list)
    handles: Dict[str, Artifact] = field(default_factory=dict)

    @property
    def tags(self) -> Dict[str, str]:
        """Alias of `labels`: what goes into `TaskTemplate.metadata.tags`."""
        return self.labels

    @property
    def typed(self) -> bool:
        """True when the task used typed declarations (a produces tuple and/or consumes_artifacts)."""
        return self.bindings is not None

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """`consumes x produces` for this entity."""
        return [(c, p) for c in self.consumes for p in self.produces]

    @property
    def resolvable_edges(self) -> List[Tuple[str, str]]:
        """Edges whose both endpoints are typed handle records in this task's bindings."""
        if not self.typed:
            return []
        return [(c, p) for c, p in self.edges if c != p and c in self.handles and p in self.handles]

    def warnings(self) -> List[str]:
        """The deploy-time warnings for this task: one per uncovered parameter or unbound produced dimension."""
        if self.pullable or not self.produces:
            return []
        return list(self.unpullable_messages)


def _unpullable_message(task: str, param: str, produces: Sequence[str]) -> str:
    what = ", ".join(produces) or "the task's inputs"
    return (
        f"{task} is not pullable: parameter {param!r} has no default, no binding, and is not named like a "
        f"partition dimension of {what}. Give it a default, bind it (artifacts.partition(dim) for a partition "
        "value), or mark it artifacts.required() so every materialization must supply it."
    )


def _unwrap_optional(t: Any) -> Tuple[Any, bool]:
    """`Optional[X]` / `X | None` -> (X, True); anything else -> (t, False)."""
    import types

    origin = typing.get_origin(t)
    if origin is typing.Union or (hasattr(types, "UnionType") and origin is getattr(types, "UnionType")):
        args = [a for a in typing.get_args(t) if a is not type(None)]
        if len(args) == 1 and len(typing.get_args(t)) == 2:
            return args[0], True
    return t, False


def _is_list_type(t: Any) -> bool:
    t, _ = _unwrap_optional(t)
    origin = typing.get_origin(t)
    return origin in (list, List) or t is list


def _is_artifactable_type(t: Any) -> bool:
    """Whether values annotated `t` can be published as artifacts (what `convert.py` accepts at run time)."""
    from flyte.io import DataFrame, Dir, File

    if not isinstance(t, type):
        return False
    if issubclass(t, (File, Dir, DataFrame)):
        return True
    return callable(getattr(t, "get_artifact_metadata", None))


def _dimension_accepts(kind: str, ptype: Any) -> bool:
    """Whether a parameter annotated `ptype` can carry a coordinate of a dimension of `kind`."""
    from datetime import date

    t, _ = _unwrap_optional(ptype)
    if not isinstance(t, type):
        return False
    if kind == "time":
        return issubclass(t, date)  # datetime is a date subclass
    if kind == "int":
        return issubclass(t, int) and not issubclass(t, bool)
    return issubclass(t, str)


_DIMENSION_TYPES = {"time": "datetime or date", "int": "int", "str": "str"}


def _default_record(tname: str, value: Any) -> Dict[str, Any]:
    """
    `{"kind": "default", "type", "default"}`; a value with no strict JSON form (including a non-finite float, which
    would encode as `NaN`/`Infinity`, invalid JSON for the backend) goes to `default_repr` instead.
    """
    if value is None or isinstance(value, (bool, int, str)) or (isinstance(value, float) and math.isfinite(value)):
        return {"kind": "default", "type": tname, "default": value}
    try:
        json.dumps(value, allow_nan=False)
        return {"kind": "default", "type": tname, "default": value}
    except (TypeError, ValueError):
        return {"kind": "default", "type": tname, "default_repr": repr(value)}


def _join_reasons(messages: Sequence[str], limit: int = MAX_DESCRIPTION_BYTES) -> str:
    """`"; ".join(messages)` within the backend's `limit` bytes: the first messages that fit, then "and N more"."""
    joined = "; ".join(messages)
    if len(joined.encode("utf-8")) <= limit:
        return joined
    kept: List[str] = []
    for i, m in enumerate(messages):
        tail = f"; and {len(messages) - i - 1} more" if i < len(messages) - 1 else ""
        candidate = "; ".join([*kept, m]) + tail
        if len(candidate.encode("utf-8")) > limit:
            break
        kept.append(m)
    rest = len(messages) - len(kept)
    if not kept:
        # Not even the first message fits: cut it.
        suffix = f"... and {rest - 1} more" if rest > 1 else "..."
        raw = messages[0].encode("utf-8")[: limit - len(suffix.encode("utf-8"))]
        return raw.decode("utf-8", "ignore") + suffix
    return "; ".join(kept) + f"; and {rest} more"


def _unbound_dim_message(task: str, handle: str, dim: str) -> str:
    return (
        f"{task} is not pullable: dimension {dim!r} of {handle} is not bound by any parameter. Name a parameter "
        f"{dim!r} (with no default) and it carries the value, or bind one with artifacts.partition({dim!r}) / "
        f"{handle}.get_partition_value({dim!r}); until then, direct runs publish {handle} only when the body returns "
        f"artifacts.new(value, {handle}.at({dim}=...))."
    )


def _instance_handles(
    produced: Sequence[Artifact], consumes_map: Mapping[str, Any]
) -> List[Tuple[Artifact, List[str]]]:
    """
    The handles whose dimensions are the instance's: each produced handle with all its dimensions; for a task
    that produces nothing (a sink), each input mapped by identity or `select`, minus the pinned dimensions.
    """
    if produced:
        return [(h, list(h.partitions)) for h in produced]
    out: List[Tuple[Artifact, List[str]]] = []
    for v in consumes_map.values():
        m = v.identity_mapping if is_handle(v) else v
        if isinstance(m, ArtifactMapping) and m.kind in ("identity", "select"):
            out.append((m.handle, [d for d in m.handle.partitions if d not in m.pinned]))
    return out


def _instance_dim(instance: Sequence[Tuple[Artifact, List[str]]], dim: str) -> Optional[Artifact]:
    """The first instance handle that carries `dim`, or None."""
    return next((h for h, dims in instance if dim in dims), None)


def implicit_partition_params(task: Any) -> Dict[str, Tuple[Artifact, str]]:
    """
    The implicit rule: each parameter with no default and no binding whose name is a dimension of the instance
    (a produced handle, or for a sink an identity/select-mapped input) and whose type can carry it, mapped to
    `(handle, dim)`. Deploy records these as `{"kind": "partition", ..., "implicit": true}`; run time reads the
    produced partition values from them.
    """
    produced_flag = getattr(task, "produces_artifacts", False)
    produced = [h for h in produced_flag if is_handle(h)] if isinstance(produced_flag, tuple) else []
    consumes_map: Mapping[str, Any] = getattr(task, "consumes_artifacts", None) or {}
    if not produced and not consumes_map:
        return {}
    instance = _instance_handles(produced, consumes_map)
    out: Dict[str, Tuple[Artifact, str]] = {}
    for pname, (ptype, default) in task.native_interface.inputs.items():
        if pname in consumes_map or default is not inspect.Parameter.empty:
            continue
        h = _instance_dim(instance, pname)
        if h is not None and _dimension_accepts(dimension_kind(h.partitions[pname]), ptype):
            out[pname] = (h, pname)
    return out


def _task_source(task: TaskTemplate, root_dir: Optional[str]) -> Tuple[str, int]:
    func = getattr(task, "func", None)
    declared = getattr(func, "_flyte_source", None)
    if declared is not None:
        # A generated task (a refresh policy): its location is where the policy was declared.
        return record_source_path(declared[0], root_dir), int(declared[1])
    code = getattr(func, "__code__", None)
    if code is None:
        return "", 0
    return record_source_path(code.co_filename, root_dir), int(code.co_firstlineno)


def _param_lines(task: TaskTemplate, default_line: int) -> Dict[str, int]:
    """Line of each parameter in the task function's signature; the decorator line where it cannot be read.

    Read once per function (source + AST of the function only) and cached by code object.
    """
    func = getattr(task, "func", None)
    code = getattr(func, "__code__", None)
    if func is None or code is None:
        return {}
    cached = _PARAM_LINES_CACHE.get(code)
    if cached is not None:
        return cached
    result = _read_param_lines(func)
    _PARAM_LINES_CACHE[code] = result
    return result


_PARAM_LINES_CACHE: "weakref.WeakKeyDictionary[Any, Dict[str, int]]" = weakref.WeakKeyDictionary()


def _read_param_lines(func: Any) -> Dict[str, int]:
    import ast
    import textwrap

    try:
        lines, start = inspect.getsourcelines(func)
        tree = ast.parse(textwrap.dedent("".join(lines)))
    except (OSError, TypeError, SyntaxError, IndentationError):
        return {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            return {
                arg.arg: start + arg.lineno - 1
                for arg in [*a.posonlyargs, *a.args, *a.kwonlyargs]
                if getattr(arg, "lineno", None)
            }
    return {}


def _handle_record(h: Artifact, root_dir: Optional[str]) -> Dict[str, Any]:
    return h.to_dict(src_file=record_source_path(h.src_path, root_dir))


def _signature(h: Artifact) -> Tuple[Tuple[Tuple[str, str, str], ...], str]:
    dims = tuple(
        (d, dimension_kind(t), t.registry if isinstance(t, _Granularity) else "") for d, t in h.partitions.items()
    )
    return dims, type_name(h.type)


def _where(h: Artifact, root_dir: Optional[str] = None) -> str:
    src = relative_source_path(h.src_path, root_dir) if h.src_path else h.src_file
    return f"{src}:{h.src_line}" if src else "<unknown>"


def _describe_signature(h: Artifact) -> str:
    dims = ", ".join(f"{d}: {g or k}" for d, k, g in _signature(h)[0])
    tn = type_name(h.type)
    return f"type={tn or 'unset'}, dims=[{dims}]"


def check_conflicts(handles: Iterable[Artifact], root_dir: Optional[str] = None) -> Dict[str, Artifact]:
    """
    The same artifact name declared with different dimensions or type fails, naming both files.

    A bare handle (no type, no partitions) is a name reference and never conflicts. A handle with no type
    is compatible with any type. An `Artifact.ref` is checked like any declaration, but the owner's handle,
    when the deploy has it too, is the one recorded.

    Returns:
        The first typed handle seen for each name (an owner's handle before a reference).
    """
    seen: Dict[str, Artifact] = {}
    for h in handles:
        if h.is_bare:
            continue
        prev = seen.get(h.name)
        if prev is None or prev is h:
            seen.setdefault(h.name, h)
            continue
        prev_dims, prev_type = _signature(prev)
        dims, tn = _signature(h)
        if prev_dims != dims or (prev_type and tn and prev_type != tn):
            raise LineageDeclarationError(
                f"artifact {h.name!r} is declared with different dimensions or type in {_where(prev, root_dir)} "
                f"({_describe_signature(prev)}) and {_where(h, root_dir)} ({_describe_signature(h)}). "
                "Every declaration of a name must agree."
            )
        if (prev.reference and not h.reference) or (not prev_type and tn and prev.reference == h.reference):
            seen[h.name] = h
    return seen


def _ladder_level(produced: Sequence[Artifact], consumes_artifacts: bool) -> int:
    if not produced and not consumes_artifacts:
        return LEVEL_PUBLISH_ONLY
    level = 3 if produced else 0
    if consumes_artifacts:
        level = 4
    if produced and any(h.expected for h in produced):
        level = 5
    return level


def extract_task_lineage(
    task: TaskTemplate,
    *,
    root_dir: Optional[str] = None,
    extra_labels: Optional[Mapping[str, str]] = None,
) -> TaskLineage:
    """
    Validate a task's declarations against its signature and compile them to lineage labels.

    Args:
        task: The task template.
        root_dir: Deploy root; source files are written relative to it.
        extra_labels: Labels applied to every deployed entity (`flyte deploy --label`).

    Returns:
        The `TaskLineage`; its `labels` are what goes into `TaskTemplate.metadata.tags`.

    Raises:
        LineageDeclarationError: on any invalid declaration.
    """
    name = task.name
    root = str(root_dir) if root_dir is not None else None
    user_labels = merge_labels(_env_labels(task), getattr(task, "labels", None), extra_labels)
    validate_labels(user_labels, entity="task", where=name)

    produces_flag = getattr(task, "produces_artifacts", False)
    slots: Tuple[Optional[Artifact], ...] = produces_flag if isinstance(produces_flag, tuple) else ()
    # (position, handle) for the declared positions; None entries are placeholders for undeclared outputs.
    positioned: List[Tuple[int, Artifact]] = [(i, h) for i, h in enumerate(slots) if h is not None]
    produced: Tuple[Artifact, ...] = tuple(h for _, h in positioned)
    consumes_map: Mapping[str, Any] = getattr(task, "consumes_artifacts", None) or {}
    src_file, src_line = _task_source(task, root)
    lineage = TaskLineage(task=name, src_file=src_file, src_line=src_line)

    hand_produces = split_node_ids(user_labels.get(PRODUCES_LABEL, ""))
    hand_consumes = split_node_ids(user_labels.get(CONSUMES_LABEL, ""))

    materialize_on = getattr(getattr(task, "func", None), "__dict__", {}).get("_flyte_materialize_on")
    if not isinstance(materialize_on, dict):
        materialize_on = None
    if not produced and not consumes_map:
        _check_triggers(task, name, strict=False)
        lineage.produces = hand_produces
        lineage.consumes = hand_consumes
        lineage.level = LEVEL_PUBLISH_ONLY
        if materialize_on is not None:
            _materialize_on_bindings(task, lineage, materialize_on, root)
        lineage.labels = _finalize_labels(user_labels, lineage)
        return lineage

    interface = task.native_interface
    inputs: Mapping[str, Tuple[Any, Any]] = interface.inputs
    outputs: Mapping[str, Any] = interface.outputs
    # Every parameter is recorded in the bindings, and the backend refuses (drops the task's lineage on) a name
    # outside its identifier rule.
    for pname in inputs:
        if not is_lineage_ident(pname):
            raise LineageDeclarationError(
                f"{name}: parameter {pname!r} cannot carry lineage: a task that declares produces_artifacts or "
                f"consumes_artifacts needs parameter names that are {LINEAGE_IDENT_RULE}. Rename the parameter."
            )
    _check_triggers(task, name, strict=True)

    # -- produces ----------------------------------------------------------------------------------
    for h in slots:
        if h is not None and not is_handle(h):
            raise LineageDeclarationError(
                f"{name}: produces_artifacts entries must be flyte.artifacts.Artifact handles (or None placeholders)"
            )
    if slots and len(slots) != len(outputs):
        declared = ", ".join(h.name if h is not None else "None" for h in slots)
        raise LineageDeclarationError(
            f"{name}: produces_artifacts declares {len(slots)} position(s) ({declared}) but the task returns "
            f"{len(outputs)} value(s); a tuple return is matched to the handles by position. Use None for an "
            "output that is not an artifact, e.g. produces_artifacts=(None, model)."
        )
    for h in produced:
        if h.reference:
            raise LineageDeclarationError(
                f"{name}: produces_artifacts names {h.name!r}, a reference (Artifact.ref) to an artifact owned "
                f"elsewhere; a reference can only be read. To produce {h.name!r} here as well, declare it with "
                "artifacts.Artifact(...) (the lineage graph then shows both producers)."
            )
    if len({h.name for h in produced}) != len(produced):
        names = [h.name for h in produced]
        dup = next(n for n in names if names.count(n) > 1)
        raise LineageDeclarationError(
            f"{name}: produces_artifacts names artifact {dup!r} twice; each output position needs its own artifact"
        )
    output_items = list(outputs.items())
    for i, h in positioned:
        slot, otype = output_items[i]
        if not _is_artifactable_type(otype):
            raise LineageDeclarationError(
                f"{name}: produces_artifacts declares output {slot} as artifact {h.name!r}, but it is annotated "
                f"{type_name(otype)}; only flyte.io.File, flyte.io.Dir, flyte.io.DataFrame (or a type exposing "
                "get_artifact_metadata) can be published as an artifact."
            )

    # -- consumes ----------------------------------------------------------------------------------
    single_partition: Dict[str, Artifact] = {h.name: h for h in produced}
    for key, value in consumes_map.items():
        if is_handle(value) or (isinstance(value, ArtifactMapping) and value.kind in ("identity", "select")):
            h = value.handle if isinstance(value, ArtifactMapping) else value
            single_partition.setdefault(h.name, h)

    referenced: List[Artifact] = list(produced)
    produced_names = {h.name for h in produced}
    instance = _instance_handles(produced, consumes_map)
    parameters: Dict[str, Dict[str, Any]] = {}
    consumed_nodes: List[str] = []
    for key, value in consumes_map.items():
        if key not in inputs:
            params = ", ".join(inputs) or "none"
            raise LineageDeclarationError(
                f"{name}: consumes_artifacts key {key!r} names no parameter of the task (parameters: {params})."
            )
        ptype = inputs[key][0]
        tname = type_name(ptype)
        binding = value.identity_mapping if is_handle(value) else value
        if isinstance(binding, RequiredParam):
            parameters[key] = {"kind": "required", "type": tname}
        elif isinstance(binding, OutputPartition):
            carrier = _instance_dim(instance, binding.dim)
            if carrier is None:
                what = (
                    f"no artifact it produces ({', '.join(h.name for h in produced)}) has it"
                    if produced
                    else "it produces nothing, and no input mapped by identity or select has it"
                )
                raise LineageDeclarationError(
                    f"{name}: consumes_artifacts[{key!r}] is artifacts.partition({binding.dim!r}), but {what}. "
                    "artifacts.partition reads a dimension of the task's own outputs."
                )
            kind = dimension_kind(carrier.partitions[binding.dim])
            if not _dimension_accepts(kind, ptype):
                raise LineageDeclarationError(
                    f"{name}: consumes_artifacts[{key!r}] is artifacts.partition({binding.dim!r}), a {kind} dimension "
                    f"of {carrier.name}, so parameter {key!r} must be typed {_DIMENSION_TYPES[kind]}; it is {tname}."
                )
            parameters[key] = {"kind": "partition", "node": carrier.name, "dim": binding.dim, "type": tname}
            referenced.append(carrier)
        elif isinstance(binding, ArtifactMapping):
            h = binding.handle
            if binding.is_list and not _is_list_type(ptype):
                raise LineageDeclarationError(
                    f"{name}: consumes_artifacts[{key!r}] is {h.name}.{binding.describe()}, which yields many "
                    f"partitions, so parameter {key!r} must be typed list[...]; it is {tname}."
                )
            if not binding.is_list and _is_list_type(ptype):
                raise LineageDeclarationError(
                    f"{name}: consumes_artifacts[{key!r}] maps {h.name} by {binding.kind}, which yields one "
                    f"partition, but parameter {key!r} is typed {tname}. Use {h.name}.all(...) or "
                    f"{h.name}.window(...) for a list."
                )
            parameters[key] = {"kind": "artifact", "node": h.name, "type": tname, "mapping": binding.to_dict()}
            if h.name in produced_names:
                # A task reading its own output (an incremental window over earlier versions): allowed. The
                # backend drops the self-edge and the planner treats it as a lookup of existing versions.
                parameters[key]["self"] = True
            referenced.append(h)
            consumed_nodes.append(h.name)
        elif isinstance(binding, PartitionValue):
            h = binding.handle
            if binding.dim not in h.partitions:
                raise LineageDeclarationError(
                    f"{name}: consumes_artifacts[{key!r}] reads dimension {binding.dim!r} of {h.name}, which has none "
                    f"(declared: {', '.join(h.partitions) or 'none'})."
                )
            if h.name not in single_partition:
                raise LineageDeclarationError(
                    f"{name}: consumes_artifacts[{key!r}] is {h.name}.get_partition_value({binding.dim!r}), "
                    f"but {h.name} does not resolve to a single partition in this declaration. get_partition_value "
                    "reads against an artifact the task produces or an input mapped by identity (or select), not a "
                    "windowed or fanned-in input."
                )
            kind = dimension_kind(h.partitions[binding.dim])
            if not _dimension_accepts(kind, ptype):
                raise LineageDeclarationError(
                    f"{name}: consumes_artifacts[{key!r}] is {h.name}.get_partition_value({binding.dim!r}), a {kind} "
                    f"dimension, so parameter {key!r} must be typed {_DIMENSION_TYPES[kind]}; it is {tname}."
                )
            parameters[key] = {"kind": "partition", "node": h.name, "dim": binding.dim, "type": tname}
            referenced.append(h)
        else:
            raise LineageDeclarationError(
                f"{name}: consumes_artifacts[{key!r}] must be an artifacts.Artifact handle, a mapping "
                f"(handle.all/window/select), handle.get_partition_value(dim), artifacts.partition(dim) or "
                f"artifacts.required(); got {type(binding).__name__}."
            )

    # -- remaining parameters -------------------------------------------------------------------------
    # The implicit rule: a parameter with no default and no binding, named like a dimension of the instance,
    # carries that dimension's value (as the factory does).
    implicit = implicit_partition_params(task)
    unbound: List[str] = []
    for pname, (ptype, default) in inputs.items():
        if pname in parameters:
            continue
        tname = type_name(ptype)
        if pname in implicit:
            h, dim = implicit[pname]
            parameters[pname] = {"kind": "partition", "node": h.name, "dim": dim, "type": tname, "implicit": True}
            referenced.append(h)
        elif default is inspect.Parameter.empty:
            parameters[pname] = {"kind": "unbound", "type": tname}
            unbound.append(pname)
        else:
            parameters[pname] = _default_record(tname, default)
    lines = _param_lines(task, src_line)
    ordered = {
        p: {**parameters[p], "src_file": src_file, "src_line": lines.get(p, src_line)}
        for p in inputs
        if p in parameters
    }

    handles = check_conflicts(referenced, root)
    # Bare handles carry a node but no record: keep them out of `artifacts` so they never look declared.
    records = {n: _handle_record(h, root) for n, h in handles.items()}

    # Every dimension of a produced handle needs a parameter that carries it: a get_partition_value binding for
    # that dimension (on the produced handle or another single-partition handle), an artifacts.partition(dim)
    # binding, or a parameter named like the dimension (the implicit rule). That is exactly what the
    # run time reads (`declared_output_metadata`), so deploy and run agree; an identity-mapped input does not
    # count, because its coordinate is not a value the task receives.
    bound_dims = {p["dim"] for p in parameters.values() if p["kind"] == "partition"}
    if produced:
        produced_dims = {d for h in produced for d in h.partitions}
        for key, v in consumes_map.items():
            m = v.identity_mapping if is_handle(v) else v
            if not isinstance(m, ArtifactMapping) or m.kind not in ("identity", "select"):
                continue
            if m.handle.name in produced_names:
                continue  # reading its own output: a lookup, not a fan-in
            for d in m.handle.partitions:
                if d in m.pinned or d in produced_dims:
                    continue
                out_names = ", ".join(h.name for h in produced)
                raise LineageDeclarationError(
                    f"{name}: consumes_artifacts[{key!r}] maps {m.handle.name} by {m.kind}, but {m.handle.name} has "
                    f"dimension {d!r} that {out_names} does not, so nothing can choose a {d!r} when building a "
                    f"partition. Use {m.handle.name}.all({d!r}) to collapse it, {m.handle.name}.select({d}=...) to "
                    f"pin it, or add {d!r} to {out_names}."
                )
    unbound_dims = [(h.name, d) for h in produced for d in h.partitions if d not in bound_dims]

    messages = [_unpullable_message(name, p, [h.name for h in produced]) for p in unbound]
    messages += [_unbound_dim_message(name, hn, d) for hn, d in unbound_dims]
    if produced:
        pullable = not messages
        reason = _join_reasons(messages)
    else:
        # A sink: it publishes nothing, and is plannable by the same rule as a build (every parameter bound,
        # defaulted or required()) as long as it reads at least one typed artifact (a bare-name binding is a
        # label-only edge, which nothing plans).
        typed_input = any(p["kind"] == "artifact" and p["node"] in records for p in parameters.values())
        if not typed_input:
            messages = [f"{name} is not pullable: it reads no typed artifact, so there is no instance to plan"]
        pullable = not messages
        reason = _join_reasons(messages)

    lineage.produces = _union([h.name for h in produced], hand_produces)
    lineage.consumes = _union(consumed_nodes, hand_consumes)
    lineage.level = _ladder_level(produced, any(p["kind"] == "artifact" for p in parameters.values()))
    lineage.pullable = pullable
    lineage.unpullable_reason = reason
    lineage.unpullable_params = unbound
    lineage.unpullable_messages = messages
    lineage.handles = handles
    lineage.bindings = {
        "version": BINDINGS_VERSION,
        "task": name,
        "src_file": src_file,
        "src_line": src_line,
        "level": lineage.level,
        "produces": [{"node": h.name, "position": i} for i, h in positioned],
        "outputs": len(outputs),
        "artifacts": records,
        "parameters": ordered,
        "pullable": pullable,
        "unpullable_reason": reason,
    }
    lineage.labels = _finalize_labels(user_labels, lineage)
    return lineage


#: What the backend accepts in a trigger name for it to appear in the lineage graph (its node id is
#: `trigger:<project>/<task>/<trigger>`; see `TriggerEntity` in cloud/lineage/entity.go).
_TRIGGER_NAME_RE = re.compile(r"[A-Za-z0-9_.:-]+(/[A-Za-z0-9_.:-]+)*")


def _check_triggers(task: Any, name: str, *, strict: bool) -> None:
    """
    The lineage side of a task's artifact triggers. A partition key the backend refuses drops the trigger from the
    graph: an error for a task that declares lineage (`strict`), already warned at `OnArtifact(...)` otherwise. A
    trigger name outside `[A-Za-z0-9_.:/-]` keeps the trigger working but out of the lineage graph: a warning.
    """
    from flyte._trigger import OnArtifact, TriggeredPartition

    for trig in getattr(task, "triggers", None) or ():
        automation = getattr(trig, "automation", None)
        if not isinstance(automation, OnArtifact):
            continue
        tname = getattr(trig, "name", "") or ""
        if strict:
            keys = list(automation.partitions or {})
            keys += [v.key for v in (getattr(trig, "inputs", None) or {}).values() if isinstance(v, TriggeredPartition)]
            for k in keys:
                if not is_lineage_ident(k):
                    raise LineageDeclarationError(
                        f"{name}: trigger {tname!r} uses partition key {k!r}, which the lineage graph cannot record; "
                        f"partition keys are {LINEAGE_IDENT_RULE}."
                    )
        if _TRIGGER_NAME_RE.fullmatch(tname) is None and (name, tname) not in _TRIGGER_NAME_WARNED:
            _TRIGGER_NAME_WARNED.add((name, tname))
            logger.warning(
                f"{name}: trigger name {tname!r} has characters outside letters, digits and '_ . : / -' (or an empty "
                "'/' segment); the trigger works, but it won't appear in the lineage graph."
            )


_TRIGGER_NAME_WARNED: set = set()


def _materialize_on_bindings(task: Any, lineage: TaskLineage, spec: Mapping[str, Any], root: Optional[str]) -> None:
    """
    The `lineage.bindings` of a refresh task (`Artifact(refresh=...)`, `handle.materialize_on(...)`): no produced or
    consumed artifact, a task-level `materialize_on` record (target, event, lag), and the target (and source)
    handle records. `materialize_on` is the wire name the graph and snapshots read.
    """
    target: Artifact = spec["target"]
    source: Optional[Artifact] = spec.get("source")
    handles = check_conflicts([h for h in (target, source) if h is not None], root)
    lineage.consumes = _union([target.name], lineage.consumes)
    lineage.handles = handles
    lineage.unpullable_reason = "a refresh task is started by its trigger, not planned"
    lineage.bindings = {
        "version": BINDINGS_VERSION,
        "task": lineage.task,
        "src_file": lineage.src_file,
        "src_line": lineage.src_line,
        "level": lineage.level,
        "produces": [],
        "outputs": len(task.native_interface.outputs),
        "artifacts": {n: _handle_record(h, root) for n, h in handles.items()},
        # Every input comes from the trigger (its time, or a partition of the triggering version): nothing to plan.
        "parameters": {},
        "pullable": False,
        "unpullable_reason": lineage.unpullable_reason,
        "materialize_on": dict(spec["record"]),
    }


def _encode_bindings(bindings: Mapping[str, Any], where: str) -> Optional[str]:
    """
    The `lineage.bindings` label value, kept under the backend's 64 KiB cap (past it, or on any payload the
    backend's validation refuses, the entity is stored without lineage while the deploy succeeds; so the payload
    is checked here first, see `check_bindings`).

    Past `BINDINGS_SOFT_LIMIT` it sheds detail in order: handle descriptions, then per-parameter source
    locations. If still too big, the label is dropped (None) with a warning; produces/consumes are unaffected.
    """

    def enc(b: Mapping[str, Any]) -> str:
        try:
            return json.dumps(b, sort_keys=False, separators=(",", ":"), allow_nan=False)
        except ValueError as e:  # NaN/Infinity: not JSON, refused by the backend
            raise LineageDeclarationError(f"{where}: lineage bindings hold a non-finite number ({e}).") from e

    check_bindings(bindings, where)
    out = enc(bindings)
    if len(out.encode("utf-8")) <= BINDINGS_SOFT_LIMIT:
        return out
    slim: Dict[str, Any] = dict(bindings)
    arts = slim.get("artifacts")
    if isinstance(arts, Mapping):
        slim["artifacts"] = {
            n: ({k: v for k, v in r.items() if k != "description"} if isinstance(r, Mapping) else r)
            for n, r in arts.items()
        }
    out = enc(slim)
    if len(out.encode("utf-8")) <= BINDINGS_SOFT_LIMIT:
        logger.warning(f"{where}: lineage bindings exceed {BINDINGS_SOFT_LIMIT} bytes; dropped handle descriptions.")
        return out
    params = slim.get("parameters")
    if isinstance(params, Mapping):
        slim["parameters"] = {
            n: ({k: v for k, v in r.items() if k not in ("src_file", "src_line")} if isinstance(r, Mapping) else r)
            for n, r in params.items()
        }
    out = enc(slim)
    if len(out.encode("utf-8")) <= BINDINGS_SOFT_LIMIT:
        logger.warning(
            f"{where}: lineage bindings exceed {BINDINGS_SOFT_LIMIT} bytes; dropped handle descriptions and "
            "parameter source locations."
        )
        return out
    logger.warning(
        f"{where}: lineage bindings are {len(out.encode('utf-8'))} bytes even without descriptions and source "
        f"locations (limit {BINDINGS_SOFT_LIMIT}); not writing '{BINDINGS_LABEL}'. The lineage edges "
        f"('{PRODUCES_LABEL}'/'{CONSUMES_LABEL}') are still recorded."
    )
    return None


def _suggest_ident(name: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "_", name)[:64] or "param"
    return f"_{out[:63]}" if out[0].isdigit() else out


def _check_text(where: str, what: str, value: Any, limit: int, *, prose: bool = False) -> None:
    if not isinstance(value, str):
        return
    size = len(value.encode("utf-8"))
    bad = re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]" if prose else r"[\x00-\x1f\x7f]", value)
    if size > limit or bad:
        problem = f"is {size} bytes (the limit is {limit})" if size > limit else "contains control characters"
        raise LineageDeclarationError(f"{where}: lineage bindings {what} {problem}.")


def _check_ident(where: str, what: str, value: Any) -> None:
    if not is_lineage_ident(value):
        raise LineageDeclarationError(f"{where}: {what} {value!r} must be {LINEAGE_IDENT_RULE}.")


def check_bindings(bindings: Mapping[str, Any], where: str) -> None:
    """
    The backend's validation of a `lineage.bindings` payload (cloud/lineage/validate.go), run before it is written:
    a payload the backend refuses would leave the entity without lineage while the deploy succeeds. The SDK
    normalizes free text where it writes it (`type_name`, descriptions, source files), so this only fires on what a
    user has to fix, and names it.

    Raises:
        LineageDeclarationError: on an identifier, length or control-character violation.
    """
    for key in ("task", "app"):
        _check_text(where, key, bindings.get(key), MAX_SHORT_TEXT_BYTES)
    _check_text(where, "src_file", bindings.get("src_file"), MAX_SRC_FILE_BYTES)
    _check_text(where, "unpullable_reason", bindings.get("unpullable_reason"), MAX_DESCRIPTION_BYTES, prose=True)
    for node, rec in (bindings.get("artifacts") or {}).items():
        if not isinstance(rec, Mapping):
            continue
        w = f"artifact {node!r}"
        for f, limit in (("type", MAX_TYPE_BYTES), ("kind", MAX_SHORT_TEXT_BYTES), ("identity", MAX_SHORT_TEXT_BYTES)):
            _check_text(where, f"{w} {f}", rec.get(f), limit)
        _check_text(where, f"{w} src_file", rec.get("src_file"), MAX_SRC_FILE_BYTES)
        _check_text(where, f"{w} description", rec.get("description"), MAX_DESCRIPTION_BYTES, prose=True)
        for f in ("project", "domain"):
            v = rec.get(f)
            if v and (not isinstance(v, str) or PROJECT_NAME_RE.fullmatch(v) is None):
                raise LineageDeclarationError(f"{where}: artifact {node!r} names an invalid {f} {v!r}.")
        for d in rec.get("dims") or ():
            _check_ident(where, f"artifact {node!r} dimension", d.get("name"))
    for pname, p in (bindings.get("parameters") or {}).items():
        _check_ident(where, "parameter name", pname)
        if not isinstance(p, Mapping):
            continue
        w = f"parameter {pname!r}"
        _check_text(where, f"{w} kind", p.get("kind"), 64)
        _check_text(where, f"{w} type", p.get("type"), MAX_TYPE_BYTES)
        _check_text(where, f"{w} src_file", p.get("src_file"), MAX_SRC_FILE_BYTES)
        if p.get("dim"):
            _check_ident(where, f"{w} dim", p["dim"])
        m = p.get("mapping")
        if isinstance(m, Mapping):
            _check_text(where, f"{w} mapping kind", m.get("kind"), 64)
            if m.get("dim"):
                _check_ident(where, f"{w} mapping dim", m["dim"])
            for k, v in (m.get("values") or {}).items():
                _check_ident(where, f"{w} select key", k)
                _check_text(where, f"{w} select value", v, MAX_SHORT_TEXT_BYTES)


def check_edge_limits(labels: Mapping[str, str], where: str, *, produces_count: Optional[int] = None) -> None:
    """
    Enforce the backend's per-entity lineage limits client-side, so `flyte deploy` fails with a readable error
    instead of the backend silently storing the entity without its lineage: each of
    `lineage.produces`/`lineage.consumes` at most `MAX_EDGE_LABEL_BYTES` bytes and `MAX_EDGE_IDS` ids, and
    consumes x produces at most `MAX_EDGE_PAIRS`.

    Args:
        labels: The entity's labels.
        where: Name of the entity, for the error message.
        produces_count: The number of produced ids when the backend derives them (an app produces one).
    """
    counts: Dict[str, int] = {}
    for key in (PRODUCES_LABEL, CONSUMES_LABEL):
        value = labels.get(key)
        if value is None:
            counts[key] = 0
            continue
        size = len(value.encode("utf-8"))
        if size > MAX_EDGE_LABEL_BYTES:
            raise LineageDeclarationError(
                f"{where}: '{key}' is {size} bytes; the limit is {MAX_EDGE_LABEL_BYTES}. Declare fewer artifacts "
                "on this entity or use shorter artifact names."
            )
        n = len(split_node_ids(value))
        if n > MAX_EDGE_IDS:
            raise LineageDeclarationError(
                f"{where}: '{key}' lists {n} artifacts; at most {MAX_EDGE_IDS} are allowed per entity."
            )
        counts[key] = n
    produced = counts[PRODUCES_LABEL] if produces_count is None else produces_count
    if produced * counts[CONSUMES_LABEL] > MAX_EDGE_PAIRS:
        raise LineageDeclarationError(
            f"{where}: {counts[CONSUMES_LABEL]} consumed x {produced} produced artifacts make "
            f"{produced * counts[CONSUMES_LABEL]} lineage edges; at most {MAX_EDGE_PAIRS} are allowed per entity. "
            "Split the entity or declare fewer artifacts."
        )


def _finalize_labels(user_labels: Mapping[str, str], lineage: TaskLineage) -> Dict[str, str]:
    out = {k: v for k, v in user_labels.items() if k not in (PRODUCES_LABEL, CONSUMES_LABEL)}
    if lineage.produces:
        out[PRODUCES_LABEL] = ",".join(lineage.produces)
    if lineage.consumes:
        out[CONSUMES_LABEL] = ",".join(lineage.consumes)
    check_edge_limits(out, lineage.task)
    if lineage.bindings is not None:
        encoded = _encode_bindings(lineage.bindings, lineage.task)
        if encoded is not None:
            out[BINDINGS_LABEL] = encoded
    check_label_count(out, lineage.task)
    return out


def _handle_fingerprint(h: Optional[Artifact]) -> Any:
    if h is None:
        return None
    return (
        h.name,
        _signature(h),
        tuple(sorted(h.expected.items())),
        h.description,
        h.kind,
        h.source,
        h.identity,
        h.project,
        h.domain,
        h.src_path,
        h.src_line,
    )


def _declaration_fingerprint(task: Any) -> Any:
    """The content of a task's declarations (handles, mappings, bindings), so in-place edits are not stale."""
    produced = getattr(task, "produces_artifacts", False)
    prod = tuple(_handle_fingerprint(h) for h in produced) if isinstance(produced, tuple) else produced
    items = []
    for k, v in (getattr(task, "consumes_artifacts", None) or {}).items():
        if is_handle(v):
            items.append((k, "handle", _handle_fingerprint(v)))
        elif isinstance(v, ArtifactMapping):
            items.append((k, json.dumps(v.to_dict(), sort_keys=True), _handle_fingerprint(v.handle)))
        elif isinstance(v, PartitionValue):
            items.append((k, f"partition:{v.dim}", _handle_fingerprint(v.handle)))
        else:
            items.append((k, repr(v), None))
    return prod, tuple(items)


def _env_labels(task: Any) -> Optional[Mapping[str, str]]:
    """The labels of the environment a task belongs to, read at serialization time."""
    ref = getattr(task, "parent_env", None)
    env = ref() if callable(ref) else None
    return getattr(env, "labels", None) if env is not None else None


# id(task) -> (weakref to the task, cache key, lineage). Kept off the task object so it never reaches the
# cloudpickle'd deployment (and its version hash).
_LINEAGE_CACHE: Dict[int, Tuple[Any, Tuple[Any, ...], TaskLineage]] = {}


def task_lineage_tags(
    task: TaskTemplate, root_dir: Any = None, extra_labels: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """The `TaskMetadata.tags` for a task, memoized per task object (serialization runs per submission)."""
    return dict(task_lineage(task, root_dir, extra_labels).labels)


def task_lineage(
    task: TaskTemplate, root_dir: Any = None, extra_labels: Optional[Mapping[str, str]] = None
) -> TaskLineage:
    """
    `extract_task_lineage`, memoized per task object and declaration content, so a deploy's summary and its
    serialization extract each task once. The result is shared: treat it as read-only.
    """

    def _items(m: Optional[Mapping[str, str]]) -> Tuple[Tuple[str, str], ...]:
        return tuple(sorted((m or {}).items()))

    key = (
        str(root_dir) if root_dir is not None else None,
        _items(extra_labels),
        _items(_env_labels(task)),
        _items(getattr(task, "labels", None)),
        _declaration_fingerprint(task),
    )
    hit = _LINEAGE_CACHE.get(id(task))
    if hit is not None and hit[0]() is task and hit[1] == key:
        return hit[2]
    lineage = extract_task_lineage(task, root_dir=key[0], extra_labels=extra_labels)
    tid = id(task)

    def _evict(_ref: Any, tid: int = tid) -> None:
        _LINEAGE_CACHE.pop(tid, None)

    try:
        ref = weakref.ref(task, _evict)
    except TypeError:
        return lineage
    _LINEAGE_CACHE[tid] = (ref, key, lineage)
    return lineage


# --------------------------------------------------------------------------------------------------
# Apps
# --------------------------------------------------------------------------------------------------


def app_lineage_labels(
    app_name: str,
    *,
    labels: Optional[Mapping[str, str]] = None,
    consumed: Sequence[str] = (),
    extra_labels: Optional[Mapping[str, str]] = None,
    bindings: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """
    The `Meta.labels` for an app: user labels plus `lineage.consumes` (consumes_artifacts names, artifact-valued
    parameters, and any hand-written value) and, when given, `lineage.bindings` (which parameter takes which
    artifact). `lineage.produces` is rejected; the backend derives it.
    """
    merged = merge_labels(labels, extra_labels)
    validate_labels(merged, entity="app", where=app_name)
    out = {k: v for k, v in merged.items() if k != CONSUMES_LABEL}
    nodes = _union(consumed, split_node_ids(merged.get(CONSUMES_LABEL, "")))
    for node in nodes:
        if not valid_node_id(node):
            raise LineageDeclarationError(
                f"{app_name}: consumes {node!r}, which is not a valid lineage node id (letters, digits, '_', '.', "
                "':' and '-')."
            )
    if nodes:
        out[CONSUMES_LABEL] = ",".join(nodes)
    # Also when no parameter takes an artifact but the app consumes something: an empty record says those
    # edges are label-only, which a reader cannot tell from an app deployed before records existed.
    check_edge_limits(out, app_name, produces_count=1)  # the backend derives produces = app:<name>
    if bindings is not None and (bindings.get("parameters") or out.get(CONSUMES_LABEL)):
        encoded = _encode_bindings(bindings, app_name)
        if encoded is not None:
            out[BINDINGS_LABEL] = encoded
    check_label_count(out, app_name, reserved=1)  # the serializer adds the managed-labels key
    return out


def app_bindings(
    app_env: Any, parameters: Optional[Sequence[Any]] = None, root_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    The `lineage.bindings` payload of an app (version 1): each artifact-valued parameter (a `consumes_artifacts`
    entry, a `Parameter(value=<handle>)` or an `ArtifactValue`) as an identity-mapped `artifact` binding, and
    the handle record of each artifact whose handle is known. A bare `ArtifactValue(name=...)` carries a node
    but no record, as a bare handle does for a task.
    """
    from flyte.app._parameter import ArtifactValue

    declared = getattr(app_env, "consumes_artifacts", None) or {}
    params: Dict[str, Any] = {}
    handles: List[Artifact] = []
    for p in parameters if parameters is not None else app_env.parameters:
        v = p.value
        if not isinstance(v, ArtifactValue):
            continue
        if not is_lineage_ident(p.name):
            raise LineageDeclarationError(
                f"{app_env.name}: parameter {p.name!r} takes artifact {v.name!r}, so its name is recorded in the "
                f"lineage graph and must be {LINEAGE_IDENT_RULE}. Rename it (e.g. {_suggest_ident(p.name)!r})."
            )
        h = getattr(v, "handle", None)
        if h is None and is_handle(declared.get(p.name)):
            h = declared[p.name]
        if is_handle(h) and h.name == v.name:
            handles.append(h)
            params[p.name] = {
                "kind": "artifact",
                "node": h.name,
                "type": type_name(h.type),
                "mapping": h.identity_mapping.to_dict(),
            }
        else:
            tname = {"file": "File", "directory": "Dir"}.get(v.type or "", v.type or "")
            params[p.name] = {"kind": "artifact", "node": v.name, "type": tname, "mapping": {"kind": "identity"}}
    records = {n: _handle_record(h, root_dir) for n, h in check_conflicts(handles, root_dir).items()}
    return {"version": BINDINGS_VERSION, "app": app_env.name, "parameters": params, "artifacts": records}


def app_env_lineage_labels(
    app_env: Any,
    *,
    parameters: Optional[Sequence[Any]] = None,
    extra_labels: Optional[Mapping[str, str]] = None,
    root_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    The `Meta.labels` of an `AppEnvironment`: artifact-valued parameters (including `consumes_artifacts`) and
    `AppEndpoint` parameters (as `app:<name>`) become `lineage.consumes`, merged with the app's labels and
    `extra_labels`; the artifact-valued parameters are also written as `lineage.bindings` (see
    `app_bindings`). `depends_on` is deliberately not an edge.
    """
    from flyte.app._parameter import AppEndpoint, ArtifactValue

    params = list(parameters if parameters is not None else app_env.parameters)
    consumed: List[str] = []
    for p in params:
        if isinstance(p.value, ArtifactValue):
            consumed = _union(consumed, [p.value.name])
        elif isinstance(p.value, AppEndpoint):
            consumed = _union(consumed, [f"app:{p.value.app_name}"])
    if root_dir is None:
        try:
            from flyte._initialize import get_init_config

            rd = get_init_config().root_dir
            root_dir = str(rd) if rd is not None else None
        except Exception:
            root_dir = None
    return app_lineage_labels(
        app_env.name,
        labels=app_env.labels,
        consumed=consumed,
        extra_labels=extra_labels,
        bindings=app_bindings(app_env, params, root_dir),
    )


# --------------------------------------------------------------------------------------------------
# Deploy-wide checks and summary
# --------------------------------------------------------------------------------------------------


def iter_tasks(envs: Iterable[Any]) -> List[TaskTemplate]:
    """Every task of every `TaskEnvironment` among `envs`, in order."""
    from flyte._task_environment import TaskEnvironment

    out: List[TaskTemplate] = []
    for env in envs:
        if isinstance(env, TaskEnvironment):
            out.extend(env.tasks.values())
    return out


@dataclass
class LineageSummary:
    """
    What a deploy declared.

    Attributes:
        tasks: Number of tasks deployed.
        handles: Distinct typed artifact handles referenced.
        edges: Distinct resolvable dependency edges (consumed handle -> produced handle).
        warnings: Pullability warnings.
        lineages: Per-task lineage.
        references_checked: `Artifact.ref` handles whose partitions matched the registry at deploy.
        notes: Informational lines (e.g. a reference the registry does not know yet).
        refreshes: The refresh environments/triggers the deploy added (`Artifact(refresh=...)`), one line each.
    """

    tasks: int = 0
    handles: int = 0
    edges: int = 0
    warnings: List[str] = field(default_factory=list)
    lineages: List[TaskLineage] = field(default_factory=list)
    references_checked: int = 0
    notes: List[str] = field(default_factory=list)
    refreshes: List[str] = field(default_factory=list)

    @property
    def nodes(self) -> List[str]:
        """Every node id any task reads or writes, in first-seen order."""
        out: Dict[str, None] = {}
        for t in self.lineages:
            out.update(dict.fromkeys(t.consumes))
            out.update(dict.fromkeys(t.produces))
        return list(out)

    @property
    def edge_set(self) -> List[Tuple[str, str]]:
        """Every `consumes x produces` edge (typed or label-only), in first-seen order."""
        out: Dict[Tuple[str, str], None] = {}
        for t in self.lineages:
            out.update(dict.fromkeys(t.edges))
        return list(out)

    @property
    def relevant(self) -> bool:
        """Whether the deploy declared anything lineage-related: handles, lineage labels, or warnings."""
        return bool(
            self.handles
            or self.warnings
            or self.notes
            or self.refreshes
            or any(t.produces or t.consumes for t in self.lineages)
        )

    def line(self) -> str:
        """`✓ 4 tasks, 5 artifact handles, 5 dependency edges resolved`."""
        return (
            f"✓ {self.tasks} task{'s' if self.tasks != 1 else ''}, {self.handles} artifact "
            f"handle{'s' if self.handles != 1 else ''}, {self.edges} dependency "
            f"edge{'s' if self.edges != 1 else ''} resolved"
        )

    def render(self) -> str:
        """The summary line, then each warning as `! ...` and each note as `· ...`."""
        lines = [self.line()]
        if self.references_checked:
            n = self.references_checked
            lines.append(f"  {n} artifact reference{'s' if n != 1 else ''} checked against the registry")
        if self.refreshes:
            lines.append("  Refresh triggers added by this deploy:")
            lines.extend(f"  {r}" for r in self.refreshes)
        lines.extend(f"· {n}" for n in self.notes)
        for w in self.warnings:
            lines.append(f"! {w}")
            lines.append(
                "  Deployed anyway. The task still runs when called directly; it cannot be a materialize target."
            )
        return "\n".join(lines)


def summarize(
    tasks: Iterable[TaskTemplate],
    *,
    root_dir: Any = None,
    extra_labels: Optional[Mapping[str, str]] = None,
) -> LineageSummary:
    """
    Extract every task's lineage, fail on conflicting declarations across them, and count what resolved.

    Raises:
        LineageDeclarationError: on an invalid declaration or a cross-module conflict.
    """
    root = str(root_dir) if root_dir is not None else None
    tasks = list(tasks)
    lineages = [task_lineage(t, root, extra_labels) for t in tasks]
    all_handles: List[Artifact] = []
    for lin in lineages:
        all_handles.extend(lin.handles.values())
    declared = check_conflicts(all_handles, root)
    edges: Dict[Tuple[str, str], None] = {}
    for lin in lineages:
        for e in lin.resolvable_edges:
            edges[e] = None
    warnings: List[str] = []
    for lin in lineages:
        warnings.extend(lin.warnings())
    return LineageSummary(
        tasks=len(tasks), handles=len(declared), edges=len(edges), warnings=warnings, lineages=lineages
    )


# --------------------------------------------------------------------------------------------------
# Run time: publishing declared handles
# --------------------------------------------------------------------------------------------------


@dataclass
class HandleDeclaration:
    """
    What a `produces_artifacts=(handle, ...)` slot publishes at run time.

    Attributes:
        handle: The declared handle.
        metadata: The `Metadata` to publish when the body did not wrap the value itself, or None when it
            cannot be built (see `error` and `skip_reason`).
        error: A coordinate that does not parse. Used only if the body did not supply the metadata itself: the
            slot is then not published (with a warning), or the task fails under `FLYTE_LINEAGE_STRICT=1`.
        skip_reason: Why the declaration alone cannot fully partition this slot (a dimension with no bound
            parameter, or a parameter that is None). The slot is then not published (with a warning) unless the
            body returned `artifacts.new(value, handle.at(...))`; the task does not fail.
        caller_declared: The caller declared this slot (`flyte.artifacts.produces`, the factory path). Nothing
            is parsed; the handle only fills the description and kind the caller left empty.
    """

    handle: Artifact
    metadata: Any = None
    error: Optional[Exception] = None
    skip_reason: Optional[str] = None
    caller_declared: bool = False

    def fill_metadata(self) -> Any:
        """The handle's description and kind, as `Metadata`, to fill what a caller declaration leaves empty."""
        from ._metadata import Metadata

        return Metadata(name=self.handle.name, description=self.handle.description, kind=self.handle.kind)

    def check_own(self, md: Any, task_name: str, slot: str) -> None:
        """
        Reject an `artifacts.new(...)` version of this handle that leaves a declared dimension unset.

        Raises RuntimeUserError("MissingPartition"); the output conversion catches it and, unless
        `FLYTE_LINEAGE_STRICT=1`, publishes nothing for the slot instead of failing the task.
        """
        if getattr(md, "name", None) != self.handle.name:
            return
        missing = [d for d in self.handle.partitions if d not in (md.partitions or {})]
        if missing:
            from flyte.errors import RuntimeUserError

            raise RuntimeUserError(
                "MissingPartition",
                f"{task_name}: output {slot} is published as {self.handle.name!r} without partition "
                f"{', '.join(repr(d) for d in missing)}; every declared dimension "
                f"({', '.join(self.handle.partitions)}) must be set. Pass it to {self.handle.name}.at(...).",
            )


def declared_output_metadata(
    task: Any, inputs: Mapping[str, Any], skip: Iterable[str] = ()
) -> Dict[str, HandleDeclaration]:
    """
    The declaration to publish for each output slot of a task that declares `produces_artifacts=(h0, h1, ...)`.

    The value returned at position i becomes a version of handle i. Its partition values come from the
    parameters bound by `get_partition_value`, `artifacts.partition(dim)` or the implicit rule (a parameter with
    no default named like the dimension): a binding on the produced handle itself (or one of the latter two) wins,
    then a binding on
    any other handle for a dimension of the same name (`raw_events.get_partition_value("date")` fills `events`'s
    `date`). A dimension nothing supplies (or whose parameter is None) makes the slot unpublishable from the
    declaration alone: it is skipped with a warning unless the body set it with `artifacts.new(...)`. A value
    that does not parse is an error. `None` placeholders in the tuple are skipped.

    Args:
        task: The task template.
        inputs: The task's native input values, by parameter name.
        skip: Output slots the caller already declared (`flyte.artifacts.produces`); nothing is parsed for them,
            and their declaration is marked `caller_declared`.

    Returns:
        Output slot name (`o0`, ...) to `HandleDeclaration`; empty for a task without a handle tuple.
    """
    from flyte.errors import RuntimeUserError

    produced = getattr(task, "produces_artifacts", False)
    if not isinstance(produced, tuple) or not produced:
        return {}
    skipped = set(skip)
    task_name = getattr(task, "name", "the task")
    outputs = list(task.native_interface.outputs)
    consumes: Mapping[str, Any] = getattr(task, "consumes_artifacts", None) or {}
    bindings = [(p, b) for p, b in consumes.items() if isinstance(b, PartitionValue)]
    # artifacts.partition(dim) and the implicit rule (a parameter named like a dimension) bind the dimension of
    # the task's own outputs: they count as a binding on every produced handle that has that dimension.
    own_dims = [(p, b.dim) for p, b in consumes.items() if isinstance(b, OutputPartition)]
    own_dims += [(p, dim) for p, (_, dim) in implicit_partition_params(task).items()]
    out: Dict[str, HandleDeclaration] = {}
    for slot, h in zip(outputs, produced):
        if h is None:
            continue  # a None placeholder: this output is not an artifact
        if slot in skipped:
            out[slot] = HandleDeclaration(handle=h, caller_declared=True)
            continue
        values: Dict[str, Any] = {}
        missing: List[str] = []
        error: Optional[Exception] = None
        for dim in h.partitions:
            own = [p for p, b in bindings if b.handle.name == h.name and b.dim == dim]
            own += [p for p, d in own_dims if d == dim]
            other = [p for p, b in bindings if b.handle.name != h.name and b.dim == dim]
            src = next(iter(own + other), None)
            if src is None or inputs.get(src) is None:
                missing.append(f"{dim!r}" + (f" (parameter {src!r} is None)" if src is not None else ""))
                continue
            try:
                values[dim] = h._coerce_partition_value(dim, inputs[src])
            except (ValueError, TypeError) as e:
                error = RuntimeUserError(
                    "BadPartitionValue",
                    f"{task_name}: parameter {src!r} = {inputs[src]!r} is not a valid value for dimension {dim!r} "
                    f"of {h.name!r}: {e}",
                )
                break
        skip_reason = None
        if error is None and missing:
            skip_reason = (
                f"{task_name}: not publishing output {slot} as {h.name!r}: no value for dimension "
                f"{', '.join(missing)}. Name a parameter like the dimension, bind one with artifacts.partition(...) "
                f"or {h.name}.get_partition_value(...), or return artifacts.new(value, {h.name}.at(...)) with every "
                "dimension set."
            )
        ok = error is None and skip_reason is None
        out[slot] = HandleDeclaration(
            handle=h, metadata=h.at(**values) if ok else None, error=error, skip_reason=skip_reason
        )
    return out
