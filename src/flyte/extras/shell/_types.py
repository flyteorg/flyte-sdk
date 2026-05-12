from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Literal, Tuple, Type, Union, get_args, get_origin

from flyte.io import Dir, File


@dataclass(frozen=True)
class Glob:
    """A multi-file output bundle. Lives in ``/var/outputs/<output_name>/``."""

    pattern: str = "*"


_SCALAR_OUTPUT_TYPES: frozenset = frozenset({int, float, str, bool})


@dataclass(frozen=True)
class Stdout:
    """Capture the task's stdout as a typed output."""

    type: Type = File

    def __post_init__(self):
        if self.type is not File and self.type not in _SCALAR_OUTPUT_TYPES:
            raise TypeError(
                f"Stdout.type must be File or a primitive (int/float/str/bool), "
                f"got {self.type!r}."
            )


@dataclass(frozen=True)
class Stderr:
    """Capture the task's stderr as a typed output. See :class:`Stdout`."""

    type: Type = File

    def __post_init__(self):
        if self.type is not File and self.type not in _SCALAR_OUTPUT_TYPES:
            raise TypeError(
                f"Stderr.type must be File or a primitive (int/float/str/bool), "
                f"got {self.type!r}."
            )


_OutputCollector = Union[Glob, Stdout, Stderr]
OutputSpec = Union[Type, _OutputCollector]
listMode = Literal["join", "repeat", "comma"]
DictMode = Literal["pairs", "equals"]


@dataclass(frozen=True)
class FlagSpec:
    """How to render a typed input as a CLI flag in ``{flags.<name>}``."""

    flag: str
    list_mode: listMode = "join"
    separator: str = " "
    dict_mode: DictMode = "pairs"

    @classmethod
    def coerce(
        cls, name: str, alias: Union[str, Tuple[str, str], "FlagSpec", None]
    ) -> "FlagSpec":
        if alias is None:
            return cls(flag=f"-{name}")
        if isinstance(alias, FlagSpec):
            return alias
        if isinstance(alias, str):
            return cls(flag=alias)
        if isinstance(alias, tuple) and len(alias) == 2:
            flag_str, mode = alias
            if mode in ("join", "repeat", "comma"):
                return cls(flag=flag_str, list_mode=mode)
            if mode in ("pairs", "equals"):
                return cls(flag=flag_str, dict_mode=mode)
            raise TypeError(
                f"Invalid mode {mode!r} for flag_aliases[{name!r}]. "
                f"Expected one of: join/repeat/comma (lists), pairs/equals (dicts)."
            )
        raise TypeError(f"Invalid flag_aliases entry for {name!r}: {alias!r}")


@dataclass(frozen=True)
class _ProcessResult:
    """Stdout / stderr / returncode captured during a container run."""

    returncode: int
    stdout: str
    stderr: str


_SCALAR_TYPES: frozenset = frozenset({int, float, str, bool})


def _is_optional(tp: Any) -> Tuple[bool, Any]:
    """Return ``(is_optional, inner_type)`` for ``T | None`` / ``Optional[T]``."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and len(get_args(tp)) == 2:
            return True, args[0]
    return False, tp


def _is_list_of(tp: Any, inner: type) -> bool:
    """Check ``list[inner]`` / ``typing.List[inner]``."""
    if get_origin(tp) is list:
        args = get_args(tp)
        return len(args) == 1 and args[0] is inner
    return False


def _is_dict_str_str(tp: Any) -> bool:
    """Check ``dict[str, str]``."""
    if get_origin(tp) is dict:
        args = get_args(tp)
        return len(args) == 2 and args[0] is str and args[1] is str
    return False


def _classify_input(name: str, tp: Any) -> str:
    """Classify an input type into a kind label."""
    _, inner = _is_optional(tp)

    if inner is bool:
        return "bool"
    if inner in _SCALAR_TYPES:
        return "scalar"
    if inner is File:
        return "file"
    if inner is Dir:
        return "dir"
    if _is_list_of(inner, File):
        return "list_file"
    if _is_dict_str_str(inner):
        return "dict_str"
    raise TypeError(
        f"Unsupported input type for {name!r}: {tp!r}. "
        f"Supported: File, Dir, list[File], dict[str, str], int, float, str, "
        f"bool, or T | None of those."
    )


_BARE_OUTPUT_TYPES: Tuple[type, ...] = (File, Dir) + tuple(_SCALAR_OUTPUT_TYPES)
_OUTPUT_COLLECTOR_TYPES: Tuple[type, ...] = (Glob, Stdout, Stderr)


def _is_bare_output_type(spec: Any) -> bool:
    """``spec`` is a bare Python type usable as an output declaration."""
    return isinstance(spec, type) and issubclass(spec, _BARE_OUTPUT_TYPES)


def _validate_outputs(outputs: dict[str, Any]) -> None:
    """Validate every value is a bare output type or a known collector."""
    for name, spec in outputs.items():
        if _is_bare_output_type(spec) or isinstance(spec, _OUTPUT_COLLECTOR_TYPES):
            continue
        raise TypeError(
            f"Output {name!r}: expected a bare type "
            f"(File, Dir, int, float, str, bool) or a collector "
            f"(Glob, Stdout, Stderr). Got {spec!r}."
        )
