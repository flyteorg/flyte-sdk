"""The values an orchestrator's source sees, converted as the remote worker converts them.

A `flyte.orchestrator.task` runs remotely on the orchestrator worker (the
`inline-leaseworker` in unionai/cloud), which passes every value between the
source and the tasks it calls as a Flyte literal. This module is that
conversion, rule for rule and message for message, so that an orchestrator run
on this machine sees the same values, and fails the same way, as it would on a
cluster:

- `from_monty` / `to_monty` are the worker's `Value::from_monty` /
  `Value::to_monty` (`orchestrator-core/src/value.rs`).
- `to_literal`, `from_literal`, `fits` and `describe` are
  `flyte-children/src/literal.rs`.
- `bind_inputs` and `read_outputs` are `flyte-children/src/launcher.rs`, and
  `encode_outputs` is `inline-leaseworker/src/handler.rs`.

A change here needs the same change in the worker, and the other way round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from flyteidl2.core import interface_pb2, literals_pb2, types_pb2

from flyte.sandbox._bridge import _BLOB_FIELDS, _BLOB_TUPLES, Opaque

# How deeply values may nest, as the worker allows.
MAX_DEPTH = 32

_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1

_TOO_DEEP = "nested too deeply"

_SINGLE = types_pb2.BlobType.BlobDimensionality.SINGLE
_MULTIPART = types_pb2.BlobType.BlobDimensionality.MULTIPART


@dataclass(frozen=True)
class Blob:
    """A file (`kind` "File") or directory ("Dir"), as the worker holds it: its name is not
    kept, but derived from the path whenever the source sees it."""

    kind: str
    path: str
    format: str
    hash: Optional[str]

    @property
    def name(self) -> str:
        path = self.path.rstrip("/")
        return path.rsplit("/", 1)[-1]


class UnsupportedType(Exception):
    """A value of a type the worker cannot pass to or return from a task."""

    def __init__(self, type_name: str):
        self.type_name = type_name
        if type_name in ("repr", _TOO_DEEP):
            message = (
                "a value passed to or returned from a task is nested too deeply, "
                "or is not plain data (None, bool, int, float, str, files and "
                "directories, and lists, tuples and str-keyed dicts of them)"
            )
        else:
            message = f"values of type '{type_name}' cannot be passed to or returned from a task"
        super().__init__(message)


class ConversionError(Exception):
    """A value that does not convert to or from a literal of the declared type."""

    ambiguous_union = False

    @classmethod
    def mismatch(cls, expected: str, found: str) -> "ConversionError":
        return cls(f"a {found} cannot be passed as {expected}")

    @classmethod
    def ambiguous(cls, found: str, variants: str) -> "ConversionError":
        error = cls(f"a {found} fits more than one variant of the union: {variants}")
        error.ambiguous_union = True
        return error

    @classmethod
    def unsupported_type(cls, described: str) -> "ConversionError":
        return cls(f"values of Flyte type {described} are not supported")

    @classmethod
    def unsupported_literal(cls, kind: str) -> "ConversionError":
        return cls(f"a literal of kind {kind} cannot be read")


def _type_name(value: Any) -> str:
    """The name the worker gives a monty value's type."""
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return "namedtuple"
    if value is None:
        return "NoneType"
    return type(value).__name__


def from_monty(value: Any, depth: int = 0) -> Any:
    """A value the source passed to the host, as the worker reads it: tuples become lists,
    `File`/`Dir` named tuples become `Blob`s, and anything else not plain data is refused."""
    if depth > MAX_DEPTH:
        raise UnsupportedType(_TOO_DEEP)
    type_name = _type_name(value)
    if type_name == "NoneType":
        return None
    if type_name == "bool":
        return value
    if type_name == "int":
        if not _I64_MIN <= value <= _I64_MAX:
            raise UnsupportedType("int")
        return value
    if type_name in ("float", "str"):
        return value
    if type_name in ("list", "tuple"):
        return [from_monty(item, depth + 1) for item in value]
    if type_name == "dict":
        return {_key(key, depth + 1): from_monty(item, depth + 1) for key, item in value.items()}
    if type_name == "namedtuple":
        blob = _blob_from_monty(value)
        if blob is not None:
            return blob
        if type(value).__name__ == "Opaque" and value._fields == Opaque._fields and isinstance(value[0], bytes):
            return Opaque(value[0])
    raise UnsupportedType(type_name)


def _key(key: Any, depth: int) -> Any:
    """A dict key, checked as any value is. The worker takes keys of any type, and refuses all
    but str only when it converts the dict, so a tuple key stays a (hashable) tuple."""
    converted = from_monty(key, depth)
    return key if isinstance(converted, list) else converted


def _blob_from_monty(value: Any) -> Optional[Blob]:
    kind = type(value).__name__
    if kind not in ("File", "Dir") or tuple(value._fields) != _BLOB_FIELDS:
        return None
    path, _name, fmt, hash_ = value
    if not isinstance(path, str) or not isinstance(fmt, str):
        return None
    if hash_ is not None and not isinstance(hash_, str):
        return None
    return Blob(kind=kind, path=path, format=fmt, hash=hash_)


def to_monty(value: Any) -> Any:
    """A value as the source sees it."""
    if isinstance(value, Blob):
        from flyte.io import Dir, File

        tuple_type = _BLOB_TUPLES[File if value.kind == "File" else Dir]
        return tuple_type(value.path, value.name, value.format, value.hash)
    if isinstance(value, list):
        return [to_monty(item) for item in value]
    if isinstance(value, dict):
        return {to_monty(key): to_monty(item) for key, item in value.items()}
    return value


def name_of(value: Any) -> str:
    """The value's type, for messages."""
    if value is None:
        return "None"
    if isinstance(value, Blob):
        return value.kind
    if isinstance(value, Opaque):
        return "Opaque"
    return {bool: "bool", int: "int", float: "float", str: "str", list: "list", dict: "dict"}[type(value)]


def _blob_kind(blob_type: types_pb2.BlobType) -> Optional[str]:
    return {_SINGLE: "File", _MULTIPART: "Dir"}.get(blob_type.dimensionality)


def describe(literal_type: types_pb2.LiteralType) -> str:
    """The type as a Python annotation, for messages."""
    kind = literal_type.WhichOneof("type")
    if kind == "simple":
        try:
            return types_pb2.SimpleType.Name(literal_type.simple).lower()
        except ValueError:
            return f"simple type {literal_type.simple}"
    if kind == "collection_type":
        return f"list[{describe(literal_type.collection_type)}]"
    if kind == "map_value_type":
        return f"dict[str, {describe(literal_type.map_value_type)}]"
    if kind == "union_type":
        return " | ".join(describe(variant) for variant in literal_type.union_type.variants)
    if kind == "blob":
        return _blob_kind(literal_type.blob) or "a file or directory"
    return {
        "structured_dataset_type": "a dataframe",
        "schema": "a schema",
        "enum_type": "an enum",
    }.get(kind or "", "an unknown type")


def _opaque_kind(literal: literals_pb2.Literal) -> Optional[str]:
    if literal.WhichOneof("value") != "scalar":
        return None
    kind = literal.scalar.WhichOneof("value")
    if kind == "structured_dataset":
        return "dataframe"
    if kind in ("binary", "generic"):
        return "struct"
    return None


def _declared_opaque_kind(literal_type: types_pb2.LiteralType) -> Optional[str]:
    kind = literal_type.WhichOneof("type")
    if kind == "structured_dataset_type":
        return "dataframe"
    if kind == "simple" and literal_type.simple == types_pb2.STRUCT:
        return "struct"
    return None


def _opaque_literal(encoded: bytes, literal_type: types_pb2.LiteralType) -> Optional[literals_pb2.Literal]:
    literal = literals_pb2.Literal()
    try:
        literal.ParseFromString(encoded)
    except Exception:
        return None
    kind = _opaque_kind(literal)
    if kind is None or kind != _declared_opaque_kind(literal_type):
        return None
    return literal


def _primitive(**value: Any) -> literals_pb2.Literal:
    return literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(**value)))


_SIMPLE_KINDS = (types_pb2.INTEGER, types_pb2.FLOAT, types_pb2.STRING, types_pb2.BOOLEAN, types_pb2.NONE)


def to_literal(value: Any, literal_type: types_pb2.LiteralType) -> literals_pb2.Literal:
    """The literal of *value* as *literal_type*, or a `ConversionError` if it is not one."""

    def mismatch() -> ConversionError:
        return ConversionError.mismatch(describe(literal_type), name_of(value))

    kind = literal_type.WhichOneof("type")
    if isinstance(value, Opaque) and kind != "union_type":
        literal = _opaque_literal(value.literal, literal_type)
        if literal is None:
            raise mismatch()
        return literal
    if kind == "simple":
        simple = literal_type.simple
        is_bool = isinstance(value, bool)
        is_int = isinstance(value, int) and not is_bool
        if simple == types_pb2.INTEGER and is_int:
            return _primitive(integer=value)
        if simple == types_pb2.FLOAT and isinstance(value, float):
            return _primitive(float_value=value)
        if simple == types_pb2.FLOAT and is_int:
            return _primitive(float_value=float(value))
        if simple == types_pb2.STRING and isinstance(value, str):
            return _primitive(string_value=value)
        if simple == types_pb2.BOOLEAN and is_bool:
            return _primitive(boolean=value)
        if simple == types_pb2.NONE and value is None:
            return literals_pb2.Literal(scalar=literals_pb2.Scalar(none_type=literals_pb2.Void()))
        if simple in _SIMPLE_KINDS:
            raise mismatch()
        raise ConversionError.unsupported_type(describe(literal_type))
    if kind == "collection_type":
        if not isinstance(value, list):
            raise mismatch()
        items = [to_literal(item, literal_type.collection_type) for item in value]
        return literals_pb2.Literal(collection=literals_pb2.LiteralCollection(literals=items))
    if kind == "map_value_type":
        if not isinstance(value, dict):
            raise mismatch()
        literals: Dict[str, literals_pb2.Literal] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise mismatch()
            literals[key] = to_literal(item, literal_type.map_value_type)
        return literals_pb2.Literal(map=literals_pb2.LiteralMap(literals=dict(sorted(literals.items()))))
    if kind == "blob":
        if not isinstance(value, Blob) or value.kind != _blob_kind(literal_type.blob):
            raise mismatch()
        dimensionality = _SINGLE if value.kind == "File" else _MULTIPART
        return literals_pb2.Literal(
            scalar=literals_pb2.Scalar(
                blob=literals_pb2.Blob(
                    metadata=literals_pb2.BlobMetadata(
                        type=types_pb2.BlobType(format=value.format, dimensionality=dimensionality)
                    ),
                    uri=value.path,
                )
            ),
            hash=value.hash or "",
        )
    if kind == "union_type":
        chosen = _union_variant(value, literal_type.union_type.variants)
        if chosen is None:
            raise mismatch()
        variant, inner = chosen
        return literals_pb2.Literal(scalar=literals_pb2.Scalar(union=literals_pb2.Union(value=inner, type=variant)))
    raise ConversionError.unsupported_type(describe(literal_type))


def _is_exactly(value: Any, variant: types_pb2.LiteralType) -> bool:
    kind = variant.WhichOneof("type")
    if kind == "simple":
        simple = variant.simple
        is_bool = isinstance(value, bool)
        return (
            (simple == types_pb2.INTEGER and isinstance(value, int) and not is_bool)
            or (simple == types_pb2.FLOAT and isinstance(value, float))
            or (simple == types_pb2.STRING and isinstance(value, str))
            or (simple == types_pb2.BOOLEAN and is_bool)
            or (simple == types_pb2.NONE and value is None)
        )
    if kind == "blob" and isinstance(value, Blob):
        return value.kind == _blob_kind(variant.blob)
    if isinstance(value, Opaque):
        return _opaque_literal(value.literal, variant) is not None
    return False


def _union_variant(
    value: Any, variants: Sequence[types_pb2.LiteralType]
) -> Optional[Tuple[types_pb2.LiteralType, literals_pb2.Literal]]:
    """The variant *value* is passed as: one it is exactly, or else the only one it converts to."""
    for variant in variants:
        if _is_exactly(value, variant):
            return variant, to_literal(value, variant)
    fits_: List[Tuple[types_pb2.LiteralType, literals_pb2.Literal]] = []
    for variant in variants:
        try:
            fits_.append((variant, to_literal(value, variant)))
        except ConversionError:
            continue
        if len(fits_) == 2:
            raise ConversionError.ambiguous(name_of(value), f"{describe(fits_[0][0])} and {describe(fits_[1][0])}")
    return fits_[0] if fits_ else None


def fits(value: Any, literal_type: types_pb2.LiteralType) -> bool:
    """Whether *value* converts to *literal_type*, or would but for which union variant it is."""
    try:
        to_literal(value, literal_type)
    except ConversionError as error:
        return error.ambiguous_union
    return True


def from_literal(literal: literals_pb2.Literal) -> Any:
    """The value of *literal*: plain data, a `Blob`, or, for a dataframe or a struct
    (a Pydantic model or dataclass), an `Opaque` holding the literal whole."""
    kind = literal.WhichOneof("value")
    if kind is None:
        return None
    if kind == "collection":
        return [from_literal(item) for item in literal.collection.literals]
    if kind == "map":
        return {key: from_literal(literal.map.literals[key]) for key in sorted(literal.map.literals)}
    if kind == "offloaded_metadata":
        raise ConversionError.unsupported_literal("offloaded")
    scalar = literal.scalar
    scalar_kind = scalar.WhichOneof("value")
    if scalar_kind == "primitive":
        primitive_kind = scalar.primitive.WhichOneof("value")
        if primitive_kind in ("integer", "float_value", "string_value", "boolean"):
            return getattr(scalar.primitive, primitive_kind)
        raise ConversionError.unsupported_literal(primitive_kind or "empty primitive")
    if scalar_kind == "none_type":
        return None
    if scalar_kind == "union":
        if not scalar.union.HasField("value"):
            raise ConversionError.unsupported_literal("empty union")
        return from_literal(scalar.union.value)
    if scalar_kind == "blob":
        blob = scalar.blob
        blob_kind = (
            _blob_kind(blob.metadata.type) if blob.HasField("metadata") and blob.metadata.HasField("type") else None
        )
        if blob_kind is None:
            raise ConversionError.unsupported_literal("blob of unknown dimensionality")
        return Blob(kind=blob_kind, path=blob.uri, format=blob.metadata.type.format, hash=literal.hash or None)
    if scalar_kind in ("structured_dataset", "binary", "generic"):
        return Opaque(literal.SerializeToString(deterministic=True))
    raise ConversionError.unsupported_literal(scalar_kind or "empty scalar")


class CallError(Exception):
    """A task call the worker fails before launching it, or whose result it cannot read: raised
    in the source as `RuntimeError("<code>: <message>")`. For the orchestrator's own result,
    the run fails with *code*."""

    def __init__(self, message: str, code: str = "TypeError"):
        super().__init__(message)
        self.code = code


def bind_inputs(
    task: str,
    declared: Sequence[interface_pb2.VariableEntry],
    parameters: Optional[Sequence[str]],
    defaults: Mapping[str, literals_pb2.Literal],
    args: Sequence[Any],
    kwargs: Sequence[Tuple[str, Any]],
) -> List[Tuple[str, literals_pb2.Literal]]:
    """Binds a call's arguments to the task's *declared* inputs (sorted by name, as in its spec),
    positionally in the order of *parameters*, its signature's, filling the ones not passed from
    *defaults*. Returns the inputs in the order the task declares them."""
    if len(args) > len(declared):
        raise CallError(f"{task}() takes {len(declared)} arguments but {len(args)} were given")
    if parameters is not None:
        positional = list(parameters)
    elif len(declared) <= 1 or not args:
        positional = [entry.key for entry in declared]
    else:
        keywords = ", ".join(f"{entry.key}=..." for entry in declared)
        raise CallError(
            f"{task}() takes its arguments by keyword, as the order of its inputs is not "
            f"known: call it as {task}({keywords})"
        )
    passed: List[Tuple[str, Any]] = []
    for index, value in enumerate(args):
        if index >= len(positional):
            raise CallError(f"{task}() takes {len(positional)} positional arguments but {len(args)} were given")
        passed.append((positional[index], value))
    for name, value in kwargs:
        if any(bound == name for bound, _ in passed):
            raise CallError(f"{task}() got multiple values for argument '{name}'")
        passed.append((name, value))
    for name, _ in passed:
        if not any(entry.key == name for entry in declared):
            raise CallError(f"{task}() got an unexpected keyword argument '{name}'")
    ordered: List[interface_pb2.VariableEntry] = []
    if parameters is not None and len(parameters) == len(declared):
        ordered = [entry for name in parameters for entry in declared if entry.key == name]
    if len(ordered) != len(declared):
        ordered = list(declared)
    literals: List[Tuple[str, literals_pb2.Literal]] = []
    for entry in ordered:
        value = next((value for name, value in passed if name == entry.key), _MISSING)
        if value is not _MISSING:
            try:
                if not entry.HasField("value") or not entry.value.HasField("type"):
                    raise CallError(f"{task}() declares no type for '{entry.key}'")
                literal = to_literal(value, entry.value.type)
            except ConversionError as error:
                raise CallError(f"{task}() argument '{entry.key}': {error}") from None
        elif entry.key in defaults:
            literal = defaults[entry.key]
        else:
            raise CallError(f"{task}() is missing the argument '{entry.key}'")
        literals.append((entry.key, literal))
    return literals


_MISSING = object()


def read_outputs(
    task: str,
    declared: Sequence[interface_pb2.VariableEntry],
    outputs: Sequence[Tuple[str, Optional[literals_pb2.Literal]]],
) -> Any:
    """A task's outputs as the value its call returns: the value for one output, a list for several."""
    try:
        return _read_outputs(declared, outputs)
    except CallError as error:
        raise CallError(f"the result of {task}: {error}") from None


def _read_outputs(
    declared: Sequence[interface_pb2.VariableEntry],
    outputs: Sequence[Tuple[str, Optional[literals_pb2.Literal]]],
) -> Any:
    for name, _ in outputs:
        if not any(entry.key == name for entry in declared):
            raise CallError(f"the task returned an undeclared output '{name}'")
    values = []
    for entry in declared:
        literal = next((literal for name, literal in outputs if name == entry.key), None)
        if literal is None:
            raise CallError(f"the task did not return the output '{entry.key}'")
        try:
            value = from_literal(literal)
        except ConversionError as error:
            raise CallError(f"output '{entry.key}': {error}") from None
        if not entry.HasField("value") or not entry.value.HasField("type"):
            raise CallError(f"the task declares no type for output '{entry.key}'")
        if not fits(value, entry.value.type):
            raise CallError(f"output '{entry.key}' is not of the declared type {describe(entry.value.type)}")
        values.append(value)
    return values[0] if len(values) == 1 else values


def encode_outputs(
    declared: Sequence[interface_pb2.VariableEntry], value: Any
) -> List[Tuple[str, literals_pb2.Literal]]:
    """The orchestrator's result as its declared outputs: the whole result for one output, one
    element each for several. Raises `CallError` with the run's `TypeError` message."""
    if len(declared) == 1:
        values = [value]
    elif isinstance(value, list) and len(value) == len(declared):
        values = value
    else:
        raise CallError(f"the orchestrator declares {len(declared)} outputs")
    literals = []
    for entry, item in zip(declared, values):
        if not entry.HasField("value") or not entry.value.HasField("type"):
            raise CallError("an output has no type", code="BadTaskSpecification")
        try:
            literals.append((entry.key, to_literal(item, entry.value.type)))
        except ConversionError as error:
            raise CallError(f"the result: {error}") from None
    return literals
