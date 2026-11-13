import dataclasses
import inspect
import typing
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, Type

from flyteidl2.core import interface_pb2, literals_pb2, types_pb2
from flyteidl2.task import common_pb2

from flyte._logging import logger
from flyte.models import NativeInterface

from ._type_engine import TypeEngine, TypeTransformer, TypeTransformerFailedError


class UnresolvedType:
    """
    Stands in for a remote input or output whose Python type could not be rebuilt from its registered schema.

    `guess_interface` uses it so that one type that cannot be rebuilt does not fail the whole interface. An input of
    this type can still be passed a value that carries its own class (see `infer_type_from_value`); converting any
    other value to or from it raises an error naming the variable.
    """

    variable: ClassVar[str]
    literal_type: ClassVar[types_pb2.LiteralType]
    reason: ClassVar[str]


def _unresolved_type(variable: str, literal_type: types_pb2.LiteralType, reason: str) -> Type[UnresolvedType]:
    return type(
        f"Unresolved_{variable}",
        (UnresolvedType,),
        {"variable": variable, "literal_type": literal_type, "reason": reason},
    )


class _UnresolvedTypeTransformer(TypeTransformer[UnresolvedType]):
    """Keeps the registered literal type of an `UnresolvedType` and refuses to convert values through it."""

    def __init__(self):
        super().__init__("Unresolved Remote Type", UnresolvedType, enable_type_assertions=False)

    def get_literal_type(self, t: Type[UnresolvedType]) -> types_pb2.LiteralType:
        return t.literal_type

    @staticmethod
    def _error(t: Type[UnresolvedType]) -> TypeTransformerFailedError:
        return TypeTransformerFailedError(
            f"The type of '{t.variable}' could not be rebuilt from the task's registered schema ({t.reason}). "
            f"Only a Pydantic model or dataclass instance of the registered class can be converted for it."
        )

    async def to_literal(
        self, python_val: Any, python_type: Type[UnresolvedType], expected: types_pb2.LiteralType
    ) -> literals_pb2.Literal:
        raise self._error(python_type)

    async def to_python_value(self, lv: literals_pb2.Literal, expected_python_type: Type[UnresolvedType]) -> Any:
        raise self._error(expected_python_type)


TypeEngine.register(_UnresolvedTypeTransformer())


def _structural_type(literal_type: types_pb2.LiteralType) -> Type:
    """Rebuild `literal_type`, or its structural shape when it cannot be rebuilt (see `_structural_fallback`)."""
    try:
        return TypeEngine.guess_python_type(literal_type)
    except Exception:
        return _structural_fallback(literal_type)


def _structural_fallback(literal_type: types_pb2.LiteralType) -> Type:
    """
    Rebuild the container shape of `literal_type`, turning any STRUCT that cannot be rebuilt into a plain `dict`.

    A STRUCT carries a msgpack payload whether it came from a model, a dataclass or a dict, so an untyped `dict`
    still reads and writes it. Raises `ValueError` for anything else that cannot be rebuilt.
    """
    kind = literal_type.WhichOneof("type")
    if kind == "collection_type":
        return typing.List[_structural_type(literal_type.collection_type)]  # type: ignore
    if kind == "map_value_type":
        return typing.Dict[str, _structural_type(literal_type.map_value_type)]  # type: ignore
    if kind == "union_type":
        return typing.Union[tuple(_structural_type(v) for v in literal_type.union_type.variants)]  # type: ignore
    if kind == "simple" and literal_type.simple == types_pb2.SimpleType.STRUCT:
        return dict
    raise ValueError(f"Cannot rebuild a Python type for {literal_type}")


def _guess_variable_type(variable: str, literal_type: types_pb2.LiteralType) -> Type:
    """Rebuild one variable's type, falling back to its structural shape and then to an `UnresolvedType`."""
    try:
        return TypeEngine.guess_python_type(literal_type)
    except Exception as e:
        reason = f"{type(e).__name__}: {e}"

    try:
        fallback = _structural_fallback(literal_type)
    except Exception:
        logger.warning(f"Could not rebuild a Python type for '{variable}' ({reason}); it only takes a typed value.")
        return _unresolved_type(variable, literal_type, reason)

    logger.warning(f"Could not rebuild a Python type for '{variable}' ({reason}); falling back to {fallback}.")
    return fallback


def _guess_variable_types(variables: Iterable[interface_pb2.VariableEntry]) -> Dict[str, Type]:
    return {entry.key: _guess_variable_type(entry.key, entry.value.type) for entry in variables}


def guess_interface(
    interface: interface_pb2.TypedInterface, default_inputs: Iterable[common_pb2.NamedParameter] | None = None
) -> NativeInterface:
    """
    Returns the interface of the task with guessed types, as types may not be present in current env.

    A variable whose type cannot be rebuilt does not fail the interface: it falls back to its structural shape (a
    STRUCT becomes `dict`), or to an `UnresolvedType` when even that fails.
    """
    if interface is None:
        return NativeInterface({}, {})

    default_input_literals: Dict[str, literals_pb2.Literal] = {}
    if default_inputs is not None:
        for param in default_inputs:
            if param.parameter.HasField("default"):
                default_input_literals[param.name] = param.parameter.default

    guessed_inputs: Dict[str, Tuple[Type[Any], Any] | Any] = {}
    input_literal_types: Dict[str, types_pb2.LiteralType] = {}
    if interface.inputs is not None and len(interface.inputs.variables) > 0:
        input_types = _guess_variable_types(interface.inputs.variables)
        input_literal_types = {entry.key: entry.value.type for entry in interface.inputs.variables}
        for name, t in input_types.items():
            if name not in default_input_literals:
                guessed_inputs[name] = (t, inspect.Parameter.empty)
            else:
                guessed_inputs[name] = (t, NativeInterface.has_default)

    guessed_outputs: Dict[str, Type[Any]] = {}
    if interface.outputs is not None and len(interface.outputs.variables) > 0:
        guessed_outputs = _guess_variable_types(interface.outputs.variables)

    return NativeInterface.from_types(
        guessed_inputs, guessed_outputs, default_input_literals, input_literal_types=input_literal_types
    )


def _is_none_literal_type(literal_type: types_pb2.LiteralType) -> bool:
    return literal_type.WhichOneof("type") == "simple" and literal_type.simple == types_pb2.SimpleType.NONE


def _is_structured_value(value: Any) -> bool:
    from pydantic import BaseModel

    if isinstance(value, type):
        return False
    return isinstance(value, BaseModel) or dataclasses.is_dataclass(value)


def infer_type_from_value(value: Any, literal_type: types_pb2.LiteralType) -> Optional[Type]:
    """
    The Python type to convert `value` with for a variable registered as `literal_type`, taken from the value itself.

    A Pydantic model or dataclass instance carries its own class, which is a better guide than a type rebuilt from
    the registered schema: it is the class that defines the value's fields and validators, and the upload hooks of
    any `flyte.io` values nested in it. Lists, string-keyed dicts and `Optional` around such values are followed too.

    Returns `None` when the value does not determine a type that fits `literal_type`; the caller then falls back to
    the rebuilt type.
    """
    kind = literal_type.WhichOneof("type")
    if kind == "union_type":
        variants = literal_type.union_type.variants
        non_none = [v for v in variants if not _is_none_literal_type(v)]
        # Only Optional[X]: a wider union needs its other variants rebuilt anyway.
        if value is None or len(non_none) != 1 or len(non_none) == len(variants):
            return None
        inner = infer_type_from_value(value, non_none[0])
        return typing.Optional[inner] if inner is not None else None  # type: ignore

    if kind == "collection_type":
        if not isinstance(value, list) or not value:
            return None
        elem_types = {infer_type_from_value(v, literal_type.collection_type) for v in value}
        if len(elem_types) != 1 or None in elem_types:
            return None
        return typing.List[elem_types.pop()]  # type: ignore

    if kind == "map_value_type":
        if not isinstance(value, dict) or not value or not all(isinstance(k, str) for k in value):
            return None
        value_types = {infer_type_from_value(v, literal_type.map_value_type) for v in value.values()}
        if len(value_types) != 1 or None in value_types:
            return None
        return typing.Dict[str, value_types.pop()]  # type: ignore

    if kind == "simple" and literal_type.simple == types_pb2.SimpleType.STRUCT and _is_structured_value(value):
        value_type = type(value)
        # The structure tag names the transformer that registered this type. A value handled by another transformer
        # (a dataclass passed for a model input) would serialize differently, so leave it to the rebuilt type.
        tag = literal_type.structure.tag if literal_type.HasField("structure") else ""
        if tag and TypeEngine.get_transformer(value_type).name != tag:
            return None
        return value_type

    return None
