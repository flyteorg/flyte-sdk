"""
Transformers for Tuple, NamedTuple, and TypedDict types.

These transformers delegate to PydanticTransformer by wrapping values in dynamically
generated Pydantic BaseModel classes.
"""

from __future__ import annotations

import dataclasses
import typing
from abc import abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Tuple, Type, TypedDict, is_typeddict

from flyteidl2.core.literals_pb2 import Literal
from flyteidl2.core.types_pb2 import LiteralType, SimpleType
from pydantic import BaseModel
from typing_extensions import NotRequired, Required, get_args, get_origin

from ._type_engine import PydanticTransformer, TypeTransformer, TypeTransformerFailedError

T = typing.TypeVar("T")
TITLE = "title"


def _json_schema_basic_type_to_python(prop_type: Optional[str]) -> Optional[Type]:
    """Convert JSON schema basic types to Python types."""
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    return type_map.get(prop_type) if prop_type else None


def _json_schema_to_python_type(
    prop_schema: dict,
    defs: dict,
    *,
    recursive_handler: typing.Callable[[dict, dict], Type],
    prefix_items_handler: typing.Callable[[dict, dict], Type],
    ref_handler: typing.Callable[[dict, str, dict], Type],
) -> Type:
    """
    Convert a JSON schema property to a Python type.

    This is a shared helper for TupleTransformer and NamedTupleTransformer
    that handles the common cases of schema-to-type conversion.

    Args:
        prop_schema: The JSON schema for the property
        defs: The $defs section of the schema
        recursive_handler: Callback for recursive type resolution
        prefix_items_handler: Callback to handle prefixItems (tuple vs NamedTuple)
        ref_handler: Callback to handle $ref resolution (transformer-specific)

    Returns:
        The Python type corresponding to the schema.
    """
    # Handle $ref for nested types
    if "$ref" in prop_schema:
        ref_path = prop_schema["$ref"]
        ref_name = ref_path.split("/")[-1]
        if ref_name in defs:
            ref_schema = defs[ref_name].copy()
            # Include $defs for nested refs
            if "$defs" not in ref_schema and defs:
                ref_schema["$defs"] = defs
            return ref_handler(ref_schema, ref_name, defs)
        return Any

    # Handle basic types
    prop_type = prop_schema.get("type")
    basic_type = _json_schema_basic_type_to_python(prop_type)
    if basic_type is not None:
        return basic_type

    if prop_type == "array":
        # Handle tuple types (prefixItems)
        if "prefixItems" in prop_schema:
            return prefix_items_handler(prop_schema, defs)
        # Handle list types
        if "items" in prop_schema:
            item_type = recursive_handler(prop_schema["items"], defs)
            return typing.List[item_type]  # type: ignore
        return typing.List[typing.Any]  # type: ignore

    if prop_type == "object":
        # Handle dict types with typed values
        if "additionalProperties" in prop_schema:
            val_type = recursive_handler(prop_schema["additionalProperties"], defs)
            return typing.Dict[str, val_type]  # type: ignore
        # Handle nested dataclass-like objects with properties
        if "properties" in prop_schema:
            title = prop_schema.get("title", "NestedObject")
            return convert_mashumaro_json_schema_to_python_class(prop_schema, title)
        # Untyped dict
        return typing.Dict[str, typing.Any]  # type: ignore

    # Default to Any for unknown types
    return Any


def _is_typed_dict(t: Type) -> bool:
    """
    Check if a type is a TypedDict.

    Uses both typing.is_typeddict and typing_extensions.is_typeddict to detect
    TypedDict types, since typing.is_typeddict doesn't recognize TypedDicts
    created with typing_extensions.TypedDict on Python < 3.12.
    """
    try:
        if is_typeddict(t):
            return True
    except TypeError:
        pass

    # Also check with typing_extensions for TypedDicts created with typing_extensions
    try:
        from typing_extensions import is_typeddict as te_is_typeddict

        return te_is_typeddict(t)
    except (ImportError, TypeError):
        return False


def _is_named_tuple(t: Type) -> bool:
    """
    Check if a type is a NamedTuple.

    NamedTuple doesn't support isinstance checks directly, so we need to check for
    specific attributes that indicate a NamedTuple type.
    """
    try:
        # Check for the essential NamedTuple characteristics:
        # 1. It's a subclass of tuple
        # 2. It has _fields attribute (tuple of field names)
        # 3. It has _field_types or __annotations__ (mapping of field names to types)
        return (
            isinstance(t, type)
            and issubclass(t, tuple)
            and hasattr(t, "_fields")
            and isinstance(t._fields, tuple)
            and hasattr(t, "__annotations__")
        )
    except TypeError:
        return False


def _is_typed_tuple(t: Type) -> bool:
    """
    Check if a type is a typed tuple (e.g., tuple[int, str] or typing.Tuple[int, str]).

    This excludes NamedTuple and untyped tuple.
    """
    if _is_named_tuple(t):
        return False

    origin = get_origin(t)
    if origin is tuple or origin is Tuple:
        args = get_args(t)
        # Exclude empty tuples and variable-length tuples (tuple[int, ...])
        if args and not (len(args) == 2 and args[1] is ...):
            return True
    return False


# Forward declaration - will be imported lazily to avoid circular import
def convert_mashumaro_json_schema_to_python_class(schema: dict, schema_name: typing.Any) -> Type[T]:
    """Import and call the actual function from _type_engine."""
    from ._type_engine import convert_mashumaro_json_schema_to_python_class as _convert

    return _convert(schema, schema_name)


class PydanticWrappingTransformer(TypeTransformer[T]):
    """
    Base class for transformers that wrap values in dynamically generated Pydantic BaseModel classes.
    values in dynamically generated Pydantic BaseModel classes.

    This base class provides common functionality for wrapping values in dynamically generated Pydantic BaseModel
    classes.
    """

    def __init__(self, name: str, python_type: Type[T]):
        super().__init__(name, python_type, enable_type_assertions=False)
        self._pydantic_transformer = PydanticTransformer()

    @abstractmethod
    def _create_wrapper_model(self, t: Type) -> Type[BaseModel]:
        """Create a Pydantic model that wraps the given type."""
        raise NotImplementedError

    @abstractmethod
    def _value_to_model(self, python_val: T, model_class: Type[BaseModel], python_type: Type) -> BaseModel:
        """Convert a Python value to a Pydantic model instance."""
        raise NotImplementedError

    @abstractmethod
    def _model_to_value(self, model_instance: BaseModel, expected_type: Type) -> T:
        """Convert a Pydantic model instance back to the expected Python type."""
        raise NotImplementedError

    @abstractmethod
    def _handle_prefix_items(self, prop_schema: dict, defs: dict) -> Type:
        """Handle prefixItems schema - specific to tuple/namedtuple."""
        raise NotImplementedError

    @abstractmethod
    def _handle_ref(self, ref_schema: dict, ref_name: str, defs: dict) -> Type:
        """Handle $ref schema resolution - specific to each transformer."""
        raise NotImplementedError

    def _convert_value_for_pydantic(self, value: Any, field_type: Optional[Type] = None) -> Any:
        """
        Recursively convert values to a format that Pydantic can validate.

        This handles:
        - NamedTuple instances -> dict (so Pydantic can validate and construct)
        - Pydantic BaseModel instances -> dict (for nested wrappers)
        - Dataclass instances -> dict (for nested dataclasses)
        - Lists/tuples containing the above
        - Dicts containing the above
        """
        if isinstance(value, BaseModel):
            # Convert Pydantic models to dict
            return value.model_dump()
        elif hasattr(value, "_asdict"):
            # Convert NamedTuple to dict, recursively handling nested values
            data = value._asdict()
            return {k: self._convert_value_for_pydantic(v) for k, v in data.items()}
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            # Convert dataclass instances to dict, recursively handling nested values
            data = dataclasses.asdict(value)
            return {k: self._convert_value_for_pydantic(v) for k, v in data.items()}
        elif isinstance(value, dict):
            # Recursively handle dict values
            return {k: self._convert_value_for_pydantic(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)) and not hasattr(value, "_fields"):
            # Handle lists and regular tuples (but not NamedTuples)
            return [self._convert_value_for_pydantic(v) for v in value]
        return value

    def _convert_value_from_pydantic(self, value: Any, expected_type: Optional[Type] = None) -> Any:
        """
        Recursively convert values from Pydantic model format back to their expected types.

        Subclasses should override this method to handle their specific type conversions.
        """
        # Handle generic containers
        origin = get_origin(expected_type) if expected_type else None
        args = get_args(expected_type) if expected_type else ()

        if isinstance(value, list) and origin is list and args:
            return [self._convert_value_from_pydantic(v, args[0]) for v in value]
        elif isinstance(value, dict) and origin is dict and len(args) >= 2:
            return {k: self._convert_value_from_pydantic(v, args[1]) for k, v in value.items()}
        elif isinstance(value, (list, tuple)) and (origin is tuple or origin is Tuple) and args:
            return tuple(self._convert_value_from_pydantic(v, t) for v, t in zip(value, args))

        return value

    def get_literal_type(self, t: Type) -> LiteralType:
        """Get the literal type by delegating to PydanticTransformer."""
        model_class = self._create_wrapper_model(t)
        return self._pydantic_transformer.get_literal_type(model_class)

    async def to_literal(self, python_val: T, python_type: Type, expected: LiteralType) -> Literal:
        """Convert a Python value to a Flyte Literal by delegating to PydanticTransformer."""
        model_class = self._create_wrapper_model(python_type)
        model_instance = self._value_to_model(python_val, model_class, python_type)
        return await self._pydantic_transformer.to_literal(model_instance, model_class, expected)

    async def to_python_value(self, lv: Literal, expected_python_type: Type) -> T:
        """Convert a Flyte Literal back to a Python value by delegating to PydanticTransformer."""
        model_class = self._create_wrapper_model(expected_python_type)
        model_instance = await self._pydantic_transformer.to_python_value(lv, model_class)
        return self._model_to_value(model_instance, expected_python_type)

    def _get_schema_metadata(self, literal_type: LiteralType) -> Optional[dict]:
        """Extract schema metadata from a literal type if available."""
        if literal_type.simple == SimpleType.STRUCT and literal_type.HasField("metadata"):
            from google.protobuf import json_format

            return json_format.MessageToDict(literal_type.metadata)
        return None

    def _schema_to_type(self, prop_schema: dict, defs: dict) -> Type:
        """Convert a JSON schema property to a Python type."""
        return _json_schema_to_python_type(
            prop_schema,
            defs,
            recursive_handler=self._schema_to_type,
            prefix_items_handler=self._handle_prefix_items,
            ref_handler=self._handle_ref,
        )


class TupleTransformer(PydanticWrappingTransformer[tuple]):
    """
    Transformer that handles typed tuples like tuple[int, str, float].

    This transformer delegates to PydanticTransformer by wrapping the tuple
    in a dynamically generated Pydantic BaseModel.
    """

    _WRAPPER_PREFIX = "TupleWrapper_"

    def __init__(self):
        super().__init__("Typed Tuple", tuple)

    def _create_wrapper_model(self, t: Type[tuple]) -> Type[BaseModel]:
        """Create a Pydantic model that wraps a tuple type."""
        args = get_args(t)
        if not args:
            raise TypeTransformerFailedError("Tuple type must have type arguments")

        # Create field definitions for each tuple element
        field_definitions: Dict[str, Any] = {}
        for i, arg in enumerate(args):
            field_definitions[f"item_{i}"] = (arg, ...)

        # Dynamically create a Pydantic model
        from pydantic import create_model

        model_name = f"{self._WRAPPER_PREFIX}{t.__name__}"
        return create_model(model_name, **field_definitions)

    def _value_to_model(self, python_val: tuple, model_class: Type[BaseModel], python_type: Type) -> BaseModel:
        """Convert a tuple to a Pydantic model instance."""
        field_names = list(model_class.model_fields.keys())
        if len(python_val) != len(field_names):
            raise TypeTransformerFailedError(
                f"Tuple length {len(python_val)} doesn't match expected length {len(field_names)}"
            )
        return model_class(**{field_names[i]: python_val[i] for i in range(len(python_val))})

    def _model_to_value(self, model_instance: BaseModel, expected_type: Type[tuple]) -> tuple:
        """Convert a Pydantic model instance back to a tuple."""
        field_names = list(type(model_instance).model_fields.keys())
        return tuple(getattr(model_instance, name) for name in field_names)

    def guess_python_type(self, literal_type: LiteralType) -> Type[tuple]:
        """
        Guess the Python type from a literal type.

        Detects tuple types by looking for "TupleWrapper_" prefix in the schema title.
        Reconstructs tuple[T1, T2, ...] from the schema properties.
        """
        metadata = self._get_schema_metadata(literal_type)
        if metadata:
            title = metadata.get(TITLE, "")
            if title.startswith(self._WRAPPER_PREFIX):
                return self._create_tuple_from_schema(metadata)

        raise ValueError(f"Tuple transformer cannot reverse {literal_type}")

    def _create_tuple_from_schema(self, schema: dict) -> Type[tuple]:
        """Create a tuple type from a JSON schema."""
        properties = schema.get("properties", {})
        defs = schema.get("$defs", {})

        # Sort by item_N to ensure correct order
        item_fields = [(k, v) for k, v in properties.items() if k.startswith("item_")]
        item_fields.sort(key=lambda x: int(x[0].split("_")[1]))

        element_types: typing.List[Type] = []
        for field_name, field_schema in item_fields:
            element_type = self._schema_to_type(field_schema, defs)
            element_types.append(element_type)

        if not element_types:
            raise ValueError("No tuple elements found in schema")

        return tuple[tuple(element_types)]  # type: ignore

    def _handle_prefix_items(self, prop_schema: dict, defs: dict) -> Type:
        """Handle prefixItems schema as a tuple type."""
        prefix_items = prop_schema["prefixItems"]
        item_types = [self._schema_to_type(item, defs) for item in prefix_items]
        return tuple[tuple(item_types)]  # type: ignore

    def _handle_ref(self, ref_schema: dict, ref_name: str, defs: dict) -> Type:
        """Handle $ref schema for tuple types."""
        ref_title = ref_schema.get("title", ref_name)

        # Check if it's a nested tuple
        if ref_title.startswith(self._WRAPPER_PREFIX):
            return self._create_tuple_from_schema(ref_schema)

        # Otherwise treat as a dataclass
        return convert_mashumaro_json_schema_to_python_class(ref_schema, ref_name)


class NamedTupleTransformer(PydanticWrappingTransformer[tuple]):
    """
    Transformer that handles NamedTuple types.

    This transformer delegates to PydanticTransformer by wrapping the NamedTuple
    in a dynamically generated Pydantic BaseModel.
    """

    _WRAPPER_PREFIX = "NamedTupleWrapper_"

    def __init__(self):
        super().__init__("NamedTuple", tuple)

    def _create_wrapper_model(self, t: Type) -> Type[BaseModel]:
        """Create a Pydantic model that wraps a NamedTuple type."""
        if not _is_named_tuple(t):
            raise TypeTransformerFailedError(f"{t} is not a NamedTuple")

        # Get field names and types from the NamedTuple
        annotations = getattr(t, "__annotations__", {})
        field_definitions: Dict[str, Any] = {}

        for field_name, field_type in annotations.items():
            field_definitions[field_name] = (field_type, ...)

        # Dynamically create a Pydantic model
        from pydantic import create_model

        model_name = f"{self._WRAPPER_PREFIX}{t.__name__}"
        return create_model(model_name, **field_definitions)

    def _value_to_model(self, python_val: tuple, model_class: Type[BaseModel], python_type: Type) -> BaseModel:
        """Convert a NamedTuple to a Pydantic model instance."""
        annotations = getattr(python_type, "__annotations__", {})

        # NamedTuple instances have _asdict() method
        if hasattr(python_val, "_asdict"):
            data = python_val._asdict()
        else:
            # Fallback: use field names from the type
            field_names = python_type._fields
            data = {field_names[i]: python_val[i] for i in range(len(python_val))}

        # Convert nested values that might be NamedTuples or Pydantic models
        # to a format Pydantic can validate (dicts)
        converted_data = {}
        for field_name, value in data.items():
            field_type = annotations.get(field_name)
            converted_data[field_name] = self._convert_value_for_pydantic(value, field_type)

        return model_class(**converted_data)

    def _model_to_value(self, model_instance: BaseModel, expected_type: Type) -> tuple:
        """Convert a Pydantic model instance back to a NamedTuple."""
        field_names = expected_type._fields
        annotations = getattr(expected_type, "__annotations__", {})
        values = []
        for name in field_names:
            value = getattr(model_instance, name)
            field_type = annotations.get(name)
            # Recursively convert nested values
            converted_value = self._convert_value_from_pydantic(value, field_type)
            values.append(converted_value)
        return expected_type(*values)

    def _convert_value_from_pydantic(self, value: Any, expected_type: Optional[Type] = None) -> Any:
        """
        Recursively convert values from Pydantic model format back to their expected types.

        This handles:
        - Pydantic BaseModel instances -> NamedTuple (if expected_type is a NamedTuple)
        - Lists containing the above
        - Dicts containing the above
        """
        if expected_type is not None and _is_named_tuple(expected_type):
            # Expected type is a NamedTuple
            if isinstance(value, BaseModel):
                # Recursively convert nested Pydantic model to NamedTuple
                return self._model_to_value(value, expected_type)
            elif isinstance(value, dict):
                # Convert dict to NamedTuple
                field_names = expected_type._fields
                annotations = getattr(expected_type, "__annotations__", {})
                values = []
                for name in field_names:
                    field_value = value.get(name)
                    field_type = annotations.get(name)
                    values.append(self._convert_value_from_pydantic(field_value, field_type))
                return expected_type(*values)

        return super()._convert_value_from_pydantic(value, expected_type)

    def guess_python_type(self, literal_type: LiteralType) -> Type:
        """
        Guess the Python type from a literal type.

        Creates a dynamic NamedTuple from the schema if the title indicates it's a NamedTuple wrapper.
        """
        metadata = self._get_schema_metadata(literal_type)
        if metadata:
            title = metadata.get(TITLE, "")

            # Check if this is a NamedTuple wrapper
            if title.startswith(self._WRAPPER_PREFIX):
                original_name = title[len(self._WRAPPER_PREFIX) :]
                return self._create_namedtuple_from_schema(metadata, original_name)

        raise ValueError(f"NamedTuple transformer cannot reverse {literal_type}")

    def _create_namedtuple_from_schema(self, schema: dict, name: str) -> Type:
        """Create a dynamic NamedTuple type from a JSON schema."""
        defs = schema.get("$defs", {})

        # Check if this is an array-based NamedTuple representation (from Pydantic)
        # Pydantic serializes NamedTuples as arrays with prefixItems
        if schema.get("type") == "array" and "prefixItems" in schema:
            return self._create_namedtuple_from_array_schema(schema, name, defs)

        # Object-based schema (NamedTupleWrapper format)
        properties = schema.get("properties", {})
        # Use 'required' field to preserve property order if available
        property_order = schema.get("required", list(properties.keys()))

        field_types: typing.List[typing.Tuple[str, Type]] = []
        for prop_name in property_order:
            prop_schema = properties.get(prop_name, {})
            field_type = self._schema_to_type(prop_schema, defs)
            field_types.append((prop_name, field_type))

        # Create a NamedTuple dynamically
        return NamedTuple(name, field_types)  # type: ignore

    def _create_namedtuple_from_array_schema(self, schema: dict, name: str, defs: dict) -> Type:
        """Create a NamedTuple from an array-based JSON schema (Pydantic's NamedTuple format)."""
        prefix_items = schema.get("prefixItems", [])

        field_types: typing.List[typing.Tuple[str, Type]] = []
        for i, item_schema in enumerate(prefix_items):
            # Use the title as field name if available, otherwise use indexed name
            field_name = item_schema.get("title", f"field_{i}").lower().replace(" ", "_")
            field_type = self._schema_to_type(item_schema, defs)
            field_types.append((field_name, field_type))

        return NamedTuple(name, field_types)  # type: ignore

    def _handle_prefix_items(self, prop_schema: dict, defs: dict) -> Type:
        """Handle prefixItems schema as a NamedTuple type."""
        title = prop_schema.get("title", "DynamicNamedTuple")
        return self._create_namedtuple_from_array_schema(prop_schema, title, defs)

    def _handle_ref(self, ref_schema: dict, ref_name: str, defs: dict) -> Type:
        """Handle $ref schema for NamedTuple types."""
        ref_title = ref_schema.get("title", ref_name)

        # Check if it's an array-based NamedTuple (Pydantic serializes nested NamedTuples as arrays)
        if ref_schema.get("type") == "array" and "prefixItems" in ref_schema:
            return self._create_namedtuple_from_array_schema(ref_schema, ref_name, defs)

        # Recursively create nested NamedTuple if it's a wrapper
        if ref_title.startswith(self._WRAPPER_PREFIX):
            return self._create_namedtuple_from_schema(ref_schema, ref_title[len(self._WRAPPER_PREFIX) :])

        # For objects with properties (like dataclasses)
        if ref_schema.get("type") == "object" and "properties" in ref_schema:
            return convert_mashumaro_json_schema_to_python_class(ref_schema, ref_name)

        # Fallback
        return Any


class TypedDictTransformer(PydanticWrappingTransformer[dict]):
    """
    Transformer that handles TypedDict types.

    This transformer delegates to PydanticTransformer by wrapping the TypedDict
    in a dynamically generated Pydantic BaseModel.
    """

    _WRAPPER_PREFIX = "TypedDictWrapper_"

    def __init__(self):
        super().__init__("TypedDict", dict)
        # Instance-level cache for wrapper models to handle self-referential TypedDicts
        self._model_cache: Dict[Type, Type[BaseModel]] = {}

    def assert_type(self, t: Type, v: Any):
        """
        Override assert_type to handle TypedDict properly.

        TypedDict doesn't support isinstance checks, so we validate by checking
        that the value is a dict with the expected keys and value types.
        """
        if not isinstance(v, dict):
            raise TypeTransformerFailedError(f"Expected a dict for TypedDict {t}, but got {type(v)}")

        # Get expected annotations
        annotations = getattr(t, "__annotations__", {})
        required_keys = getattr(t, "__required_keys__", frozenset(annotations.keys()))

        # Check all required keys are present
        for key in required_keys:
            if key not in v:
                raise TypeTransformerFailedError(f"Missing required key '{key}' for TypedDict {t}")

    def _create_wrapper_model(
        self, t: Type, _model_cache: Optional[Dict[Type, Type[BaseModel]]] = None
    ) -> Type[BaseModel]:
        """Create a Pydantic model that wraps a TypedDict type."""
        if not _is_typed_dict(t):
            raise TypeTransformerFailedError(f"{t} is not a TypedDict")

        # Use instance-level cache by default to handle self-referential TypedDicts
        # and persist cache across get_literal_type/to_literal/to_python_value calls
        if _model_cache is None:
            _model_cache = self._model_cache

        if t in _model_cache:
            return _model_cache[t]

        # Get field names and types from the TypedDict
        # Use get_type_hints(include_extras=True) to preserve NotRequired/Required wrappers,
        # which we need to detect optional fields. We can't rely on __required_keys__ because
        # it is wrong when `from __future__ import annotations` is used (all fields appear required).
        try:
            annotations_with_extras = typing.get_type_hints(t, include_extras=True)
        except Exception:
            annotations_with_extras = getattr(t, "__annotations__", {})

        field_definitions: Dict[str, Any] = {}

        # Dynamically create a Pydantic model
        from pydantic import create_model

        model_name = f"{self._WRAPPER_PREFIX}{t.__name__}"

        # Use the model name as a placeholder in the cache before processing fields.
        # For self-referential types (e.g. TreeNode with children: List[TreeNode]),
        # the recursive call to _create_wrapper_model will return this string, which
        # becomes a forward reference (e.g. List["TypedDictWrapper_TreeNode"]) that
        # Pydantic resolves via model_rebuild() after model creation.
        _model_cache[t] = model_name  # type: ignore

        for field_name, field_type in annotations_with_extras.items():
            # Check if the field is NotRequired before unwrapping
            origin = get_origin(field_type)
            is_not_required = origin is NotRequired

            # Unwrap NotRequired and Required type hints to get the inner type
            # These are only used by TypedDict to mark optional/required fields
            # and should not be passed to Pydantic
            inner_type = self._unwrap_typeddict_field_type(field_type)

            # Convert nested TypedDict types to their Pydantic wrapper models
            # This is necessary because isinstance() doesn't work with TypedDict on Python < 3.12
            pydantic_type = self._convert_field_type_for_pydantic(inner_type, _model_cache)

            if is_not_required:
                # Optional fields get a default of None
                field_definitions[field_name] = (typing.Optional[pydantic_type], None)
            else:
                field_definitions[field_name] = (pydantic_type, ...)

        model = create_model(model_name, **field_definitions)
        _model_cache[t] = model

        # Rebuild to resolve any forward references from self-referential types
        rebuild_ns = {f"{self._WRAPPER_PREFIX}{k.__name__}": v for k, v in _model_cache.items() if isinstance(v, type)}
        model.model_rebuild(_types_namespace=rebuild_ns)

        return model

    def _convert_field_type_for_pydantic(self, field_type: Type, _model_cache: Dict[Type, Type[BaseModel]]) -> Type:
        """
        Convert a field type to a Pydantic-compatible type.

        This recursively converts nested TypedDict types to their Pydantic wrapper models,
        which is necessary because isinstance() doesn't work with TypedDict on Python < 3.12.
        """
        # Check if it's a TypedDict - convert to Pydantic wrapper model
        if _is_typed_dict(field_type):
            return self._create_wrapper_model(field_type, _model_cache)

        # Handle generic types (List, Dict, Optional, etc.)
        origin = get_origin(field_type)
        args = get_args(field_type)

        if origin is not None and args:
            # Convert each type argument recursively
            converted_args = tuple(self._convert_field_type_for_pydantic(arg, _model_cache) for arg in args)

            # Reconstruct the generic type with converted arguments
            if origin is list:
                return typing.List[converted_args[0]]  # type: ignore
            elif origin is dict:
                return typing.Dict[converted_args[0], converted_args[1]]  # type: ignore
            elif origin is set:
                return typing.Set[converted_args[0]]  # type: ignore
            elif origin is tuple:
                return typing.Tuple[converted_args]  # type: ignore
            elif origin is typing.Union:
                return typing.Union[converted_args]  # type: ignore
            else:
                # For other generic types, try to reconstruct with __class_getitem__
                try:
                    return origin[converted_args]  # type: ignore
                except TypeError:
                    # If reconstruction fails, return the original type
                    return field_type

        return field_type

    def _unwrap_typeddict_field_type(self, field_type: Type) -> Type:
        """
        Unwrap NotRequired and Required type hints from TypedDict field types.

        TypedDict uses NotRequired[T] and Required[T] to mark optional/required fields,
        but these wrappers should not be passed to Pydantic - only the inner type T.
        """
        origin = get_origin(field_type)
        if origin is NotRequired or origin is Required:
            args = get_args(field_type)
            if args:
                return args[0]
        return field_type

    def _value_to_model(self, python_val: dict, model_class: Type[BaseModel], python_type: Type) -> BaseModel:
        """Convert a TypedDict to a Pydantic model instance."""
        try:
            annotations = typing.get_type_hints(python_type)
        except Exception:
            annotations = getattr(python_type, "__annotations__", {})

        # Convert nested values that might be TypedDicts, dataclasses, or Pydantic models
        # to a format Pydantic can validate (dicts)
        converted_data = {}
        for field_name, value in python_val.items():
            field_type = annotations.get(field_name)
            converted_data[field_name] = self._convert_value_for_pydantic(value, field_type)

        return model_class(**converted_data)

    def _model_to_value(self, model_instance: BaseModel, expected_type: Type) -> dict:
        """Convert a Pydantic model instance back to a TypedDict."""
        try:
            annotations_with_extras = typing.get_type_hints(expected_type, include_extras=True)
        except Exception:
            annotations_with_extras = getattr(expected_type, "__annotations__", {})
        result = {}
        for name, field_type in annotations_with_extras.items():
            if hasattr(model_instance, name):
                value = getattr(model_instance, name)
                # Skip NotRequired fields when value is None
                # This ensures that optional fields not provided in the input
                # are absent from the output dict (not set to None)
                origin = get_origin(field_type)
                if origin is NotRequired and value is None:
                    continue
                # Unwrap NotRequired/Required before converting
                inner_type = self._unwrap_typeddict_field_type(field_type)
                # Recursively convert nested values
                converted_value = self._convert_value_from_pydantic(value, inner_type)
                result[name] = converted_value
        return result

    def _convert_value_from_pydantic(self, value: Any, expected_type: Optional[Type] = None) -> Any:
        """
        Recursively convert values from Pydantic model format back to their expected types.

        This handles:
        - Pydantic BaseModel instances -> TypedDict (if expected_type is a TypedDict)
        - Lists containing the above
        - Dicts containing the above
        """
        # Unwrap NotRequired/Required type hints
        if expected_type is not None:
            origin = get_origin(expected_type)
            if origin is NotRequired or origin is Required:
                args = get_args(expected_type)
                expected_type = args[0] if args else expected_type

        if expected_type is not None and _is_typed_dict(expected_type):
            # Expected type is a TypedDict
            if isinstance(value, BaseModel):
                # Recursively convert nested Pydantic model to TypedDict
                return self._model_to_value(value, expected_type)
            elif isinstance(value, dict):
                # Convert dict to TypedDict format
                annotations = getattr(expected_type, "__annotations__", {})
                result = {}
                for name, field_type in annotations.items():
                    if name in value:
                        result[name] = self._convert_value_from_pydantic(value[name], field_type)
                return result

        return super()._convert_value_from_pydantic(value, expected_type)

    async def to_literal(self, python_val: dict, python_type: Type, expected: LiteralType) -> Literal:
        """Convert a TypedDict to a Flyte Literal."""
        if not isinstance(python_val, dict):
            raise TypeTransformerFailedError(f"Expected a dict but got {type(python_val)}")
        return await super().to_literal(python_val, python_type, expected)

    def guess_python_type(self, literal_type: LiteralType) -> Type:
        """
        Guess the Python type from a literal type.

        Creates a dynamic TypedDict from the schema if the title indicates it's a TypedDict wrapper.
        """
        metadata = self._get_schema_metadata(literal_type)
        if metadata:
            title = metadata.get(TITLE, "")

            # Check if this is a TypedDict wrapper
            if title.startswith(self._WRAPPER_PREFIX):
                original_name = title[len(self._WRAPPER_PREFIX) :]
                return self._create_typeddict_from_schema(metadata, original_name)

        raise ValueError(f"TypedDict transformer cannot reverse {literal_type}")

    def _create_typeddict_from_schema(self, schema: dict, name: str) -> Type:
        """Create a dynamic TypedDict type from a JSON schema."""
        defs = schema.get("$defs", {})

        # Object-based schema (TypedDictWrapper format)
        properties = schema.get("properties", {})

        field_types: typing.Dict[str, Type] = {}
        for prop_name, prop_schema in properties.items():
            field_type = self._schema_to_type(prop_schema, defs)
            field_types[prop_name] = field_type

        # Create a TypedDict dynamically
        # Note: Using TypedDict functional syntax
        return TypedDict(name, field_types)  # type: ignore

    def _schema_to_type(self, prop_schema: dict, defs: dict) -> Type:
        """Convert a JSON schema property to a Python type."""
        # Handle $ref for nested types
        if "$ref" in prop_schema:
            ref_path = prop_schema["$ref"]
            ref_name = ref_path.split("/")[-1]
            if ref_name in defs:
                ref_schema = defs[ref_name].copy()
                ref_schema["$defs"] = defs
                ref_title = ref_schema.get("title", ref_name)

                # Recursively create nested TypedDict if it's a wrapper
                if ref_title.startswith(self._WRAPPER_PREFIX):
                    return self._create_typeddict_from_schema(ref_schema, ref_title[len(self._WRAPPER_PREFIX) :])

                # For objects with properties (like dataclasses)
                if ref_schema.get("type") == "object" and "properties" in ref_schema:
                    return convert_mashumaro_json_schema_to_python_class(ref_schema, ref_name)

                # Fallback
                return Any

        # Handle basic types
        prop_type = prop_schema.get("type")
        if prop_type == "string":
            return str
        elif prop_type == "integer":
            return int
        elif prop_type == "number":
            return float
        elif prop_type == "boolean":
            return bool
        elif prop_type == "array":
            # Regular list
            items = prop_schema.get("items", {})
            item_type = self._schema_to_type(items, defs)
            return typing.List[item_type]  # type: ignore
        elif prop_type == "object":
            # Handle nested objects
            if "additionalProperties" in prop_schema:
                additional_props = prop_schema["additionalProperties"]
                # additionalProperties can be a boolean or a schema dict
                if isinstance(additional_props, dict):
                    value_type = self._schema_to_type(additional_props, defs)
                    return typing.Dict[str, value_type]  # type: ignore
                elif additional_props is True:
                    # True means any additional properties are allowed
                    return typing.Dict[str, Any]  # type: ignore
                # If False, fall through to check properties or return dict
            # For nested dataclass-like objects with properties
            if "properties" in prop_schema:
                title = prop_schema.get("title", "NestedObject")
                return convert_mashumaro_json_schema_to_python_class(prop_schema, title)
            # Untyped dict
            return dict

        # Default fallback
        return Any

    def _handle_prefix_items(self, prop_schema: dict, defs: dict) -> Type:
        """Handle prefixItems schema - TypedDict doesn't use this, delegate to tuple."""
        prefix_items = prop_schema["prefixItems"]
        item_types = [self._schema_to_type(item, defs) for item in prefix_items]
        return tuple[tuple(item_types)]  # type: ignore

    def _handle_ref(self, ref_schema: dict, ref_name: str, defs: dict) -> Type:
        """Handle $ref schema for TypedDict types."""
        ref_title = ref_schema.get("title", ref_name)

        # Recursively create nested TypedDict if it's a wrapper
        if ref_title.startswith(self._WRAPPER_PREFIX):
            return self._create_typeddict_from_schema(ref_schema, ref_title[len(self._WRAPPER_PREFIX) :])

        # For objects with properties (like dataclasses)
        if ref_schema.get("type") == "object" and "properties" in ref_schema:
            return convert_mashumaro_json_schema_to_python_class(ref_schema, ref_name)

        # Fallback
        return Any


__all__ = [
    "NamedTupleTransformer",
    "PydanticWrappingTransformer",
    "TupleTransformer",
    "TypedDictTransformer",
    "_is_named_tuple",
    "_is_typed_dict",
    "_is_typed_tuple",
    "_json_schema_basic_type_to_python",
    "_json_schema_to_python_type",
]
