"""Convert Flyte LiteralType (protobuf) to JSON schema.
"""

from __future__ import annotations

from typing import Any, Dict

from flyteidl2.core import types_pb2
from google.protobuf.json_format import MessageToDict


def literal_type_to_json_schema(lt: types_pb2.LiteralType) -> Dict[str, Any]:
    """Convert a Flyte LiteralType protobuf to a JSON schema dict.
    """
    if lt is None:
        return {"type": "null"}

    if lt.HasField("simple"):
        return _simple_to_json_schema(lt)

    if lt.HasField("collection_type"):
        return {"type": "array", "items": literal_type_to_json_schema(lt.collection_type)}

    if lt.HasField("map_value_type"):
        return {"type": "object", "additionalProperties": literal_type_to_json_schema(lt.map_value_type)}

    if lt.HasField("enum_type"):
        return {"type": "string", "enum": list(lt.enum_type.values)}

    if lt.HasField("blob"):
        return _blob_to_json_schema(lt.blob)

    if lt.HasField("structured_dataset_type"):
        return {
            "type": "object",
            "format": "structured-dataset",
            "properties": {
                "uri": {"type": "string"},
                "format": {"type": "string"},
            },
        }

    if lt.HasField("union_type"):
        return _union_to_json_schema(lt.union_type)

    # schema type is deprecated — fall back to string
    return {"type": "string"}


def _simple_to_json_schema(lt: types_pb2.LiteralType) -> Dict[str, Any]:
    """Convert a simple LiteralType to JSON schema."""
    simple = lt.simple

    if simple == types_pb2.SimpleType.NONE:
        return {"type": "null"}
    if simple == types_pb2.SimpleType.STRING:
        return {"type": "string"}
    if simple == types_pb2.SimpleType.INTEGER:
        return {"type": "integer"}
    if simple == types_pb2.SimpleType.FLOAT:
        return {"type": "number", "format": "float"}
    if simple == types_pb2.SimpleType.BOOLEAN:
        return {"type": "boolean"}
    if simple == types_pb2.SimpleType.DATETIME:
        return {"type": "string", "format": "datetime"}
    if simple == types_pb2.SimpleType.DURATION:
        return {"type": "string", "format": "duration"}
    if simple == types_pb2.SimpleType.BINARY:
        return {"type": "string", "format": "binary"}
    if simple == types_pb2.SimpleType.ERROR:
        return {"type": "object", "format": "error"}
    if simple == types_pb2.SimpleType.STRUCT:
        return _struct_to_json_schema(lt)

    return {"type": "string"}


def _struct_to_json_schema(lt: types_pb2.LiteralType) -> Dict[str, Any]:
    """Convert a STRUCT LiteralType to JSON schema.
    """
    if lt.HasField("metadata"):
        schema = MessageToDict(lt.metadata)
        if schema:
            title = schema.pop("title", None)
            schema.pop("additionalProperties", None)
            if title is not None:
                schema["dataclass"] = title
            return schema

    return {"type": "object"}


def _blob_to_json_schema(blob: types_pb2.BlobType) -> Dict[str, Any]:
    """Convert a BlobType to JSON schema. Used for FlyteFile and FlyteDirectory."""
    uri_prop: Dict[str, Any] = {"type": "string", "default": ""}

    format_prop: Dict[str, Any] = {"type": "string", "default": blob.format if blob.format else ""}

    dim = blob.dimensionality
    if dim == types_pb2.BlobType.BlobDimensionality.MULTIPART:
        dim_default = "MULTIPART"
    else:
        dim_default = "SINGLE"

    dim_prop: Dict[str, Any] = {
        "type": "string",
        "enum": ["SINGLE", "MULTIPART"],
        "default": dim_default,
    }

    return {
        "type": "object",
        "format": "blob",
        "properties": {
            "uri": uri_prop,
            "format": format_prop,
            "dimensionality": dim_prop,
        },
    }


def _union_to_json_schema(union_type: types_pb2.UnionType) -> Dict[str, Any]:
    """Convert a UnionType to JSON schema.

    Optional[X] (Union[X, None] with a single non-null variant) is simplified
    to just X's schema. True multi-type unions use oneOf with format:union.
    """
    # Separate null from non-null variants
    null_simple = types_pb2.SimpleType.NONE
    non_null = [
        v for v in union_type.variants
        if not (v.HasField("simple") and v.simple == null_simple)
    ]

    # Optional[X] — simplify to X's schema (no null variant needed for tool use)
    if len(non_null) == 1:
        return literal_type_to_json_schema(non_null[0])

    # True union — oneOf with title + structure per variant
    one_of = []
    for variant in union_type.variants:
        variant_schema = literal_type_to_json_schema(variant)
        tag = variant.structure.tag if variant.HasField("structure") else ""
        variant_schema["title"] = tag
        variant_schema["structure"] = tag
        one_of.append(variant_schema)

    return {"oneOf": one_of, "format": "union"}