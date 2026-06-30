"""Convert JSON-like values to/from native Python types via the Flyte type engine.

Shared by the CLI (``flyte run`` JSON params) and agent tool I/O. LLM tool
schemas and CLI JSON both use the Flyte JSON-schema shape (``uri`` for blobs,
etc.), which differs from Python model fields (``path`` on ``File``/``Dir``).
The type engine already knows how to bridge Literal ↔ native; this module
constructs Literals from JSON dicts and projects Literals back to tool/CLI JSON.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Callable
from typing import Any, Union, get_args, get_origin

from flyteidl2.core import literals_pb2, types_pb2

from flyte.io import DataFrame, Dir, File

# Optional hook for CLI/local path strings → File/Dir/DataFrame (upload, validation).
StringBlobConverter = Callable[[str, Any], Any]


def unwrap_optional_type(tp: Any) -> Any:
    """Strip ``Optional[T]`` / ``Annotated`` wrappers."""
    from flyte.types._type_engine import get_underlying_type

    tp = get_underlying_type(tp)
    origin = get_origin(tp)
    if origin is Union:
        args = typing.get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return unwrap_optional_type(non_none[0])
    return tp


def json_dict_to_literal(value: dict[str, Any], lt: types_pb2.LiteralType) -> literals_pb2.Literal:
    """Build a Flyte ``Literal`` from a JSON-schema-shaped dict."""
    hash_val = value.get("hash") or ""

    if lt.HasField("blob"):
        uri = value.get("uri") or value.get("path") or ""
        fmt = value.get("format")
        if fmt is None:
            fmt = lt.blob.format or ""
        return literals_pb2.Literal(
            scalar=literals_pb2.Scalar(
                blob=literals_pb2.Blob(
                    metadata=literals_pb2.BlobMetadata(
                        type=types_pb2.BlobType(format=fmt, dimensionality=lt.blob.dimensionality)
                    ),
                    uri=uri,
                )
            ),
            hash=hash_val or None,
        )

    if lt.HasField("structured_dataset_type"):
        uri = value.get("uri") or ""
        fmt = value.get("format")
        if fmt is None:
            fmt = ""
        return literals_pb2.Literal(
            scalar=literals_pb2.Scalar(
                structured_dataset=literals_pb2.StructuredDataset(
                    metadata=literals_pb2.StructuredDatasetMetadata(
                        structured_dataset_type=types_pb2.StructuredDatasetType(format=fmt)
                    ),
                    uri=uri,
                )
            ),
            hash=hash_val or None,
        )

    raise TypeError(f"Cannot construct Literal from JSON dict for literal type {lt}")


def literal_to_json_dict(lit: literals_pb2.Literal, lt: types_pb2.LiteralType) -> dict[str, Any]:
    """Project a Flyte ``Literal`` to the JSON-schema shape tools/CLI expose."""
    if lt.HasField("blob"):
        dim = lit.scalar.blob.metadata.type.dimensionality
        dim_str = "MULTIPART" if dim == types_pb2.BlobType.BlobDimensionality.MULTIPART else "SINGLE"
        out: dict[str, Any] = {
            "uri": lit.scalar.blob.uri,
            "format": lit.scalar.blob.metadata.type.format or "",
            "dimensionality": dim_str,
        }
        if lit.hash:
            out["hash"] = lit.hash
        return out

    if lt.HasField("structured_dataset_type"):
        sd = lit.scalar.structured_dataset
        out = {
            "uri": sd.uri,
            "format": sd.metadata.structured_dataset_type.format or "",
        }
        if lit.hash:
            out["hash"] = lit.hash
        return out

    raise TypeError(f"Cannot project Literal to JSON dict for literal type {lt}")


def _supplement_io_metadata(out: dict[str, Any], value: Any) -> dict[str, Any]:
    """Add Python-only metadata (e.g. ``name``) not stored on the Literal."""
    if isinstance(value, (File, Dir)) and value.name:
        out = dict(out)
        out["name"] = value.name
    return out


def _literal_type_is_io(lt: types_pb2.LiteralType) -> bool:
    return lt.HasField("blob") or lt.HasField("structured_dataset_type")


def _is_io_python_type(py_type: Any) -> bool:
    if not isinstance(py_type, type):
        return False
    from flyte.types._type_engine import TypeEngine

    try:
        return _literal_type_is_io(TypeEngine.to_literal_type(py_type))
    except Exception:
        return py_type in (File, Dir, DataFrame)


async def coerce_json_value(
    value: Any,
    py_type: Any,
    *,
    string_blob_converter: StringBlobConverter | None = None,
) -> Any:
    """Coerce a JSON-like value into *py_type* using the Flyte type engine."""
    if value is None:
        return None

    py_type = unwrap_optional_type(py_type)
    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin is list and isinstance(value, list):
        elem_type = args[0] if args else Any
        return [await coerce_json_value(v, elem_type, string_blob_converter=string_blob_converter) for v in value]

    if origin is dict and isinstance(value, dict):
        val_type = args[1] if len(args) >= 2 else Any
        return {
            k: await coerce_json_value(v, val_type, string_blob_converter=string_blob_converter)
            for k, v in value.items()
        }

    if isinstance(value, dict) and isinstance(py_type, type) and dataclasses.is_dataclass(py_type):
        field_types = typing.get_type_hints(py_type, include_extras=True)
        field_names = {f.name for f in dataclasses.fields(py_type)}
        return py_type(
            **{
                name: await coerce_json_value(
                    val,
                    field_types.get(name, type(val)),
                    string_blob_converter=string_blob_converter,
                )
                for name, val in value.items()
                if name in field_names
            }
        )

    if isinstance(py_type, type) and isinstance(value, py_type):
        return value

    if isinstance(value, str):
        if string_blob_converter is not None and _is_io_python_type(unwrap_optional_type(py_type)):
            return string_blob_converter(value, py_type)
        if py_type is File:
            return File(path=value)
        if py_type is Dir:
            return Dir(path=value)
        if py_type is DataFrame:
            return DataFrame.from_existing_remote(remote_path=value)

    if isinstance(value, dict) and isinstance(py_type, type):
        from flyte.types._type_engine import TypeEngine

        lt = TypeEngine.to_literal_type(py_type)
        if lt.HasField("blob") or lt.HasField("structured_dataset_type"):
            lit = json_dict_to_literal(value, lt)
            return await TypeEngine.to_python_value(lit, py_type)

        # Pydantic models, dataclass-backed structs, etc.: let the transformer coerce.
        lm = await TypeEngine.dict_to_literal_map({"_": value}, {"_": py_type})
        kwargs = await TypeEngine.literal_map_to_kwargs(lm, {"_": py_type})
        return kwargs["_"]

    return value


def coerce_json_value_sync(
    value: Any,
    py_type: Any,
    *,
    string_blob_converter: StringBlobConverter | None = None,
) -> Any:
    """Synchronous wrapper around :func:`coerce_json_value` for CLI use."""
    from flyte._utils.asyn import run_sync

    return run_sync(coerce_json_value, value, py_type, string_blob_converter=string_blob_converter)


def serialize_json_value_sync(value: Any, py_type: Any | None = None) -> Any:
    """Synchronous wrapper around :func:`serialize_json_value`."""
    from flyte._utils.asyn import run_sync

    return run_sync(serialize_json_value, value, py_type)


async def coerce_json_args(
    args: dict[str, Any],
    inputs: dict[str, tuple[Any, Any]],
) -> dict[str, Any]:
    """Coerce a kwargs dict using a :class:`~flyte.models.NativeInterface` inputs map."""
    coerced: dict[str, Any] = {}
    for name, value in args.items():
        input_info = inputs.get(name)
        if input_info is None:
            coerced[name] = value
            continue
        py_type, _default = input_info
        if py_type is inspect.Parameter.empty:
            coerced[name] = value
            continue
        coerced[name] = await coerce_json_value(value, py_type)
    return coerced


async def serialize_json_value(value: Any, py_type: Any | None = None) -> Any:
    """Serialize a native value to a JSON-schema-compatible structure."""
    if value is None:
        return None

    if py_type is None:
        py_type = type(value)

    py_type = unwrap_optional_type(py_type)

    if isinstance(py_type, type):
        from flyte.types._type_engine import TypeEngine

        try:
            lt = TypeEngine.to_literal_type(py_type)
            if lt.HasField("blob") or lt.HasField("structured_dataset_type"):
                lit = await TypeEngine.to_literal(value, py_type, lt)
                return _supplement_io_metadata(literal_to_json_dict(lit, lt), value)
        except Exception:
            pass

    origin = get_origin(py_type)

    if origin is list and isinstance(value, list):
        elem_type = get_args(py_type)[0] if get_args(py_type) else Any
        return [await serialize_json_value(v, elem_type) for v in value]

    if origin is dict and isinstance(value, dict):
        args = get_args(py_type)
        val_type = args[1] if len(args) >= 2 else Any
        return {k: await serialize_json_value(v, val_type) for k, v in value.items()}

    if dataclasses.is_dataclass(py_type) and dataclasses.is_dataclass(value) and not isinstance(value, type):
        field_types = typing.get_type_hints(py_type, include_extras=True)
        return {
            f.name: await serialize_json_value(
                getattr(value, f.name), field_types.get(f.name, type(getattr(value, f.name)))
            )
            for f in dataclasses.fields(value)
        }

    from pydantic import BaseModel

    if isinstance(value, BaseModel):
        model_type = py_type if isinstance(py_type, type) else type(value)
        return await serialize_json_value(value.model_dump(), model_type)

    return value
