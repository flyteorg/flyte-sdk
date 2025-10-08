import base64
import json
from typing import Any, Dict, Union

import msgpack
from flyteidl2.core import literals_pb2
from flyteidl2.task import common_pb2
from google.protobuf.json_format import MessageToDict


def _primitive_to_string(primitive: literals_pb2.Primitive) -> Any:
    """
    This method is used to convert a primitive to a string representation.
    """
    match primitive.WhichOneof("value"):
        case "integer":
            return primitive.integer
        case "float_value":
            return primitive.float_value
        case "boolean":
            return primitive.boolean
        case "string_value":
            return primitive.string_value
        case "datetime":
            return primitive.datetime.ToDatetime().isoformat()
        case "duration":
            return primitive.duration.ToSeconds()
        case _:
            raise ValueError(f"Unknown primitive type {primitive}")


def _scalar_to_string(scalar: literals_pb2.Scalar) -> Any:
    """
    This method is used to convert a scalar to a string representation.
    """
    match scalar.WhichOneof("value"):
        case "primitive":
            return _primitive_to_string(scalar.primitive)
        case "none_type":
            return None
        case "error":
            return scalar.error.message
        case "structured_dataset":
            return scalar.structured_dataset.uri
        case "schema":
            return scalar.schema.uri
        case "blob":
            return scalar.blob.uri
        case "binary":
            if scalar.binary.tag == "msgpack":
                return json.dumps(msgpack.unpackb(scalar.binary.value))
            return base64.b64encode(scalar.binary.value)
        case "generic":
            return MessageToDict(scalar.generic)
        case "union":
            return _literal_string_repr(scalar.union.value)
        case _:
            raise ValueError(f"Unknown scalar type {scalar}")


def _literal_string_repr(lit: literals_pb2.Literal) -> Any:
    """
    This method is used to convert a literal to a string representation. This is useful in places, where we need to
    use a shortened string representation of a literal, especially a FlyteFile, FlyteDirectory, or StructuredDataset.
    """
    match lit.WhichOneof("value"):
        case "scalar":
            return _scalar_to_string(lit.scalar)
        case "collection":
            return [literal_string_repr(i) for i in lit.collection.literals]
        case "map":
            return {k: literal_string_repr(v) for k, v in lit.map.literals.items()}
        case "offloaded_metadata":
            # TODO: load literal from offloaded literal?
            return f"Offloaded literal metadata: {lit.offloaded_metadata}"
        case _:
            raise ValueError(f"Unknown literal type {lit}")


def _dict_literal_repr(lmd: Dict[str, literals_pb2.Literal]) -> Dict[str, Any]:
    """
    This method is used to convert a literal map to a string representation.
    """
    return {k: _literal_string_repr(v) for k, v in lmd.items()}


def literal_string_repr(
    lm: Union[
        literals_pb2.Literal,
        common_pb2.NamedLiteral,
        common_pb2.Inputs,
        common_pb2.Outputs,
        literals_pb2.LiteralMap,
        Dict[str, literals_pb2.Literal],
    ],
) -> Dict[str, Any]:
    """
    This method is used to convert a literal map to a string representation.
    """
    if lm is None:
        return {}
    match lm:
        case literals_pb2.Literal():
            return _literal_string_repr(lm)
        case literals_pb2.LiteralMap():
            return _dict_literal_repr(lm.literals)
        case common_pb2.NamedLiteral():
            lmd = {lm.name: lm.value}
            return _dict_literal_repr(lmd)
        case common_pb2.Inputs():
            lmd = {n.name: n.value for n in lm.literals}
            return _dict_literal_repr(lmd)
        case common_pb2.Outputs():
            lmd = {n.name: n.value for n in lm.literals}
            return _dict_literal_repr(lmd)
        case dict():
            return _dict_literal_repr(lm)
        case _:
            raise ValueError(f"Unknown literal type {lm}, type{type(lm)}")
