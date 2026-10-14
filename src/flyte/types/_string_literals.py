import base64
import json
from typing import Any, Dict, Mapping, Union

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


def artifact_annotation(lit: literals_pb2.Literal) -> str | None:
    """
    Human-readable artifact annotation for a literal, or None when the value carries no
    artifact identity. The identity is the typed `core.Literal.artifact_id`, stamped at
    artifact registration and on artifact-bound run inputs, and it travels with the value
    through every copy. (Produced-artifact declarations live on the Outputs envelope —
    see `produced_artifact_annotation`.)
    """
    if not lit.HasField("artifact_id"):
        return None
    key = lit.artifact_id.key
    return f"artifact: {key.org}/{key.project}/{key.domain}/{key.name}@{lit.artifact_id.version}"


def produced_artifact_annotation(decl: common_pb2.ProducedArtifact) -> str | None:
    """
    Human-readable annotation for a produced-artifact declaration carried on the Outputs
    envelope (`task.Outputs.produced_artifacts`).
    """
    name = decl.name
    if not name:
        return None
    if decl.version:
        name = f"{name}@{decl.version}"
    return f"produced artifact: {name}"


def _literal_string_repr(lit: literals_pb2.Literal) -> Any:
    """
    This method is used to convert a literal to a string representation. This is useful in places, where we need to
    use a shortened string representation of a literal, especially a FlyteFile, FlyteDirectory, or StructuredDataset.
    """
    rendered: Any
    match lit.WhichOneof("value"):
        case "scalar":
            rendered = _scalar_to_string(lit.scalar)
        case "collection":
            rendered = [literal_string_repr(i) for i in lit.collection.literals]
        case "map":
            rendered = {k: literal_string_repr(v) for k, v in lit.map.literals.items()}
        case "offloaded_metadata":
            # TODO: load literal from offloaded literal?
            rendered = f"Offloaded literal metadata: {lit.offloaded_metadata}"
        case _:
            raise ValueError(f"Unknown literal type {lit}")

    # Surface artifact markers stamped on the literal (consumed provenance or produced
    # metadata) so `flyte get io` shows the artifact linkage alongside the value.
    if annotation := artifact_annotation(lit):
        return f"{rendered} ({annotation})"
    return rendered


def _dict_literal_repr(lmd: Mapping[str, literals_pb2.Literal]) -> Dict[str, Any]:
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
            rendered = _dict_literal_repr({n.name: n.value for n in lm.literals})
            # Produced-artifact declarations live on the Outputs envelope; surface
            # them next to the declared output's value.
            for decl in lm.produced_artifacts:
                if (a := produced_artifact_annotation(decl)) and decl.output in rendered:
                    rendered[decl.output] = f"{rendered[decl.output]} ({a})"
            return rendered
        case dict():
            return _dict_literal_repr(lm)
        case _:
            raise ValueError(f"Unknown literal type {lm}, type{type(lm)}")
