import base64
import datetime
import json

import msgpack
import pytest
from flyteidl.core import literals_pb2
from google.protobuf.duration_pb2 import Duration
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp

from flyte._protos.workflow import run_definition_pb2
from flyte.types._string_literals import literal_string_repr


class TestLiteralStringReprWithLiteral:
    def test_primitive_integer(self):
        lit = literals_pb2.Literal()
        lit.scalar.primitive.integer = 42
        assert literal_string_repr(lit) == 42

    def test_primitive_float(self):
        lit = literals_pb2.Literal()
        lit.scalar.primitive.float_value = 42.5
        assert literal_string_repr(lit) == 42.5

    def test_primitive_boolean(self):
        lit = literals_pb2.Literal()
        lit.scalar.primitive.boolean = True
        assert literal_string_repr(lit) is True

    def test_primitive_string(self):
        lit = literals_pb2.Literal()
        lit.scalar.primitive.string_value = "hello"
        assert literal_string_repr(lit) == "hello"

    def test_primitive_datetime(self):
        lit = literals_pb2.Literal()
        dt = Timestamp()
        dt.FromDatetime(datetime.datetime(2023, 1, 1, 12, 0, 0))
        lit.scalar.primitive.datetime.CopyFrom(dt)
        assert literal_string_repr(lit) == dt.ToDatetime().isoformat()

    def test_primitive_duration(self):
        lit = literals_pb2.Literal()
        duration = Duration()
        duration.FromSeconds(300)
        lit.scalar.primitive.duration.CopyFrom(duration)
        assert literal_string_repr(lit) == 300.0

    def test_scalar_none_type(self):
        lit = literals_pb2.Literal()
        lit.scalar.none_type.SetInParent()
        assert literal_string_repr(lit) is None

    def test_scalar_error(self):
        lit = literals_pb2.Literal()
        lit.scalar.error.message = "error message"
        assert literal_string_repr(lit) == "error message"

    def test_scalar_structured_dataset(self):
        lit = literals_pb2.Literal()
        lit.scalar.structured_dataset.uri = "s3://bucket/key"
        assert literal_string_repr(lit) == "s3://bucket/key"

    def test_scalar_schema(self):
        lit = literals_pb2.Literal()
        lit.scalar.schema.uri = "s3://bucket/schema"
        assert literal_string_repr(lit) == "s3://bucket/schema"

    def test_scalar_blob(self):
        lit = literals_pb2.Literal()
        lit.scalar.blob.uri = "s3://bucket/blob"
        assert literal_string_repr(lit) == "s3://bucket/blob"

    def test_scalar_binary_msgpack(self):
        lit = literals_pb2.Literal()
        test_data = {"key": "value"}
        packed = msgpack.packb(test_data)
        lit.scalar.binary.value = packed
        lit.scalar.binary.tag = "msgpack"
        assert literal_string_repr(lit) == json.dumps(test_data)

    def test_scalar_binary_other(self):
        lit = literals_pb2.Literal()
        test_data = b"hello world"
        lit.scalar.binary.value = test_data
        lit.scalar.binary.tag = "other"
        assert literal_string_repr(lit) == base64.b64encode(test_data)

    def test_scalar_generic(self):
        lit = literals_pb2.Literal()
        struct = Struct()
        struct.update({"key": "value"})
        lit.scalar.generic.CopyFrom(struct)
        assert literal_string_repr(lit) == MessageToDict(struct)

    def test_scalar_union(self):
        lit = literals_pb2.Literal()
        inner_lit = literals_pb2.Literal()
        inner_lit.scalar.primitive.integer = 42
        lit.scalar.union.value.CopyFrom(inner_lit)
        assert literal_string_repr(lit) == 42

    def test_collection(self):
        lit = literals_pb2.Literal()
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 42
        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.string_value = "hello"
        lit.collection.literals.extend([lit1, lit2])
        assert literal_string_repr(lit) == [42, "hello"]

    def test_map(self):
        lit1 = literals_pb2.Literal(
            scalar=literals_pb2.Scalar(
                primitive=literals_pb2.Primitive(
                    integer=42,
                ),
            )
        )
        lit2 = literals_pb2.Literal(
            scalar=literals_pb2.Scalar(
                primitive=literals_pb2.Primitive(
                    string_value="hello",
                ),
            ),
        )
        lit = literals_pb2.Literal(
            map=literals_pb2.LiteralMap(
                literals={
                    "key1": lit1,
                    "key2": lit2,
                }
            ),
        )
        assert literal_string_repr(lit) == {"key1": 42, "key2": "hello"}

    def test_offloaded_metadata(self):
        lit = literals_pb2.Literal(
            offloaded_metadata=literals_pb2.LiteralOffloadedMetadata(
                uri="s3://bucket/key",
            ),
        )
        assert "Offloaded literal metadata:" in literal_string_repr(lit)


class TestLiteralStringReprWithLiteralMap:
    def test_literal_map(self):
        lit1 = literals_pb2.Literal(
            scalar=literals_pb2.Scalar(
                primitive=literals_pb2.Primitive(
                    integer=42,
                ),
            )
        )
        lit2 = literals_pb2.Literal(
            scalar=literals_pb2.Scalar(
                primitive=literals_pb2.Primitive(
                    string_value="hello",
                ),
            ),
        )
        lit = literals_pb2.LiteralMap(
            literals={
                "key1": lit1,
                "key2": lit2,
            }
        )
        assert literal_string_repr(lit) == {"key1": 42, "key2": "hello"}

    def test_empty_literal_map(self):
        lm = literals_pb2.LiteralMap()
        assert literal_string_repr(lm) == {}


class TestLiteralStringReprWithNamedLiteral:
    def test_named_literal(self):
        nl = run_definition_pb2.NamedLiteral()
        nl.name = "test_param"
        lit = literals_pb2.Literal()
        lit.scalar.primitive.integer = 42
        nl.value.CopyFrom(lit)
        assert literal_string_repr(nl) == {"test_param": 42}


class TestLiteralStringReprWithInputs:
    def test_inputs(self):
        inputs = run_definition_pb2.Inputs()
        nl1 = run_definition_pb2.NamedLiteral()
        nl1.name = "param1"
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 42
        nl1.value.CopyFrom(lit1)

        nl2 = run_definition_pb2.NamedLiteral()
        nl2.name = "param2"
        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.string_value = "hello"
        nl2.value.CopyFrom(lit2)

        inputs.literals.extend([nl1, nl2])
        assert literal_string_repr(inputs) == {"param1": 42, "param2": "hello"}

    def test_empty_inputs(self):
        inputs = run_definition_pb2.Inputs()
        assert literal_string_repr(inputs) == {}


class TestLiteralStringReprWithOutputs:
    def test_outputs(self):
        outputs = run_definition_pb2.Outputs()
        nl1 = run_definition_pb2.NamedLiteral()
        nl1.name = "result1"
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 42
        nl1.value.CopyFrom(lit1)

        nl2 = run_definition_pb2.NamedLiteral()
        nl2.name = "result2"
        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.string_value = "hello"
        nl2.value.CopyFrom(lit2)

        outputs.literals.extend([nl1, nl2])
        assert literal_string_repr(outputs) == {"result1": 42, "result2": "hello"}

    def test_empty_outputs(self):
        outputs = run_definition_pb2.Outputs()
        assert literal_string_repr(outputs) == {}


class TestLiteralStringReprWithDict:
    def test_dict_of_literals(self):
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 42
        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.string_value = "hello"
        lm = {"key1": lit1, "key2": lit2}
        assert literal_string_repr(lm) == {"key1": 42, "key2": "hello"}

    def test_empty_dict(self):
        lm = {}
        assert literal_string_repr(lm) == {}


class TestLiteralStringReprWithNoneOrInvalid:
    def test_none_input(self):
        assert literal_string_repr(None) == {}

    def test_invalid_type(self):
        invalid_obj = "not a valid type"
        with pytest.raises(ValueError, match="Unknown literal type"):
            literal_string_repr(invalid_obj)
