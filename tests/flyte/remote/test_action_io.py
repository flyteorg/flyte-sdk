import base64
import datetime

import msgpack
from flyteidl2.core import literals_pb2
from flyteidl2.task import common_pb2
from google.protobuf.duration_pb2 import Duration
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp

from flyte.remote._action import ActionOutputs


class TestActionOutputsTupleBehavior:
    """Test that ActionOutputs behaves correctly as a tuple."""

    def test_single_output(self):
        """Test ActionOutputs with a single output value."""
        lit = literals_pb2.Literal()
        lit.scalar.primitive.integer = 42

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(42,))

        # Test tuple behavior
        assert len(outputs) == 1
        assert outputs[0] == 42
        assert tuple(outputs) == (42,)

    def test_multiple_outputs(self):
        """Test ActionOutputs with multiple output values."""
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 42

        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.string_value = "hello"

        lit3 = literals_pb2.Literal()
        lit3.scalar.primitive.float_value = 3.14

        nl1 = common_pb2.NamedLiteral()
        nl1.name = "o0"
        nl1.value.CopyFrom(lit1)

        nl2 = common_pb2.NamedLiteral()
        nl2.name = "o1"
        nl2.value.CopyFrom(lit2)

        nl3 = common_pb2.NamedLiteral()
        nl3.name = "o2"
        nl3.value.CopyFrom(lit3)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.extend([nl1, nl2, nl3])

        outputs = ActionOutputs(pb2=outputs_pb2, data=(42, "hello", 3.14))

        # Test tuple behavior
        assert len(outputs) == 3
        assert outputs[0] == 42
        assert outputs[1] == "hello"
        assert outputs[2] == 3.14
        assert tuple(outputs) == (42, "hello", 3.14)

    def test_iteration(self):
        """Test that ActionOutputs can be iterated."""
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 1

        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.integer = 2

        lit3 = literals_pb2.Literal()
        lit3.scalar.primitive.integer = 3

        nl1 = common_pb2.NamedLiteral()
        nl1.name = "o0"
        nl1.value.CopyFrom(lit1)

        nl2 = common_pb2.NamedLiteral()
        nl2.name = "o1"
        nl2.value.CopyFrom(lit2)

        nl3 = common_pb2.NamedLiteral()
        nl3.name = "o2"
        nl3.value.CopyFrom(lit3)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.extend([nl1, nl2, nl3])

        outputs = ActionOutputs(pb2=outputs_pb2, data=(1, 2, 3))

        # Test iteration
        result = list(outputs)
        assert result == [1, 2, 3]

    def test_unpacking(self):
        """Test that ActionOutputs can be unpacked like a tuple."""
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.string_value = "first"

        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.string_value = "second"

        nl1 = common_pb2.NamedLiteral()
        nl1.name = "o0"
        nl1.value.CopyFrom(lit1)

        nl2 = common_pb2.NamedLiteral()
        nl2.name = "o1"
        nl2.value.CopyFrom(lit2)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.extend([nl1, nl2])

        outputs = ActionOutputs(pb2=outputs_pb2, data=("first", "second"))

        # Test unpacking
        a, b = outputs
        assert a == "first"
        assert b == "second"

    def test_slicing(self):
        """Test that ActionOutputs supports slicing."""
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 10

        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.integer = 20

        lit3 = literals_pb2.Literal()
        lit3.scalar.primitive.integer = 30

        nl1 = common_pb2.NamedLiteral()
        nl1.name = "o0"
        nl1.value.CopyFrom(lit1)

        nl2 = common_pb2.NamedLiteral()
        nl2.name = "o1"
        nl2.value.CopyFrom(lit2)

        nl3 = common_pb2.NamedLiteral()
        nl3.name = "o2"
        nl3.value.CopyFrom(lit3)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.extend([nl1, nl2, nl3])

        outputs = ActionOutputs(pb2=outputs_pb2, data=(10, 20, 30))

        # Test slicing
        assert outputs[0:2] == (10, 20)
        assert outputs[1:] == (20, 30)
        assert outputs[:2] == (10, 20)

    def test_empty_outputs(self):
        """Test ActionOutputs with no outputs."""
        outputs_pb2 = common_pb2.Outputs()
        outputs = ActionOutputs(pb2=outputs_pb2, data=())

        # Test empty tuple behavior
        assert len(outputs) == 0
        assert tuple(outputs) == ()
        assert list(outputs) == []


class TestActionOutputsNamedOutputs:
    """Test the named_outputs property of ActionOutputs."""

    def test_named_outputs_single(self):
        """Test named_outputs with a single output."""
        lit = literals_pb2.Literal()
        lit.scalar.primitive.integer = 42

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(42,))

        # Test named_outputs
        assert outputs.named_outputs == {"o0": 42}

    def test_named_outputs_multiple(self):
        """Test named_outputs with multiple outputs."""
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 42

        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.string_value = "hello"

        lit3 = literals_pb2.Literal()
        lit3.scalar.primitive.boolean = True

        nl1 = common_pb2.NamedLiteral()
        nl1.name = "o0"
        nl1.value.CopyFrom(lit1)

        nl2 = common_pb2.NamedLiteral()
        nl2.name = "o1"
        nl2.value.CopyFrom(lit2)

        nl3 = common_pb2.NamedLiteral()
        nl3.name = "o2"
        nl3.value.CopyFrom(lit3)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.extend([nl1, nl2, nl3])

        outputs = ActionOutputs(pb2=outputs_pb2, data=(42, "hello", True))

        # Test named_outputs
        assert outputs.named_outputs == {"o0": 42, "o1": "hello", "o2": True}

    def test_named_outputs_empty(self):
        """Test named_outputs with no outputs."""
        outputs_pb2 = common_pb2.Outputs()
        outputs = ActionOutputs(pb2=outputs_pb2, data=())

        # Test named_outputs
        assert outputs.named_outputs == {}

    def test_named_outputs_cached_property(self):
        """Test that named_outputs is a cached property."""
        lit = literals_pb2.Literal()
        lit.scalar.primitive.integer = 42

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(42,))

        # Access named_outputs multiple times
        result1 = outputs.named_outputs
        result2 = outputs.named_outputs

        # Should be the same object (cached)
        assert result1 is result2


class TestActionOutputsToDict:
    """Test the to_dict method of ActionOutputs."""

    def test_to_dict_single_output(self):
        """Test to_dict with a single output."""
        lit = literals_pb2.Literal()
        lit.scalar.primitive.integer = 42

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(42,))

        # Test to_dict
        result = outputs.to_dict()
        expected = MessageToDict(outputs_pb2)
        assert result == expected

    def test_to_dict_multiple_outputs(self):
        """Test to_dict with multiple outputs."""
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 42

        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.string_value = "hello"

        nl1 = common_pb2.NamedLiteral()
        nl1.name = "o0"
        nl1.value.CopyFrom(lit1)

        nl2 = common_pb2.NamedLiteral()
        nl2.name = "o1"
        nl2.value.CopyFrom(lit2)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.extend([nl1, nl2])

        outputs = ActionOutputs(pb2=outputs_pb2, data=(42, "hello"))

        # Test to_dict
        result = outputs.to_dict()
        expected = MessageToDict(outputs_pb2)
        assert result == expected

    def test_to_dict_empty_outputs(self):
        """Test to_dict with no outputs."""
        outputs_pb2 = common_pb2.Outputs()
        outputs = ActionOutputs(pb2=outputs_pb2, data=())

        # Test to_dict
        result = outputs.to_dict()
        expected = MessageToDict(outputs_pb2)
        assert result == expected


class TestActionOutputsWithVariousTypes:
    """Test ActionOutputs with various literal types similar to test_literal_repr.py."""

    def test_primitive_integer(self):
        """Test ActionOutputs with integer output."""
        lit = literals_pb2.Literal()
        lit.scalar.primitive.integer = 12345

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(12345,))

        assert outputs[0] == 12345
        assert outputs.named_outputs == {"o0": 12345}

    def test_primitive_float(self):
        """Test ActionOutputs with float output."""
        lit = literals_pb2.Literal()
        lit.scalar.primitive.float_value = 3.14159

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(3.14159,))

        assert outputs[0] == 3.14159
        assert outputs.named_outputs == {"o0": 3.14159}

    def test_primitive_boolean(self):
        """Test ActionOutputs with boolean output."""
        lit = literals_pb2.Literal()
        lit.scalar.primitive.boolean = False

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(False,))

        assert outputs[0] is False
        assert outputs.named_outputs == {"o0": False}

    def test_primitive_string(self):
        """Test ActionOutputs with string output."""
        lit = literals_pb2.Literal()
        lit.scalar.primitive.string_value = "test string"

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=("test string",))

        assert outputs[0] == "test string"
        assert outputs.named_outputs == {"o0": "test string"}

    def test_primitive_datetime(self):
        """Test ActionOutputs with datetime output."""
        dt = datetime.datetime(2023, 6, 15, 14, 30, 0, tzinfo=datetime.timezone.utc)

        lit = literals_pb2.Literal()
        timestamp = Timestamp()
        timestamp.FromDatetime(dt)
        lit.scalar.primitive.datetime.CopyFrom(timestamp)

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(dt,))

        assert outputs[0] == dt
        assert outputs.named_outputs == {"o0": dt}

    def test_primitive_duration(self):
        """Test ActionOutputs with duration output."""
        duration_seconds = 3600  # 1 hour
        duration_obj = datetime.timedelta(seconds=duration_seconds)

        lit = literals_pb2.Literal()
        duration = Duration()
        duration.FromSeconds(duration_seconds)
        lit.scalar.primitive.duration.CopyFrom(duration)

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(duration_obj,))

        assert outputs[0] == duration_obj
        assert outputs.named_outputs == {"o0": duration_obj}

    def test_scalar_none_type(self):
        """Test ActionOutputs with None output."""
        lit = literals_pb2.Literal()
        lit.scalar.none_type.SetInParent()

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(None,))

        assert outputs[0] is None
        assert outputs.named_outputs == {"o0": None}

    def test_collection_list(self):
        """Test ActionOutputs with list/collection output."""
        test_list = [1, 2, 3, 4, 5]

        lit = literals_pb2.Literal()
        for val in test_list:
            item_lit = literals_pb2.Literal()
            item_lit.scalar.primitive.integer = val
            lit.collection.literals.append(item_lit)

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(test_list,))

        assert outputs[0] == test_list
        assert outputs.named_outputs == {"o0": test_list}

    def test_map_dict(self):
        """Test ActionOutputs with map/dict output."""
        test_dict = {"key1": 42, "key2": "value2", "key3": True}

        lit = literals_pb2.Literal()

        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 42

        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.string_value = "value2"

        lit3 = literals_pb2.Literal()
        lit3.scalar.primitive.boolean = True

        lit.map.literals["key1"].CopyFrom(lit1)
        lit.map.literals["key2"].CopyFrom(lit2)
        lit.map.literals["key3"].CopyFrom(lit3)

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(test_dict,))

        assert outputs[0] == test_dict
        assert outputs.named_outputs == {"o0": test_dict}

    def test_scalar_blob(self):
        """Test ActionOutputs with blob output."""
        blob_uri = "s3://my-bucket/my-blob"

        lit = literals_pb2.Literal()
        lit.scalar.blob.uri = blob_uri

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(blob_uri,))

        assert outputs[0] == blob_uri
        assert outputs.named_outputs == {"o0": blob_uri}

    def test_scalar_structured_dataset(self):
        """Test ActionOutputs with structured dataset output."""
        dataset_uri = "s3://my-bucket/my-dataset"

        lit = literals_pb2.Literal()
        lit.scalar.structured_dataset.uri = dataset_uri

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(dataset_uri,))

        assert outputs[0] == dataset_uri
        assert outputs.named_outputs == {"o0": dataset_uri}

    def test_scalar_schema(self):
        """Test ActionOutputs with schema output."""
        schema_uri = "s3://my-bucket/my-schema"

        lit = literals_pb2.Literal()
        lit.scalar.schema.uri = schema_uri

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(schema_uri,))

        assert outputs[0] == schema_uri
        assert outputs.named_outputs == {"o0": schema_uri}

    def test_scalar_binary_msgpack(self):
        """Test ActionOutputs with binary msgpack output."""
        test_data = {"nested": {"key": "value"}, "number": 123}
        packed = msgpack.packb(test_data)

        lit = literals_pb2.Literal()
        lit.scalar.binary.value = packed
        lit.scalar.binary.tag = "msgpack"

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(test_data,))

        assert outputs[0] == test_data
        assert outputs.named_outputs == {"o0": test_data}

    def test_scalar_binary_other(self):
        """Test ActionOutputs with binary (non-msgpack) output."""
        test_bytes = b"binary data here"
        encoded = base64.b64encode(test_bytes)

        lit = literals_pb2.Literal()
        lit.scalar.binary.value = test_bytes
        lit.scalar.binary.tag = "custom"

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(encoded,))

        assert outputs[0] == encoded
        assert outputs.named_outputs == {"o0": encoded}

    def test_scalar_generic(self):
        """Test ActionOutputs with generic/struct output."""
        test_dict = {"key1": "value1", "key2": 42, "nested": {"inner": True}}

        lit = literals_pb2.Literal()
        struct = Struct()
        struct.update(test_dict)
        lit.scalar.generic.CopyFrom(struct)

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(test_dict,))

        assert outputs[0] == test_dict
        assert outputs.named_outputs == {"o0": test_dict}

    def test_scalar_union(self):
        """Test ActionOutputs with union output."""
        union_value = 99

        lit = literals_pb2.Literal()
        inner_lit = literals_pb2.Literal()
        inner_lit.scalar.primitive.integer = union_value
        lit.scalar.union.value.CopyFrom(inner_lit)

        nl = common_pb2.NamedLiteral()
        nl.name = "o0"
        nl.value.CopyFrom(lit)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.append(nl)

        outputs = ActionOutputs(pb2=outputs_pb2, data=(union_value,))

        assert outputs[0] == union_value
        assert outputs.named_outputs == {"o0": union_value}

    def test_mixed_types_multiple_outputs(self):
        """Test ActionOutputs with multiple outputs of different types."""
        # Integer
        lit1 = literals_pb2.Literal()
        lit1.scalar.primitive.integer = 42

        # String
        lit2 = literals_pb2.Literal()
        lit2.scalar.primitive.string_value = "hello"

        # List
        lit3 = literals_pb2.Literal()
        for val in [1, 2, 3]:
            item_lit = literals_pb2.Literal()
            item_lit.scalar.primitive.integer = val
            lit3.collection.literals.append(item_lit)

        # Dict
        lit4 = literals_pb2.Literal()
        dict_val_lit = literals_pb2.Literal()
        dict_val_lit.scalar.primitive.boolean = True
        lit4.map.literals["flag"].CopyFrom(dict_val_lit)

        # None
        lit5 = literals_pb2.Literal()
        lit5.scalar.none_type.SetInParent()

        nl1 = common_pb2.NamedLiteral()
        nl1.name = "o0"
        nl1.value.CopyFrom(lit1)

        nl2 = common_pb2.NamedLiteral()
        nl2.name = "o1"
        nl2.value.CopyFrom(lit2)

        nl3 = common_pb2.NamedLiteral()
        nl3.name = "o2"
        nl3.value.CopyFrom(lit3)

        nl4 = common_pb2.NamedLiteral()
        nl4.name = "o3"
        nl4.value.CopyFrom(lit4)

        nl5 = common_pb2.NamedLiteral()
        nl5.name = "o4"
        nl5.value.CopyFrom(lit5)

        outputs_pb2 = common_pb2.Outputs()
        outputs_pb2.literals.extend([nl1, nl2, nl3, nl4, nl5])

        test_data = (42, "hello", [1, 2, 3], {"flag": True}, None)
        outputs = ActionOutputs(pb2=outputs_pb2, data=test_data)

        # Test tuple behavior
        assert len(outputs) == 5
        assert outputs[0] == 42
        assert outputs[1] == "hello"
        assert outputs[2] == [1, 2, 3]
        assert outputs[3] == {"flag": True}
        assert outputs[4] is None

        # Test named_outputs
        assert outputs.named_outputs == {
            "o0": 42,
            "o1": "hello",
            "o2": [1, 2, 3],
            "o3": {"flag": True},
            "o4": None,
        }

        # Test unpacking
        int_val, str_val, list_val, dict_val, none_val = outputs
        assert int_val == 42
        assert str_val == "hello"
        assert list_val == [1, 2, 3]
        assert dict_val == {"flag": True}
        assert none_val is None

        # Test to_dict returns valid dict
        result_dict = outputs.to_dict()
        assert isinstance(result_dict, dict)
        assert "literals" in result_dict
