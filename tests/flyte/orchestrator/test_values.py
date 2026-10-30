"""The orchestrator worker's value rules, as `flyte.orchestrator._values` ports them.

Every expectation here is the remote worker's (unionai/cloud
`inline-leaseworker`: `orchestrator-core/src/value.rs`,
`flyte-children/src/literal.rs` and `launcher.rs`, `inline-leaseworker/src/handler.rs`),
values and messages exactly, so that an orchestrator run on this machine behaves as
it does on a cluster. A test that fails here means local runs no longer match
remote ones: change the worker too, or put the behavior back.
"""

import collections

import pytest
from flyteidl2.core import interface_pb2, literals_pb2, types_pb2

from flyte.io import Dir, File
from flyte.orchestrator import _values as v
from flyte.orchestrator._values import Blob, CallError, ConversionError, UnsupportedType
from flyte.sandbox._bridge import _BLOB_TUPLES, Opaque

FileTuple = _BLOB_TUPLES[File]
DirTuple = _BLOB_TUPLES[Dir]


def simple(kind: int) -> types_pb2.LiteralType:
    return types_pb2.LiteralType(simple=kind)


INT = simple(types_pb2.INTEGER)
FLOAT = simple(types_pb2.FLOAT)
STR = simple(types_pb2.STRING)
BOOL = simple(types_pb2.BOOLEAN)
NONE = simple(types_pb2.NONE)
STRUCT = simple(types_pb2.STRUCT)
DATETIME = simple(types_pb2.DATETIME)
FILE = types_pb2.LiteralType(blob=types_pb2.BlobType(dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE))
DIR = types_pb2.LiteralType(blob=types_pb2.BlobType(dimensionality=types_pb2.BlobType.BlobDimensionality.MULTIPART))
DATAFRAME = types_pb2.LiteralType(structured_dataset_type=types_pb2.StructuredDatasetType(format="parquet"))


def list_of(inner):
    return types_pb2.LiteralType(collection_type=inner)


def dict_of(inner):
    return types_pb2.LiteralType(map_value_type=inner)


def union_of(*variants):
    return types_pb2.LiteralType(union_type=types_pb2.UnionType(variants=list(variants)))


def int_literal(n):
    return literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=n)))


def str_literal(s):
    return literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(string_value=s)))


def dataframe_literal():
    return literals_pb2.Literal(
        scalar=literals_pb2.Scalar(
            structured_dataset=literals_pb2.StructuredDataset(
                uri="s3://bucket/df.parquet",
                metadata=literals_pb2.StructuredDatasetMetadata(
                    structured_dataset_type=DATAFRAME.structured_dataset_type
                ),
            )
        ),
        hash="df-hash",
    )


def model_literal():
    return literals_pb2.Literal(
        scalar=literals_pb2.Scalar(binary=literals_pb2.Binary(value=b"\x81\xa1x\x01", tag="msgpack"))
    )


def entry(name, literal_type):
    return interface_pb2.VariableEntry(key=name, value=interface_pb2.Variable(type=literal_type))


def opaque(literal):
    return Opaque(literal.SerializeToString(deterministic=True))


class TestFromMonty:
    """What the source can pass to a task or return, and what it becomes."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, None),
            (True, True),
            (False, False),
            (0, 0),
            (2**63 - 1, 2**63 - 1),
            (-(2**63), -(2**63)),
            (1.5, 1.5),
            ("text", "text"),
            ([1, "a", None], [1, "a", None]),
            ((1, 2), [1, 2]),
            ([(1, (2,))], [[1, [2]]]),
            ({"b": 1, "a": 2}, {"b": 1, "a": 2}),
            ({1: "x"}, {1: "x"}),
        ],
    )
    def test_plain_data(self, value, expected):
        result = v.from_monty(value)
        assert result == expected
        assert type(result) is type(expected)

    def test_dicts_keep_the_sources_order(self):
        assert list(v.from_monty({"b": 1, "a": 2})) == ["b", "a"]

    def test_tuple_keys_are_kept_to_be_refused_when_converted(self):
        assert v.from_monty({(1, 2): "x"}) == {(1, 2): "x"}

    @pytest.mark.parametrize(
        "value, type_name",
        [
            (2**63, "int"),
            (-(2**63) - 1, "int"),
            (b"bytes", "bytes"),
            (bytearray(b"x"), "bytearray"),
            ({1, 2}, "set"),
            (frozenset({1}), "frozenset"),
            ([1, {2}], "set"),
            ({"k": b"v"}, "bytes"),
            (collections.namedtuple("Point", "x y")(1, 2), "namedtuple"),
        ],
    )
    def test_anything_else_is_refused(self, value, type_name):
        with pytest.raises(UnsupportedType) as raised:
            v.from_monty(value)
        assert raised.value.type_name == type_name
        assert str(raised.value) == f"values of type '{type_name}' cannot be passed to or returned from a task"

    def test_values_may_nest_32_deep(self):
        value = 1
        for _ in range(v.MAX_DEPTH):
            value = [value]
        assert v.from_monty(value) == value
        with pytest.raises(UnsupportedType) as raised:
            v.from_monty([value])
        assert str(raised.value) == (
            "a value passed to or returned from a task is nested too deeply, or is not plain data "
            "(None, bool, int, float, str, files and directories, and lists, tuples and str-keyed "
            "dicts of them)"
        )

    def test_files_and_directories(self):
        assert v.from_monty(FileTuple("s3://b/k/f.csv", "ignored", "csv", "h")) == Blob(
            "File", "s3://b/k/f.csv", "csv", "h"
        )
        assert v.from_monty(DirTuple("s3://b/d", "d", "", None)) == Blob("Dir", "s3://b/d", "", None)

    @pytest.mark.parametrize(
        "value",
        [
            FileTuple(1, "n", "", None),
            FileTuple("p", "n", None, None),
            FileTuple("p", "n", "", 7),
            collections.namedtuple("File", "path name")("p", "n"),
        ],
    )
    def test_malformed_files_are_refused(self, value):
        with pytest.raises(UnsupportedType, match="'namedtuple'"):
            v.from_monty(value)

    def test_opaque_values(self):
        assert v.from_monty(Opaque(b"\x01\x02")) == Opaque(b"\x01\x02")
        with pytest.raises(UnsupportedType, match="'namedtuple'"):
            v.from_monty(Opaque("not bytes"))  # type: ignore[arg-type]


class TestToMonty:
    """What the source sees of a value."""

    @pytest.mark.parametrize(
        "path, name",
        [
            ("s3://bucket/key/file.txt", "file.txt"),
            ("s3://bucket/dir/", "dir"),
            ("s3://bucket/dir//", "dir"),
            ("s3://bucket", "bucket"),
            ("file.txt", "file.txt"),
            ("", ""),
        ],
    )
    def test_a_files_name_is_the_last_part_of_its_path(self, path, name):
        assert Blob("File", path, "", None).name == name
        assert v.to_monty(Blob("File", path, "", None)).name == name

    def test_files_and_directories_are_named_tuples(self):
        f = v.to_monty(Blob("File", "s3://b/f.csv", "csv", "h"))
        assert f == ("s3://b/f.csv", "f.csv", "csv", "h")
        assert type(f) is FileTuple
        assert f._fields == ("path", "name", "format", "hash")
        d = v.to_monty(Blob("Dir", "s3://b/d/", "", None))
        assert type(d) is DirTuple
        assert d.hash is None

    def test_nested(self):
        assert v.to_monty({"f": [Blob("File", "p/x", "", None)], "n": 1}) == {
            "f": [FileTuple("p/x", "x", "", None)],
            "n": 1,
        }

    def test_opaque_values_are_unchanged(self):
        assert v.to_monty(Opaque(b"x")) == Opaque(b"x")
        assert repr(Opaque(b"x")) == "Opaque(literal=b'x')"


class TestDescribe:
    @pytest.mark.parametrize(
        "literal_type, described",
        [
            (INT, "integer"),
            (FLOAT, "float"),
            (STR, "string"),
            (BOOL, "boolean"),
            (NONE, "none"),
            (STRUCT, "struct"),
            (DATETIME, "datetime"),
            (list_of(INT), "list[integer]"),
            (dict_of(list_of(STR)), "dict[str, list[string]]"),
            (union_of(INT, NONE), "integer | none"),
            (FILE, "File"),
            (DIR, "Dir"),
            (types_pb2.LiteralType(blob=types_pb2.BlobType(dimensionality=7)), "a file or directory"),
            (DATAFRAME, "a dataframe"),
            (types_pb2.LiteralType(schema=types_pb2.SchemaType()), "a schema"),
            (types_pb2.LiteralType(enum_type=types_pb2.EnumType(values=["a"])), "an enum"),
            (types_pb2.LiteralType(), "an unknown type"),
        ],
    )
    def test_types_are_described_as_the_worker_describes_them(self, literal_type, described):
        assert v.describe(literal_type) == described


class TestToLiteral:
    """How a value the source passes becomes the literal a task receives."""

    @pytest.mark.parametrize(
        "value, literal_type, field, expected",
        [
            (3, INT, "integer", 3),
            (1.5, FLOAT, "float_value", 1.5),
            (3, FLOAT, "float_value", 3.0),
            ("s", STR, "string_value", "s"),
            (True, BOOL, "boolean", True),
        ],
    )
    def test_primitives(self, value, literal_type, field, expected):
        literal = v.to_literal(value, literal_type)
        assert getattr(literal.scalar.primitive, field) == expected
        assert type(getattr(literal.scalar.primitive, field)) is type(expected)

    def test_none(self):
        assert v.to_literal(None, NONE).scalar.HasField("none_type")

    @pytest.mark.parametrize(
        "value, literal_type, message",
        [
            (True, INT, "a bool cannot be passed as integer"),
            (1, BOOL, "a int cannot be passed as boolean"),
            (1.5, INT, "a float cannot be passed as integer"),
            ("1", INT, "a str cannot be passed as integer"),
            (None, STR, "a None cannot be passed as string"),
            (1, NONE, "a int cannot be passed as none"),
            ([1], INT, "a list cannot be passed as integer"),
            ({"a": 1}, list_of(INT), "a dict cannot be passed as list[integer]"),
            ([1], dict_of(INT), "a list cannot be passed as dict[str, integer]"),
            ({1: 1}, dict_of(INT), "a dict cannot be passed as dict[str, integer]"),
            ({(1,): 1}, dict_of(INT), "a dict cannot be passed as dict[str, integer]"),
            (["a"], list_of(INT), "a str cannot be passed as integer"),
            ({"k": "a"}, dict_of(INT), "a str cannot be passed as integer"),
            ("s3://f", FILE, "a str cannot be passed as File"),
            (Blob("File", "p", "", None), DIR, "a File cannot be passed as Dir"),
            (Blob("Dir", "p", "", None), FILE, "a Dir cannot be passed as File"),
            ("x", union_of(INT, NONE), "a str cannot be passed as integer | none"),
            (1, DATETIME, "values of Flyte type datetime are not supported"),
            (1, STRUCT, "values of Flyte type struct are not supported"),
            (1, DATAFRAME, "values of Flyte type a dataframe are not supported"),
            (1, types_pb2.LiteralType(), "values of Flyte type an unknown type are not supported"),
            (
                1,
                types_pb2.LiteralType(enum_type=types_pb2.EnumType(values=["a"])),
                "values of Flyte type an enum are not supported",
            ),
        ],
    )
    def test_values_that_do_not_fit(self, value, literal_type, message):
        with pytest.raises(ConversionError) as raised:
            v.to_literal(value, literal_type)
        assert str(raised.value) == message

    def test_lists_and_dicts(self):
        literal = v.to_literal([1, 2], list_of(INT))
        assert [item.scalar.primitive.integer for item in literal.collection.literals] == [1, 2]
        literal = v.to_literal({"b": 2, "a": 1}, dict_of(FLOAT))
        assert set(literal.map.literals) == {"a", "b"}
        assert literal.map.literals["a"].scalar.primitive.float_value == 1.0

    def test_files_keep_their_format_and_hash(self):
        literal = v.to_literal(Blob("File", "s3://b/f.csv", "csv", "abc"), FILE)
        assert literal.scalar.blob.uri == "s3://b/f.csv"
        assert literal.scalar.blob.metadata.type.format == "csv"
        assert literal.scalar.blob.metadata.type.dimensionality == types_pb2.BlobType.BlobDimensionality.SINGLE
        assert literal.hash == "abc"
        assert v.to_literal(Blob("File", "p", "", None), FILE).hash == ""
        assert (
            v.to_literal(Blob("Dir", "p", "", None), DIR).scalar.blob.metadata.type.dimensionality
            == types_pb2.BlobType.BlobDimensionality.MULTIPART
        )

    def test_a_union_takes_the_variant_the_value_is_exactly(self):
        number = union_of(FLOAT, INT)
        assert v.to_literal(3, number).scalar.union.type == INT
        assert v.to_literal(3, number).scalar.union.value.scalar.primitive.integer == 3
        assert v.to_literal(1.5, number).scalar.union.type == FLOAT
        optional = union_of(INT, NONE)
        assert v.to_literal(None, optional).scalar.union.type == NONE

    def test_a_union_takes_the_only_variant_the_value_converts_to(self):
        literal = v.to_literal(3, union_of(FLOAT, STR))
        assert literal.scalar.union.type == FLOAT
        assert literal.scalar.union.value.scalar.primitive.float_value == 3.0

    def test_a_value_that_converts_to_several_variants_is_ambiguous(self):
        with pytest.raises(ConversionError) as raised:
            v.to_literal([1], union_of(list_of(INT), list_of(FLOAT)))
        assert str(raised.value) == "a list fits more than one variant of the union: list[integer] and list[float]"

    def test_opaque_values_pass_whole_to_their_kind(self):
        df, model = dataframe_literal(), model_literal()
        assert v.to_literal(opaque(df), DATAFRAME) == df
        assert v.to_literal(opaque(model), STRUCT) == model
        assert v.to_literal(opaque(df), union_of(DATAFRAME, NONE)).scalar.union.value == df
        for value, literal_type, message in [
            (opaque(df), STRUCT, "a Opaque cannot be passed as struct"),
            (opaque(model), DATAFRAME, "a Opaque cannot be passed as a dataframe"),
            (opaque(model), INT, "a Opaque cannot be passed as integer"),
            (opaque(int_literal(1)), INT, "a Opaque cannot be passed as integer"),
            (Opaque(b"\xff\xff"), STRUCT, "a Opaque cannot be passed as struct"),
            (opaque(model), union_of(INT, NONE), "a Opaque cannot be passed as integer | none"),
        ]:
            with pytest.raises(ConversionError) as raised:
                v.to_literal(value, literal_type)
            assert str(raised.value) == message


class TestFromLiteral:
    """What a task's result, or an input, is in the source."""

    @pytest.mark.parametrize(
        "literal, expected",
        [
            (int_literal(3), 3),
            (str_literal("s"), "s"),
            (literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(float_value=2.0))), 2.0),
            (literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(boolean=False))), False),
            (literals_pb2.Literal(scalar=literals_pb2.Scalar(none_type=literals_pb2.Void())), None),
            (literals_pb2.Literal(), None),
        ],
    )
    def test_primitives(self, literal, expected):
        result = v.from_literal(literal)
        assert result == expected
        assert type(result) is type(expected)

    def test_dicts_are_sorted_by_key(self):
        literal = literals_pb2.Literal(
            map=literals_pb2.LiteralMap(literals={"b": int_literal(2), "c": int_literal(3), "a": int_literal(1)})
        )
        assert list(v.from_literal(literal).items()) == [("a", 1), ("b", 2), ("c", 3)]

    def test_lists_and_unions(self):
        literal = literals_pb2.Literal(
            collection=literals_pb2.LiteralCollection(literals=[int_literal(1), str_literal("a")])
        )
        assert v.from_literal(literal) == [1, "a"]
        union = literals_pb2.Literal(
            scalar=literals_pb2.Scalar(union=literals_pb2.Union(value=int_literal(4), type=INT))
        )
        assert v.from_literal(union) == 4

    def test_files_and_directories(self):
        literal = v.to_literal(Blob("File", "s3://b/f.csv", "csv", "abc"), FILE)
        assert v.from_literal(literal) == Blob("File", "s3://b/f.csv", "csv", "abc")
        literal.hash = ""
        assert v.from_literal(literal).hash is None
        assert v.from_literal(v.to_literal(Blob("Dir", "s3://b/d", "", None), DIR)) == Blob("Dir", "s3://b/d", "", None)

    def test_dataframes_and_structs_are_opaque(self):
        for literal in [
            dataframe_literal(),
            model_literal(),
            literals_pb2.Literal(scalar=literals_pb2.Scalar(generic={"a": 1})),
        ]:
            value = v.from_literal(literal)
            assert type(value) is Opaque
            assert value.literal == literal.SerializeToString(deterministic=True)
            parsed = literals_pb2.Literal()
            parsed.ParseFromString(value.literal)
            assert parsed == literal

    def test_equal_literals_are_equal_opaque_values(self):
        assert v.from_literal(model_literal()) == v.from_literal(model_literal())
        assert v.from_literal(dataframe_literal()) != v.from_literal(model_literal())

    @pytest.mark.parametrize(
        "literal, kind",
        [
            (
                literals_pb2.Literal(
                    scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(datetime={"seconds": 1}))
                ),
                "datetime",
            ),
            (
                literals_pb2.Literal(
                    scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(duration={"seconds": 1}))
                ),
                "duration",
            ),
            (literals_pb2.Literal(scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive())), "empty primitive"),
            (literals_pb2.Literal(scalar=literals_pb2.Scalar(union=literals_pb2.Union(type=INT))), "empty union"),
            (
                literals_pb2.Literal(scalar=literals_pb2.Scalar(blob=literals_pb2.Blob(uri="p"))),
                "blob of unknown dimensionality",
            ),
            (literals_pb2.Literal(scalar=literals_pb2.Scalar(schema=literals_pb2.Schema(uri="p"))), "schema"),
            (literals_pb2.Literal(scalar=literals_pb2.Scalar(error=types_pb2.Error(message="m"))), "error"),
            (literals_pb2.Literal(scalar=literals_pb2.Scalar()), "empty scalar"),
            (literals_pb2.Literal(offloaded_metadata=literals_pb2.LiteralOffloadedMetadata(uri="p")), "offloaded"),
        ],
    )
    def test_literals_that_cannot_be_read(self, literal, kind):
        with pytest.raises(ConversionError) as raised:
            v.from_literal(literal)
        assert str(raised.value) == f"a literal of kind {kind} cannot be read"

    def test_round_trips(self):
        for value, literal_type in [
            (3, INT),
            ([{"a": [1.5]}], list_of(dict_of(list_of(FLOAT)))),
            (Blob("File", "s3://b/f", "csv", None), union_of(FILE, NONE)),
            (None, union_of(FILE, NONE)),
            (opaque(model_literal()), union_of(STRUCT, NONE)),
        ]:
            assert v.from_literal(v.to_literal(value, literal_type)) == value


class TestFits:
    def test_fits(self):
        assert v.fits(1, INT)
        assert v.fits(1, FLOAT)
        assert not v.fits("1", INT)
        assert not v.fits(1, DATETIME)
        # Ambiguity is the caller's to resolve, so a value that fits several variants fits.
        assert v.fits([1], union_of(list_of(INT), list_of(FLOAT)))
        assert v.fits([[1]], list_of(union_of(list_of(INT), list_of(FLOAT))))


# The worker's own tests use `add(x: int, y: int)`, with `y` defaulting to 10.
ADD = [entry("x", INT), entry("y", INT)]
DEFAULTS = {"y": int_literal(10)}


def bind(args, kwargs, parameters=("x", "y"), declared=ADD, defaults=DEFAULTS):
    return v.bind_inputs("add", declared, parameters, defaults, list(args), list(kwargs))


class TestBindInputs:
    """How a call's arguments bind to a task's inputs."""

    def test_arguments_bind_the_same_however_they_are_passed(self):
        bound = bind([1, 2], [])
        assert bound == bind([], [("y", 2), ("x", 1)])
        assert bound == bind([1], [("y", 2)])
        assert [(name, literal.scalar.primitive.integer) for name, literal in bound] == [("x", 1), ("y", 2)]

    def test_positional_arguments_bind_in_declared_order(self):
        bound = bind([1, 2], [], parameters=("y", "x"))
        assert [name for name, _ in bound] == ["y", "x"]
        assert bound == bind([], [("x", 2), ("y", 1)], parameters=("y", "x"))
        by_spec = bind([], [("y", 1), ("x", 2)], parameters=None)
        assert [name for name, _ in by_spec] == ["x", "y"]

    def test_without_declared_order_arguments_go_by_keyword(self):
        with pytest.raises(CallError) as raised:
            bind([1, 2], [], parameters=None)
        assert str(raised.value) == (
            "add() takes its arguments by keyword, as the order of its inputs is not known: "
            "call it as add(x=..., y=...)"
        )
        assert bind([], [("x", 1), ("y", 2)], parameters=None) == bind([1, 2], [])
        one = [entry("x", INT)]
        assert bind([1], [], parameters=None, declared=one, defaults={}) == [("x", int_literal(1))]

    def test_missing_arguments_take_the_task_defaults(self):
        defaulted = bind([1], [])
        assert defaulted == bind([1, 10], [])
        assert defaulted[1] == ("y", int_literal(10))

    @pytest.mark.parametrize(
        "args, kwargs, message",
        [
            ([1, 2, 3], [], "add() takes 2 arguments but 3 were given"),
            ([1], [("x", 2)], "add() got multiple values for argument 'x'"),
            ([], [("x", 1), ("z", 2)], "add() got an unexpected keyword argument 'z'"),
            ([], [("y", 2)], "add() is missing the argument 'x'"),
            (["1"], [], "add() argument 'x': a str cannot be passed as integer"),
            ([1, 1.5], [], "add() argument 'y': a float cannot be passed as integer"),
        ],
    )
    def test_calls_that_do_not_bind(self, args, kwargs, message):
        with pytest.raises(CallError) as raised:
            bind(args, kwargs)
        assert str(raised.value) == message

    def test_more_positional_arguments_than_the_declared_order_names(self):
        with pytest.raises(CallError) as raised:
            bind([1, 2], [], parameters=("x",))
        assert str(raised.value) == "add() takes 1 positional arguments but 2 were given"

    def test_an_input_without_a_type(self):
        with pytest.raises(CallError) as raised:
            bind([1], [], declared=[interface_pb2.VariableEntry(key="x")], parameters=("x",), defaults={})
        assert str(raised.value) == "add() declares no type for 'x'"


O0 = [entry("o0", INT)]


class TestReadOutputs:
    """What a task call returns to the source."""

    def test_one_output_is_its_value_several_are_a_list(self):
        assert v.read_outputs("t", O0, [("o0", int_literal(3))]) == 3
        two = [entry("o0", INT), entry("o1", INT)]
        assert v.read_outputs("t", two, [("o1", int_literal(2)), ("o0", int_literal(1))]) == [1, 2]

    @pytest.mark.parametrize(
        "outputs, message",
        [
            ([], "the result of t: the task did not return the output 'o0'"),
            ([("o0", None)], "the result of t: the task did not return the output 'o0'"),
            (
                [("o0", int_literal(1)), ("o1", int_literal(2))],
                "the result of t: the task returned an undeclared output 'o1'",
            ),
            ([("o0", str_literal("x"))], "the result of t: output 'o0' is not of the declared type integer"),
            (
                [("o0", literals_pb2.Literal(scalar=literals_pb2.Scalar(schema=literals_pb2.Schema())))],
                "the result of t: output 'o0': a literal of kind schema cannot be read",
            ),
        ],
    )
    def test_outputs_that_do_not_match_the_declaration(self, outputs, message):
        with pytest.raises(CallError) as raised:
            v.read_outputs("t", O0, outputs)
        assert str(raised.value) == message


class TestEncodeOutputs:
    """How the orchestrator's result becomes its outputs."""

    def test_one_output_takes_the_whole_result(self):
        assert v.encode_outputs([entry("o0", list_of(INT))], [1, 2]) == [("o0", v.to_literal([1, 2], list_of(INT)))]

    def test_several_outputs_take_one_element_each(self):
        declared = [entry("o0", INT), entry("o1", STR)]
        assert v.encode_outputs(declared, [1, "a"]) == [("o0", int_literal(1)), ("o1", str_literal("a"))]

    @pytest.mark.parametrize(
        "declared, value, code, message",
        [
            ([entry("o0", INT), entry("o1", STR)], [1], "TypeError", "the orchestrator declares 2 outputs"),
            ([entry("o0", INT), entry("o1", STR)], 1, "TypeError", "the orchestrator declares 2 outputs"),
            ([entry("o0", INT)], "x", "TypeError", "the result: a str cannot be passed as integer"),
            ([interface_pb2.VariableEntry(key="o0")], 1, "BadTaskSpecification", "an output has no type"),
        ],
    )
    def test_results_that_do_not_match_the_declaration(self, declared, value, code, message):
        with pytest.raises(CallError) as raised:
            v.encode_outputs(declared, value)
        assert (raised.value.code, str(raised.value)) == (code, message)
