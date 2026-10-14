import enum
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import click
import pytest
from flyteidl2.core.interface_pb2 import Variable
from flyteidl2.core.types_pb2 import LiteralType, SimpleType

from flyte.cli._params import EnumParamType, JsonParamType, to_click_option
from flyte.cli._run import RunArguments
from flyte.io import DataFrame, Dir, File


class Color(str, enum.Enum):
    RED = "red-value"
    GREEN = "green-value"
    BLUE = "blue-value"


class Size(str, enum.Enum):
    SMALL = "sm-value"
    MEDIUM = "md-value"
    LARGE = "lg-value"


def test_enum_param_type_choices_are_names():
    """EnumParamType should expose enum names as CLI choices, not values."""
    param_type = EnumParamType(Color)
    assert list(param_type.choices) == ["RED", "GREEN", "BLUE"]


def test_enum_param_type_convert_name():
    """Passing an enum name (e.g. GREEN) should return the corresponding enum instance."""
    param_type = EnumParamType(Color)
    assert param_type.convert("GREEN", param=None, ctx=None) is Color.GREEN


def test_enum_param_type_convert_str_enum():
    """EnumParamType should work with StrEnum subclasses."""
    param_type = EnumParamType(Size)
    assert param_type.convert("SMALL", param=None, ctx=None) is Size.SMALL


def test_enum_param_type_rejects_value():
    """Passing an enum value (e.g. 'red-value') should be rejected — only names are valid."""
    param_type = EnumParamType(Color)
    with pytest.raises((click.exceptions.BadParameter, SystemExit)):
        param_type.convert("red-value", param=None, ctx=None)


def test_enum_param_type_passthrough_instance():
    """Passing an already-converted enum instance should be returned as-is."""
    param_type = EnumParamType(Color)
    assert param_type.convert(Color.BLUE, param=None, ctx=None) is Color.BLUE


def test_boolean_flag_false_default():
    """Test boolean parameter with False default creates a simple flag."""
    literal_var = Variable(type=LiteralType(simple=SimpleType.BOOLEAN), description="A boolean flag")

    option = to_click_option(input_name="flag", literal_var=literal_var, python_type=bool, default_val=False)

    assert option.opts == ["--flag"]
    assert not option.secondary_opts
    assert option.is_flag is True
    assert option.default is False
    assert option.required is False


def test_boolean_flag_true_default():
    """Test boolean parameter with True default creates a --flag/--no-flag pattern."""
    literal_var = Variable(type=LiteralType(simple=SimpleType.BOOLEAN), description="A boolean flag with True default")

    option = to_click_option(input_name="enabled", literal_var=literal_var, python_type=bool, default_val=True)

    assert option.opts == ["--enabled"]
    assert option.secondary_opts == ["--no-enabled"]
    assert option.is_flag is True
    assert option.default is True
    assert option.required is False


def test_boolean_flag_no_default():
    """Test boolean parameter with no default creates a simple flag with False default."""
    literal_var = Variable(type=LiteralType(simple=SimpleType.BOOLEAN), description="A boolean flag with no default")

    option = to_click_option(input_name="debug", literal_var=literal_var, python_type=bool, default_val=None)

    assert option.opts == ["--debug"]
    assert option.is_flag is True
    assert option.default is False  # Should default to False for boolean flags
    assert option.required is False


# ---------------------------------------------------------------------------
# JsonParamType: File, Dir, and DataFrame CLI JSON parsing
# ---------------------------------------------------------------------------


@pytest.fixture
def local_mode_ctx():
    run_args = RunArguments(local=True)
    mock_cli_obj = MagicMock()
    mock_cli_obj.run_args = run_args
    ctx = MagicMock()
    ctx.obj = mock_cli_obj
    return ctx


@pytest.fixture
def temp_txt_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("test content")
        path = f.name
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as path:
        yield path


pd = None
try:
    import pandas as pd
except ImportError:
    pass


@pytest.fixture
def temp_parquet_file():
    if pd is None:
        pytest.skip("pandas is not installed")
    df = pd.DataFrame({"name": ["Alice"], "age": [25]})
    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
        df.to_parquet(f.name)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


def test_json_param_type_list_of_file_paths(local_mode_ctx):
    result = JsonParamType(list[File]).convert(
        json.dumps(["s3://bucket/a.txt", "s3://bucket/b.txt"]),
        None,
        local_mode_ctx,
    )
    assert len(result) == 2
    assert all(isinstance(f, File) for f in result)
    assert result[0].path == "s3://bucket/a.txt"
    assert result[1].path == "s3://bucket/b.txt"


def test_json_param_type_list_of_dir_paths(local_mode_ctx):
    result = JsonParamType(list[Dir]).convert(
        json.dumps(["s3://bucket/dir-a", "s3://bucket/dir-b"]),
        None,
        local_mode_ctx,
    )
    assert len(result) == 2
    assert all(isinstance(d, Dir) for d in result)
    assert result[0].path == "s3://bucket/dir-a"
    assert result[1].path == "s3://bucket/dir-b"


def test_json_param_type_dict_of_file_paths(local_mode_ctx, temp_txt_file):
    result = JsonParamType(dict[str, File]).convert(
        json.dumps({"remote": "s3://bucket/file.txt", "local": temp_txt_file}),
        None,
        local_mode_ctx,
    )
    assert isinstance(result["remote"], File)
    assert isinstance(result["local"], File)
    assert result["remote"].path == "s3://bucket/file.txt"
    assert result["local"].path == temp_txt_file


def test_json_param_type_dict_of_dir_paths(local_mode_ctx, temp_dir):
    result = JsonParamType(dict[str, Dir]).convert(
        json.dumps({"remote": "s3://bucket/data", "local": temp_dir}),
        None,
        local_mode_ctx,
    )
    assert isinstance(result["remote"], Dir)
    assert isinstance(result["local"], Dir)
    assert result["remote"].path == "s3://bucket/data"
    assert result["local"].path == temp_dir


def test_json_param_type_dataclass_with_file_and_dir(local_mode_ctx):
    @dataclass
    class FlyteTypes:
        flytefile: Optional[File] = None
        flytedir: Optional[Dir] = None

    result = JsonParamType(FlyteTypes).convert(
        json.dumps({"flytefile": "s3://bucket/file.txt", "flytedir": "s3://bucket/dir"}),
        None,
        local_mode_ctx,
    )
    assert isinstance(result, FlyteTypes)
    assert isinstance(result.flytefile, File)
    assert isinstance(result.flytedir, Dir)
    assert result.flytefile.path == "s3://bucket/file.txt"
    assert result.flytedir.path == "s3://bucket/dir"


def test_json_param_type_list_of_dataclass_with_file(local_mode_ctx):
    @dataclass
    class Item:
        flytefile: File

    result = JsonParamType(list[Item]).convert(
        json.dumps([{"flytefile": "s3://bucket/a.txt"}, {"flytefile": "s3://bucket/b.txt"}]),
        None,
        local_mode_ctx,
    )
    assert len(result) == 2
    assert all(isinstance(item, Item) for item in result)
    assert all(isinstance(item.flytefile, File) for item in result)
    assert result[0].flytefile.path == "s3://bucket/a.txt"


def test_json_param_type_nested_dataclass_with_file(local_mode_ctx):
    @dataclass
    class Inner:
        flytefile: File

    @dataclass
    class Outer:
        inner: Inner
        list_inner: Optional[List[Inner]] = None

    result = JsonParamType(Outer).convert(
        json.dumps(
            {
                "inner": {"flytefile": "s3://inner-path"},
                "list_inner": [{"flytefile": "s3://list-1"}, {"flytefile": "s3://list-2"}],
            }
        ),
        None,
        local_mode_ctx,
    )
    assert isinstance(result.inner.flytefile, File)
    assert result.inner.flytefile.path == "s3://inner-path"
    assert len(result.list_inner) == 2
    assert result.list_inner[0].flytefile.path == "s3://list-1"


def test_json_param_type_accepts_structured_dict_for_file_and_dir(local_mode_ctx):
    @dataclass
    class FlyteTypes:
        flytefile: File
        flytedir: Dir

    result = JsonParamType(FlyteTypes).convert(
        json.dumps({"flytefile": {"path": "s3://bucket/file.txt"}, "flytedir": {"path": "s3://bucket/dir"}}),
        None,
        local_mode_ctx,
    )
    assert isinstance(result.flytefile, File)
    assert isinstance(result.flytedir, Dir)
    assert result.flytefile.path == "s3://bucket/file.txt"
    assert result.flytedir.path == "s3://bucket/dir"


def test_json_param_type_passthrough_file_dir_objects(local_mode_ctx):
    existing_file = File(path="s3://already/file.txt")
    existing_dir = Dir(path="s3://already/dir")

    @dataclass
    class FlyteTypes:
        flytefile: File
        flytedir: Dir

    result = JsonParamType(FlyteTypes).convert(
        {"flytefile": existing_file, "flytedir": existing_dir},
        None,
        local_mode_ctx,
    )
    assert result.flytefile is existing_file
    assert result.flytedir is existing_dir


@pytest.mark.skipif(pd is None, reason="pandas is not installed")
def test_json_param_type_list_of_dataframe_paths(local_mode_ctx, temp_parquet_file):
    result = JsonParamType(list[DataFrame]).convert(
        json.dumps(["s3://bucket/data.parquet", temp_parquet_file]),
        None,
        local_mode_ctx,
    )
    assert len(result) == 2
    assert all(isinstance(df, DataFrame) for df in result)
    assert result[0].uri == "s3://bucket/data.parquet"
    assert result[1].uri == temp_parquet_file


@pytest.mark.skipif(pd is None, reason="pandas is not installed")
def test_json_param_type_dict_of_dataframe_paths(local_mode_ctx, temp_parquet_file):
    result = JsonParamType(dict[str, DataFrame]).convert(
        json.dumps({"remote": "s3://bucket/data.parquet", "local": temp_parquet_file}),
        None,
        local_mode_ctx,
    )
    assert isinstance(result["remote"], DataFrame)
    assert isinstance(result["local"], DataFrame)
    assert result["remote"].uri == "s3://bucket/data.parquet"
    assert result["local"].uri == temp_parquet_file


@pytest.mark.skipif(pd is None, reason="pandas is not installed")
def test_json_param_type_dataclass_with_dataframe_path(local_mode_ctx, temp_parquet_file):
    @dataclass
    class DataFrameInput:
        dataframe: Optional[DataFrame] = None

    result = JsonParamType(DataFrameInput).convert(
        json.dumps({"dataframe": temp_parquet_file}),
        None,
        local_mode_ctx,
    )
    assert isinstance(result, DataFrameInput)
    assert isinstance(result.dataframe, DataFrame)
    assert result.dataframe.uri == temp_parquet_file


@pytest.mark.skipif(pd is None, reason="pandas is not installed")
def test_json_param_type_accepts_structured_dict_for_dataframe(local_mode_ctx, temp_parquet_file):
    @dataclass
    class DataFrameInput:
        dataframe: DataFrame

    result = JsonParamType(DataFrameInput).convert(
        json.dumps({"dataframe": {"uri": temp_parquet_file, "format": "parquet"}}),
        None,
        local_mode_ctx,
    )
    assert isinstance(result.dataframe, DataFrame)
    assert result.dataframe.uri == temp_parquet_file
    assert result.dataframe.format == "parquet"
