"""Tests for File, Dir, and DataFrame inside nested Pydantic models.

Verifies that these special types roundtrip correctly through
TypeEngine.to_literal / to_python_value when nested in containers
(List, Dict, Optional) within Pydantic BaseModel classes.
"""

import dataclasses
import typing

import pytest
from pydantic import BaseModel

from flyte.io import DataFrame, Dir, File
from flyte.types import TypeEngine

# -- Models --


class ModelWithFile(BaseModel):
    file: File


class ModelWithDir(BaseModel):
    dir: Dir


class ModelWithListOfFiles(BaseModel):
    files: typing.List[File]


class ModelWithDictOfFiles(BaseModel):
    file_map: typing.Dict[str, File]


class ModelWithListOfDirs(BaseModel):
    dirs: typing.List[Dir]


class ModelWithDictOfDirs(BaseModel):
    dir_map: typing.Dict[str, Dir]


class ModelWithNestedFiles(BaseModel):
    nested_files: typing.List[typing.List[File]]


class ModelWithOptionalFile(BaseModel):
    file: typing.Optional[File] = None


class ModelWithOptionalDir(BaseModel):
    dir: typing.Optional[Dir] = None


class ModelWithDataFrame(BaseModel):
    df: DataFrame


class ModelWithListOfDataFrames(BaseModel):
    dfs: typing.List[DataFrame]


class ModelWithOptionalDataFrame(BaseModel):
    df: typing.Optional[DataFrame] = None


class CombinedModel(BaseModel):
    files: typing.List[File]
    dir_map: typing.Dict[str, Dir]
    df: DataFrame
    optional_file: typing.Optional[File] = None


# -- File tests --


@pytest.mark.asyncio
async def test_file_in_model_roundtrip():
    input_val = ModelWithFile(file=File(path="dummy.txt"))
    lit = TypeEngine.to_literal_type(ModelWithFile)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithFile, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithFile)
    assert pv == input_val


@pytest.mark.asyncio
async def test_list_of_files_roundtrip():
    input_val = ModelWithListOfFiles(files=[File(path="a.txt"), File(path="b.txt")])
    lit = TypeEngine.to_literal_type(ModelWithListOfFiles)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithListOfFiles, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithListOfFiles)
    assert pv == input_val


@pytest.mark.asyncio
async def test_dict_of_files_roundtrip():
    input_val = ModelWithDictOfFiles(file_map={"x": File(path="c.txt"), "y": File(path="d.txt")})
    lit = TypeEngine.to_literal_type(ModelWithDictOfFiles)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithDictOfFiles, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithDictOfFiles)
    assert pv == input_val


@pytest.mark.asyncio
async def test_nested_list_of_files_roundtrip():
    input_val = ModelWithNestedFiles(nested_files=[[File(path="a.txt")], [File(path="b.txt"), File(path="c.txt")]])
    lit = TypeEngine.to_literal_type(ModelWithNestedFiles)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithNestedFiles, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithNestedFiles)
    assert pv == input_val


@pytest.mark.asyncio
async def test_optional_file_with_value_roundtrip():
    input_val = ModelWithOptionalFile(file=File(path="e.txt"))
    lit = TypeEngine.to_literal_type(ModelWithOptionalFile)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithOptionalFile, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithOptionalFile)
    assert pv == input_val


@pytest.mark.asyncio
async def test_optional_file_with_none_roundtrip():
    input_val = ModelWithOptionalFile(file=None)
    lit = TypeEngine.to_literal_type(ModelWithOptionalFile)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithOptionalFile, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithOptionalFile)
    assert pv.file is None


# -- Dir tests --


@pytest.mark.asyncio
async def test_dir_in_model_roundtrip():
    input_val = ModelWithDir(dir=Dir(path="/tmp/mydir"))
    lit = TypeEngine.to_literal_type(ModelWithDir)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithDir, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithDir)
    assert pv == input_val


@pytest.mark.asyncio
async def test_list_of_dirs_roundtrip():
    input_val = ModelWithListOfDirs(dirs=[Dir(path="/tmp/d1"), Dir(path="/tmp/d2")])
    lit = TypeEngine.to_literal_type(ModelWithListOfDirs)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithListOfDirs, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithListOfDirs)
    assert pv == input_val


@pytest.mark.asyncio
async def test_dict_of_dirs_roundtrip():
    input_val = ModelWithDictOfDirs(dir_map={"a": Dir(path="/tmp/d1"), "b": Dir(path="/tmp/d2")})
    lit = TypeEngine.to_literal_type(ModelWithDictOfDirs)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithDictOfDirs, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithDictOfDirs)
    assert pv == input_val


@pytest.mark.asyncio
async def test_optional_dir_with_none_roundtrip():
    input_val = ModelWithOptionalDir(dir=None)
    lit = TypeEngine.to_literal_type(ModelWithOptionalDir)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithOptionalDir, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithOptionalDir)
    assert pv.dir is None


# -- DataFrame tests --
# Note: DataFrame.__eq__ compares by identity, so we use model_dump() for assertions.


@pytest.mark.asyncio
async def test_dataframe_in_model_roundtrip():
    input_val = ModelWithDataFrame(df=DataFrame(uri="s3://bucket/data.parquet"))
    lit = TypeEngine.to_literal_type(ModelWithDataFrame)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithDataFrame, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithDataFrame)
    assert pv.model_dump() == input_val.model_dump()


@pytest.mark.asyncio
async def test_list_of_dataframes_roundtrip():
    input_val = ModelWithListOfDataFrames(
        dfs=[DataFrame(uri="s3://bucket/d1.parquet"), DataFrame(uri="s3://bucket/d2.parquet")]
    )
    lit = TypeEngine.to_literal_type(ModelWithListOfDataFrames)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithListOfDataFrames, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithListOfDataFrames)
    assert pv.model_dump() == input_val.model_dump()


@pytest.mark.asyncio
async def test_optional_dataframe_with_value_roundtrip():
    input_val = ModelWithOptionalDataFrame(df=DataFrame(uri="s3://bucket/opt.parquet"))
    lit = TypeEngine.to_literal_type(ModelWithOptionalDataFrame)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithOptionalDataFrame, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithOptionalDataFrame)
    assert pv.model_dump() == input_val.model_dump()


@pytest.mark.asyncio
async def test_optional_dataframe_with_none_roundtrip():
    input_val = ModelWithOptionalDataFrame(df=None)
    lit = TypeEngine.to_literal_type(ModelWithOptionalDataFrame)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithOptionalDataFrame, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, ModelWithOptionalDataFrame)
    assert pv.df is None


# -- Combined: File + Dir + DataFrame --


@pytest.mark.asyncio
async def test_combined_model_roundtrip():
    input_val = CombinedModel(
        files=[File(path="a.txt"), File(path="b.txt")],
        dir_map={"d1": Dir(path="/tmp/d1"), "d2": Dir(path="/tmp/d2")},
        df=DataFrame(uri="s3://bucket/data.parquet"),
        optional_file=File(path="opt.txt"),
    )
    lit = TypeEngine.to_literal_type(CombinedModel)
    lv = await TypeEngine.to_literal(input_val, python_type=CombinedModel, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, CombinedModel)
    assert pv.files == input_val.files
    assert pv.dir_map == input_val.dir_map
    assert pv.df.model_dump() == input_val.df.model_dump()
    assert pv.optional_file == input_val.optional_file


@pytest.mark.asyncio
async def test_combined_model_with_none_roundtrip():
    input_val = CombinedModel(
        files=[],
        dir_map={},
        df=DataFrame(uri="s3://bucket/data.parquet"),
        optional_file=None,
    )
    lit = TypeEngine.to_literal_type(CombinedModel)
    lv = await TypeEngine.to_literal(input_val, python_type=CombinedModel, expected=lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, CombinedModel)
    assert pv.files == []
    assert pv.dir_map == {}
    assert pv.optional_file is None


# -- guess_python_type: deep structure verification --


def test_list_of_files_guess_type_structure():
    """Verify guess_python_type reconstructs List[File] with correct nested structure."""
    lit = TypeEngine.to_literal_type(ModelWithListOfFiles)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    hints = typing.get_type_hints(guessed)
    files_type = hints.get("files")
    assert files_type is not None, "files field not found in guessed type"

    origin = typing.get_origin(files_type)
    assert origin is list, f"Expected List, got {origin}"

    args = typing.get_args(files_type)
    assert len(args) == 1
    # The inner type should be File (via schema_match) or a dataclass with File's fields
    inner = args[0]
    assert inner is File or dataclasses.is_dataclass(inner), f"Expected File or dataclass, got {inner}"


def test_nested_list_of_files_guess_type_structure():
    """Verify guess_python_type reconstructs List[List[File]] — inner should be List, not str."""
    lit = TypeEngine.to_literal_type(ModelWithNestedFiles)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    hints = typing.get_type_hints(guessed)
    nested_type = hints.get("nested_files")
    assert nested_type is not None, "nested_files field not found"

    # Outer should be List
    assert typing.get_origin(nested_type) is list, f"Expected outer List, got {typing.get_origin(nested_type)}"

    # Inner should be List[...], not str
    inner_type = typing.get_args(nested_type)[0]
    inner_origin = typing.get_origin(inner_type)
    assert inner_origin is list, f"Expected inner List, got {inner_type} (origin: {inner_origin})"

    # Innermost should be File or a dataclass with File's fields
    innermost = typing.get_args(inner_type)[0]
    assert innermost is File or dataclasses.is_dataclass(innermost), f"Expected File or dataclass, got {innermost}"


def test_dict_of_dirs_guess_type_structure():
    """Verify guess_python_type reconstructs Dict[str, Dir] with correct value type."""
    lit = TypeEngine.to_literal_type(ModelWithDictOfDirs)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    hints = typing.get_type_hints(guessed)
    dir_map_type = hints.get("dir_map")
    assert dir_map_type is not None, "dir_map field not found"

    assert typing.get_origin(dir_map_type) is dict, f"Expected Dict, got {typing.get_origin(dir_map_type)}"

    key_type, val_type = typing.get_args(dir_map_type)
    assert key_type is str, f"Expected str key, got {key_type}"
    assert val_type is Dir or dataclasses.is_dataclass(val_type), f"Expected Dir or dataclass, got {val_type}"


def test_list_of_dataframes_guess_type_structure():
    """Verify guess_python_type reconstructs List[DataFrame] with correct inner type."""
    lit = TypeEngine.to_literal_type(ModelWithListOfDataFrames)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    hints = typing.get_type_hints(guessed)
    dfs_type = hints.get("dfs")
    assert dfs_type is not None, "dfs field not found"

    assert typing.get_origin(dfs_type) is list, f"Expected List, got {typing.get_origin(dfs_type)}"

    inner = typing.get_args(dfs_type)[0]
    assert inner is DataFrame or dataclasses.is_dataclass(inner), f"Expected DataFrame or dataclass, got {inner}"


def test_combined_model_guess_type_structure():
    """Verify guess_python_type reconstructs CombinedModel with all field types correct."""
    lit = TypeEngine.to_literal_type(CombinedModel)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    hints = typing.get_type_hints(guessed)

    # files: List[File]
    files_type = hints["files"]
    assert typing.get_origin(files_type) is list
    files_inner = typing.get_args(files_type)[0]
    assert files_inner is File or dataclasses.is_dataclass(files_inner)

    # dir_map: Dict[str, Dir]
    dir_map_type = hints["dir_map"]
    assert typing.get_origin(dir_map_type) is dict
    _, dir_val = typing.get_args(dir_map_type)
    assert dir_val is Dir or dataclasses.is_dataclass(dir_val)

    # optional_file: Optional[File] — optional fields with defaults are dropped
    # by guess_python_type, so we just verify it's absent rather than asserting structure
    if "optional_file" in hints:
        opt_type = hints["optional_file"]
        assert typing.get_origin(opt_type) is typing.Union
        non_none_args = [a for a in typing.get_args(opt_type) if a is not type(None)]
        assert len(non_none_args) == 1
        assert non_none_args[0] is File or dataclasses.is_dataclass(non_none_args[0])


# -- Roundtrip via guessed type (simulates pyflyte run) --


@pytest.mark.asyncio
async def test_list_of_files_roundtrip_via_guessed_type():
    """Roundtrip List[File] using the guessed type instead of the original Pydantic model."""
    input_val = ModelWithListOfFiles(files=[File(path="a.txt"), File(path="b.txt")])
    lit = TypeEngine.to_literal_type(ModelWithListOfFiles)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithListOfFiles, expected=lit)

    guessed = TypeEngine.guess_python_type(lit)
    pv = await TypeEngine.to_python_value(lv, guessed)
    assert dataclasses.is_dataclass(pv)
    assert len(pv.files) == 2


@pytest.mark.asyncio
async def test_nested_files_roundtrip_via_guessed_type():
    """Roundtrip List[List[File]] using guessed type — verifies nested structure survives."""
    input_val = ModelWithNestedFiles(nested_files=[[File(path="a.txt")], [File(path="b.txt"), File(path="c.txt")]])
    lit = TypeEngine.to_literal_type(ModelWithNestedFiles)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithNestedFiles, expected=lit)

    guessed = TypeEngine.guess_python_type(lit)
    pv = await TypeEngine.to_python_value(lv, guessed)
    assert dataclasses.is_dataclass(pv)
    assert len(pv.nested_files) == 2
    assert len(pv.nested_files[0]) == 1
    assert len(pv.nested_files[1]) == 2


@pytest.mark.asyncio
async def test_dict_of_dirs_roundtrip_via_guessed_type():
    """Roundtrip Dict[str, Dir] using guessed type."""
    input_val = ModelWithDictOfDirs(dir_map={"a": Dir(path="/tmp/d1"), "b": Dir(path="/tmp/d2")})
    lit = TypeEngine.to_literal_type(ModelWithDictOfDirs)
    lv = await TypeEngine.to_literal(input_val, python_type=ModelWithDictOfDirs, expected=lit)

    guessed = TypeEngine.guess_python_type(lit)
    pv = await TypeEngine.to_python_value(lv, guessed)
    assert dataclasses.is_dataclass(pv)
    assert len(pv.dir_map) == 2
    assert "a" in pv.dir_map
    assert "b" in pv.dir_map


@pytest.mark.asyncio
async def test_combined_model_roundtrip_via_guessed_type():
    """Roundtrip CombinedModel using guessed type — verifies File, Dir, DataFrame all survive."""
    input_val = CombinedModel(
        files=[File(path="a.txt")],
        dir_map={"d1": Dir(path="/tmp/d1")},
        df=DataFrame(uri="s3://bucket/data.parquet"),
        optional_file=File(path="opt.txt"),
    )
    lit = TypeEngine.to_literal_type(CombinedModel)
    lv = await TypeEngine.to_literal(input_val, python_type=CombinedModel, expected=lit)

    guessed = TypeEngine.guess_python_type(lit)
    pv = await TypeEngine.to_python_value(lv, guessed)
    assert dataclasses.is_dataclass(pv)
    assert len(pv.files) == 1
    assert len(pv.dir_map) == 1
