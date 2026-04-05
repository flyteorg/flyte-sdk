import dataclasses

from mashumaro.jsonschema import build_json_schema

from flyte.io import DataFrame, Dir, File
from flyte.types._type_engine import PydanticSchemaPlugin


def test_file_schema_description_is_short():
    schema = File.model_json_schema()
    assert schema.get("description") == "A file reference with an optional format type."
    assert schema.get("x-flyte-type") == "file"


def test_dir_schema_description_is_short():
    schema = Dir.model_json_schema()
    assert schema.get("description") == "A directory reference with an optional format type."
    assert schema.get("x-flyte-type") == "dir"


def test_dataframe_schema_description_is_short():
    schema = DataFrame.model_json_schema()
    assert schema.get("description") == "A tabular data reference backed by a remote file."
    assert schema.get("x-flyte-type") == "dataframe"


@dataclasses.dataclass
class JobWithFile:
    input_file: File
    label: str


@dataclasses.dataclass
class JobWithDir:
    input_dir: Dir
    label: str


@dataclasses.dataclass
class JobWithDataFrame:
    input_df: DataFrame
    label: str


def _dataclass_schema(cls):
    return build_json_schema(cls, plugins=[PydanticSchemaPlugin()]).to_dict()


def test_dataclass_with_file_field_description_is_short():
    schema = _dataclass_schema(JobWithFile)
    assert schema["properties"]["input_file"].get("description") == "A file reference with an optional format type."


def test_dataclass_with_dir_field_description_is_short():
    schema = _dataclass_schema(JobWithDir)
    assert schema["properties"]["input_dir"].get("description") == "A directory reference with an optional format type."


def test_dataclass_with_dataframe_field_description_is_short():
    schema = _dataclass_schema(JobWithDataFrame)
    assert schema["properties"]["input_df"].get("description") == "A tabular data reference backed by a remote file."
