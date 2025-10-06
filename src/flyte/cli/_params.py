import dataclasses
import datetime
import enum
import importlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import typing
import typing as t
from typing import get_args

import rich_click as click
import yaml
from click import Parameter
from flyteidl2.core.interface_pb2 import Variable
from flyteidl2.core.literals_pb2 import Literal
from flyteidl2.core.types_pb2 import BlobType, LiteralType, SimpleType
from google.protobuf.json_format import MessageToDict
from mashumaro.codecs.json import JSONEncoder

from flyte._logging import logger
from flyte.io import Dir, File
from flyte.types._pickle import FlytePickleTransformer


class StructuredDataset:
    def __init__(self, uri: str | None = None, dataframe: typing.Any = None):
        self.uri = uri
        self.dataframe = dataframe


# ---------------------------------------------------


def key_value_callback(_: typing.Any, param: str, values: typing.List[str]) -> typing.Optional[typing.Dict[str, str]]:
    """
    Callback for click to parse key-value pairs.
    """
    if not values:
        return None
    result = {}
    for v in values:
        if "=" not in v:
            raise click.BadParameter(f"Expected key-value pair of the form key=value, got {v}")
        k, val = v.split("=", 1)
        result[k.strip()] = val.strip()
    return result


def labels_callback(_: typing.Any, param: str, values: typing.List[str]) -> typing.Optional[typing.Dict[str, str]]:
    """
    Callback for click to parse labels.
    """
    if not values:
        return None
    result = {}
    for v in values:
        if "=" not in v:
            result[v.strip()] = ""
        else:
            k, val = v.split("=", 1)
            result[k.strip()] = val.strip()
    return result


class DirParamType(click.ParamType):
    name = "directory path"

    def convert(
        self, value: typing.Any, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> typing.Any:
        from flyte.storage import is_remote

        if not is_remote(value):
            p = pathlib.Path(value)
            if not p.exists() or not p.is_dir():
                raise click.BadParameter(f"parameter should be a valid flytedirectory path, {value}")
        return Dir(path=value)


class StructuredDatasetParamType(click.ParamType):
    """
    TODO handle column types
    """

    name = "structured dataset path (dir/file)"

    def convert(
        self, value: typing.Any, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> typing.Any:
        if isinstance(value, str):
            return StructuredDataset(uri=value)
        elif isinstance(value, StructuredDataset):
            return value
        return StructuredDataset(dataframe=value)


class FileParamType(click.ParamType):
    name = "file path"

    def convert(
        self, value: typing.Any, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> typing.Any:
        from flyte.storage import is_remote

        if not is_remote(value):
            p = pathlib.Path(value)
            if not p.exists() or not p.is_file():
                raise click.BadParameter(f"parameter should be a valid file path, {value}")
        return File.from_existing_remote(value)


class PickleParamType(click.ParamType):
    name = "pickle"

    def get_metavar(self, param: "Parameter", *args) -> t.Optional[str]:
        return "Python Object <Module>:<Object>"

    def convert(
        self, value: typing.Any, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> typing.Any:
        if not isinstance(value, str):
            return value
        parts = value.split(":")
        if len(parts) != 2:
            if ctx and ctx.obj and ctx.obj.log_level >= 10:  # DEBUG level
                click.echo(f"Did not receive a string in the expected format <MODULE>:<VAR>, falling back to: {value}")
            return value
        try:
            sys.path.insert(0, os.getcwd())
            m = importlib.import_module(parts[0])
            return m.__getattribute__(parts[1])
        except ModuleNotFoundError as e:
            raise click.BadParameter(f"Failed to import module {parts[0]}, error: {e}")
        except AttributeError as e:
            raise click.BadParameter(f"Failed to find attribute {parts[1]} in module {parts[0]}, error: {e}")


class JSONIteratorParamType(click.ParamType):
    name = "json iterator"

    def convert(
        self, value: typing.Any, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> typing.Any:
        return value


def parse_iso8601_duration(iso_duration: str) -> datetime.timedelta:
    pattern = re.compile(
        r"^P"  # Starts with 'P'
        r"(?:(?P<days>\d+)D)?"  # Optional days
        r"(?:T"  # Optional time part
        r"(?:(?P<hours>\d+)H)?"
        r"(?:(?P<minutes>\d+)M)?"
        r"(?:(?P<seconds>\d+)S)?"
        r")?$"
    )
    match = pattern.match(iso_duration)
    if not match:
        raise ValueError(f"Invalid ISO 8601 duration format: {iso_duration}")

    parts = {k: int(v) if v else 0 for k, v in match.groupdict().items()}
    return datetime.timedelta(**parts)


def parse_human_durations(text: str) -> list[datetime.timedelta]:
    raw_parts = text.strip("[]").split("|")
    durations = []

    for part in raw_parts:
        new_part = part.strip().lower()

        # Match 1:24 or :45
        m_colon = re.match(r"^(?:(\d+):)?(\d+)$", new_part)
        if m_colon:
            minutes = int(m_colon.group(1)) if m_colon.group(1) else 0
            seconds = int(m_colon.group(2))
            durations.append(datetime.timedelta(minutes=minutes, seconds=seconds))
            continue

        # Match "10 days", "1 minute", etc.
        m_units = re.match(r"^(\d+)\s*(day|hour|minute|second)s?$", new_part)
        if m_units:
            value = int(m_units.group(1))
            unit = m_units.group(2)
            durations.append(datetime.timedelta(**{unit + "s": value}))
            continue

        print(f"Warning: could not parse '{part}'")

    return durations


def parse_duration(s: str) -> datetime.timedelta:
    try:
        return parse_iso8601_duration(s)
    except ValueError:
        parts = parse_human_durations(s)
        if not parts:
            raise ValueError(f"Could not parse duration: {s}")
        return sum(parts, datetime.timedelta())


class DateTimeType(click.DateTime):
    _NOW_FMT = "now"
    _TODAY_FMT = "today"
    _FIXED_FORMATS: typing.ClassVar[typing.List[str]] = [_NOW_FMT, _TODAY_FMT]
    _FLOATING_FORMATS: typing.ClassVar[typing.List[str]] = ["<FORMAT> - <ISO8601 duration>"]
    _ADDITONAL_FORMATS: typing.ClassVar[typing.List[str]] = [*_FIXED_FORMATS, *_FLOATING_FORMATS]
    _FLOATING_FORMAT_PATTERN = r"(.+)\s+([-+])\s+(.+)"

    def __init__(self):
        super().__init__()
        self.formats.extend(self._ADDITONAL_FORMATS)

    def _datetime_from_format(
        self, value: str, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> datetime.datetime:
        if value in self._FIXED_FORMATS:
            if value == self._NOW_FMT:
                return datetime.datetime.now()
            if value == self._TODAY_FMT:
                n = datetime.datetime.now()
                return datetime.datetime(n.year, n.month, n.day)
        return super().convert(value, param, ctx)

    def convert(
        self, value: typing.Any, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> typing.Any:
        if isinstance(value, str) and " " in value:
            import re

            m = re.match(self._FLOATING_FORMAT_PATTERN, value)
            if m:
                parts = m.groups()
                if len(parts) != 3:
                    raise click.BadParameter(f"Expected format <FORMAT> - <ISO8601 duration>, got {value}")
                dt = self._datetime_from_format(parts[0], param, ctx)
                try:
                    delta = parse_duration(parts[2])
                except Exception as e:
                    raise click.BadParameter(
                        f"Matched format {self._FLOATING_FORMATS}, but failed to parse duration {parts[2]}, error: {e}"
                    )
                if parts[1] == "-":
                    return dt - delta
                return dt + delta
            else:
                value = datetime.datetime.fromisoformat(value)

        return self._datetime_from_format(value, param, ctx)


class DurationParamType(click.ParamType):
    name = "[1:24 | :22 | 1 minute | 10 days | ...]"

    def convert(
        self, value: typing.Any, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> typing.Any:
        if value is None:
            raise click.BadParameter("None value cannot be converted to a Duration type.")
        return parse_duration(value)


class EnumParamType(click.Choice):
    def __init__(self, enum_type: typing.Type[enum.Enum]):
        super().__init__([str(e.value) for e in enum_type])
        self._enum_type = enum_type

    def convert(
        self, value: typing.Any, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> enum.Enum:
        if isinstance(value, self._enum_type):
            return value
        return self._enum_type(super().convert(value, param, ctx))


class UnionParamType(click.ParamType):
    """
    A composite type that allows for multiple types to be specified. This is used for union types.
    """

    def __init__(self, types: typing.List[click.ParamType | None]):
        super().__init__()
        self._types = self._sort_precedence(types)
        self.name = "|".join([t.name for t in self._types if t is not None])
        self.optional = False
        if None in types:
            self.name = f"Optional[{self.name}]"
            self.optional = True

    @staticmethod
    def _sort_precedence(tp: typing.List[click.ParamType | None]) -> typing.List[click.ParamType]:
        unprocessed = []
        str_types = []
        others = []
        for p in tp:
            if isinstance(p, type(click.UNPROCESSED)):
                unprocessed.append(p)
            elif isinstance(p, type(click.STRING)):
                str_types.append(p)
            else:
                others.append(p)
        return others + str_types + unprocessed  # type: ignore

    def convert(
        self, value: typing.Any, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> typing.Any:
        """
        Important to implement NoneType / Optional.
        Also could we just determine the click types from the python types
        """
        for p in self._types:
            try:
                if p is None and value is None:
                    return None
                return p.convert(value, param, ctx)
            except Exception as e:
                logger.debug(f"Ignoring conversion error for type {p} trying other variants in Union. Error: {e}")
        raise click.BadParameter(f"Failed to convert {value} to any of the types {self._types}")


class JsonParamType(click.ParamType):
    name = "json object OR json/yaml file path"

    def __init__(self, python_type: typing.Type):
        super().__init__()
        self._python_type = python_type

    def _parse(self, value: typing.Any, param: typing.Optional[click.Parameter]):
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            try:
                # We failed to load the json, so we'll try to load it as a file
                if os.path.exists(value):
                    # if the value is a yaml file, we'll try to load it as yaml
                    if value.endswith((".yaml", "yml")):
                        with open(value, "r") as f:
                            return yaml.safe_load(f)
                    with open(value, "r") as f:
                        return json.load(f)
                raise
            except json.JSONDecodeError as e:
                raise click.BadParameter(f"parameter {param} should be a valid json object, {value}, error: {e}")

    def convert(
        self, value: typing.Any, param: typing.Optional[click.Parameter], ctx: typing.Optional[click.Context]
    ) -> typing.Any:
        if value is None:
            raise click.BadParameter("None value cannot be converted to a Json type.")

        parsed_value = self._parse(value, param)

        # We compare the origin type because the json parsed value for list or dict is always a list or dict without
        # the covariant type information.
        if type(parsed_value) is typing.get_origin(self._python_type) or type(parsed_value) is self._python_type:
            # Indexing the return value of get_args will raise an error for native dict and list types.
            # We don't support native list/dict types with nested dataclasses.
            if get_args(self._python_type) == ():
                return parsed_value
            elif isinstance(parsed_value, list) and dataclasses.is_dataclass(get_args(self._python_type)[0]):
                j = JsonParamType(get_args(self._python_type)[0])
                # turn object back into json string
                return [j.convert(json.dumps(v), param, ctx) for v in parsed_value]
            elif isinstance(parsed_value, dict) and dataclasses.is_dataclass(get_args(self._python_type)[1]):
                j = JsonParamType(get_args(self._python_type)[1])
                # turn object back into json string
                return {k: j.convert(json.dumps(v), param, ctx) for k, v in parsed_value.items()}

            return parsed_value

        from pydantic import BaseModel

        if issubclass(self._python_type, BaseModel):
            return typing.cast(BaseModel, self._python_type).model_validate_json(
                json.dumps(parsed_value), strict=False, context={"deserialize": True}
            )
        elif dataclasses.is_dataclass(self._python_type):
            from mashumaro.codecs.json import JSONDecoder

            decoder = JSONDecoder(self._python_type)
            return decoder.decode(value)

        return parsed_value


SIMPLE_TYPE_CONVERTER = {
    SimpleType.FLOAT: click.FLOAT,
    SimpleType.INTEGER: click.INT,
    SimpleType.STRING: click.STRING,
    SimpleType.BOOLEAN: click.BOOL,
    SimpleType.DURATION: DurationParamType(),
    SimpleType.DATETIME: DateTimeType(),
}


def literal_type_to_click_type(lt: LiteralType, python_type: typing.Type) -> click.ParamType:
    """
    Converts a Flyte LiteralType given a python_type to a click.ParamType
    """
    if lt.HasField("simple"):
        if lt.simple == SimpleType.STRUCT:
            ct = JsonParamType(python_type)
            ct.name = f"JSON object {python_type.__name__}"
            return ct
        if lt.simple in SIMPLE_TYPE_CONVERTER:
            return SIMPLE_TYPE_CONVERTER[lt.simple]
        raise NotImplementedError(f"Type {lt.simple} is not supported in `flyte run`")

    if lt.HasField("structured_dataset_type"):
        return StructuredDatasetParamType()

    if lt.HasField("collection_type") or lt.HasField("map_value_type"):
        ct = JsonParamType(python_type)
        if lt.HasField("collection_type"):
            ct.name = "json list"
        else:
            ct.name = "json dictionary"
        return ct

    if lt.HasField("blob"):
        if lt.blob.dimensionality == BlobType.BlobDimensionality.SINGLE:
            if lt.blob.format == FlytePickleTransformer.PYTHON_PICKLE_FORMAT:
                return PickleParamType()
            # TODO: Add JSONIteratorTransformer
            # elif lt.blob.format == JSONIteratorTransformer.JSON_ITERATOR_FORMAT:
            #     return JSONIteratorParamType()
            return FileParamType()
        return DirParamType()

    if lt.HasField("union_type"):
        cts = []
        for i in range(len(lt.union_type.variants)):
            variant = lt.union_type.variants[i]
            variant_python_type = typing.get_args(python_type)[i]
            if variant_python_type is type(None):
                cts.append(None)
            else:
                cts.append(literal_type_to_click_type(variant, variant_python_type))
        return UnionParamType(cts)

    if lt.HasField("enum_type"):
        return EnumParamType(python_type)  # type: ignore

    return click.UNPROCESSED


class FlyteLiteralConverter(object):
    name = "literal_type"

    def __init__(
        self,
        literal_type: LiteralType,
        python_type: typing.Type,
    ):
        self._literal_type = literal_type
        self._python_type = python_type
        self._click_type = literal_type_to_click_type(literal_type, python_type)

    @property
    def click_type(self) -> click.ParamType:
        return self._click_type

    def is_bool(self) -> bool:
        return self.click_type == click.BOOL

    def is_optional(self) -> bool:
        return isinstance(self.click_type, UnionParamType) and self.click_type.optional

    def convert(
        self, ctx: click.Context, param: typing.Optional[click.Parameter], value: typing.Any
    ) -> typing.Union[Literal, typing.Any]:
        """
        Convert the value to a python native type. This is used by click to convert the input.
        """
        try:
            # If the expected Python type is datetime.date, adjust the value to date
            if self._python_type is datetime.date:
                # Click produces datetime, so converting to date to avoid type mismatch error
                value = value.date()

            return value
        except click.BadParameter:
            raise
        except Exception as e:
            raise click.BadParameter(
                f"Failed to convert param: {param if param else 'NA'}, value: {value} to type: {self._python_type}."
                f" Reason {e}"
            ) from e


def to_click_option(
    input_name: str,
    literal_var: Variable,
    python_type: typing.Type,
    default_val: typing.Any,
) -> click.Option:
    """
    This handles converting workflow input types to supported click parameters with callbacks to initialize
    the input values to their expected types.
    """
    from flyteidl2.core.types_pb2 import SimpleType

    if input_name != input_name.lower():
        # Click does not support uppercase option names: https://github.com/pallets/click/issues/837
        raise ValueError(f"Workflow input name must be lowercase: {input_name!r}")

    literal_converter = FlyteLiteralConverter(
        literal_type=literal_var.type,
        python_type=python_type,
    )

    if literal_converter.is_bool() and not default_val:
        default_val = False

    description_extra = ""
    if literal_var.type.simple == SimpleType.STRUCT:
        if default_val:
            # pydantic v2
            if hasattr(default_val, "model_dump_json"):
                default_val = default_val.model_dump_json()
            else:
                encoder = JSONEncoder(python_type)
                default_val = encoder.encode(default_val)
        if literal_var.type.metadata:
            description_extra = f": {MessageToDict(literal_var.type.metadata)}"

    required = False if default_val is not None else True
    is_flag: typing.Optional[bool] = None
    param_decls = [f"--{input_name}"]
    if literal_converter.is_bool():
        required = False
        is_flag = True
        if default_val is True:
            param_decls = [f"--{input_name}/--no-{input_name}"]
    if literal_converter.is_optional():
        required = False

    return click.Option(
        param_decls=param_decls,
        type=literal_converter.click_type,
        is_flag=is_flag,
        default=default_val,
        show_default=True,
        required=required,
        help=literal_var.description + description_extra,
        callback=literal_converter.convert,
    )
