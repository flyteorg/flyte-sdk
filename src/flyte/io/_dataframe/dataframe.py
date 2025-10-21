from __future__ import annotations

import _datetime
import collections
import types
import typing
from abc import ABC, abstractmethod
from dataclasses import is_dataclass
from typing import Any, ClassVar, Coroutine, Dict, Generic, List, Optional, Type, Union

from flyteidl2.core import literals_pb2, types_pb2
from fsspec.utils import get_protocol
from mashumaro.types import SerializableType
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_serializer, model_validator
from typing_extensions import Annotated, TypeAlias, get_args, get_origin

import flyte.storage as storage
from flyte._logging import logger
from flyte._utils import lazy_module
from flyte._utils.asyn import loop_manager
from flyte.types import TypeEngine, TypeTransformer, TypeTransformerFailedError
from flyte.types._renderer import Renderable
from flyte.types._type_engine import modify_literal_uris

MESSAGEPACK = "msgpack"


if typing.TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa
else:
    pd = lazy_module("pandas")
    pa = lazy_module("pyarrow")

T = typing.TypeVar("T")  # DataFrame type or a dataframe type
DF = typing.TypeVar("DF")  # Dataframe type

# For specifying the storage formats of DataFrames. It's just a string, nothing fancy.
DataFrameFormat: TypeAlias = str

# Storage formats
PARQUET: DataFrameFormat = "parquet"
CSV: DataFrameFormat = "csv"
GENERIC_FORMAT: DataFrameFormat = ""
GENERIC_PROTOCOL: str = "generic protocol"


class DataFrame(BaseModel, SerializableType):
    """
    This is the user facing DataFrame class. Please don't confuse it with the literals.StructuredDataset
    class (that is just a model, a Python class representation of the protobuf).
    """

    uri: typing.Optional[str] = Field(default=None)
    format: typing.Optional[str] = Field(default=GENERIC_FORMAT)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Private attributes that are not part of the Pydantic model schema
    _raw_df: typing.Optional[typing.Any] = PrivateAttr(default=None)
    _metadata: typing.Optional[literals_pb2.StructuredDatasetMetadata] = PrivateAttr(default=None)
    _literal_sd: Optional[literals_pb2.StructuredDataset] = PrivateAttr(default=None)
    _dataframe_type: Optional[Type[Any]] = PrivateAttr(default=None)
    _already_uploaded: bool = PrivateAttr(default=False)

    # loop manager is working better than synchronicity for some reason, was getting an error but may be an easy fix
    def _serialize(self) -> Dict[str, Optional[str]]:
        # dataclass case
        lt = TypeEngine.to_literal_type(type(self))
        engine = DataFrameTransformerEngine()
        lv = loop_manager.run_sync(engine.to_literal, self, type(self), lt)
        sd = DataFrame(uri=lv.scalar.structured_dataset.uri)
        sd.format = lv.scalar.structured_dataset.metadata.structured_dataset_type.format
        return {
            "uri": sd.uri,
            "format": sd.format,
        }

    @classmethod
    def _deserialize(cls, value) -> DataFrame:
        uri = value.get("uri", None)
        format_val = value.get("format", None)

        if uri is None:
            raise ValueError("DataFrame's uri and file format should not be None")

        engine = DataFrameTransformerEngine()
        return loop_manager.run_sync(
            engine.to_python_value,
            literals_pb2.Literal(
                scalar=literals_pb2.Scalar(
                    structured_dataset=literals_pb2.StructuredDataset(
                        metadata=literals_pb2.StructuredDatasetMetadata(
                            structured_dataset_type=types_pb2.StructuredDatasetType(format=format_val)
                        ),
                        uri=uri,
                    )
                )
            ),
            cls,
        )

    @model_serializer
    def serialize_dataframe(self) -> Dict[str, Optional[str]]:
        lt = TypeEngine.to_literal_type(type(self))
        sde = DataFrameTransformerEngine()
        lv = loop_manager.run_sync(sde.to_literal, self, type(self), lt)
        return {
            "uri": lv.scalar.structured_dataset.uri,
            "format": lv.scalar.structured_dataset.metadata.structured_dataset_type.format,
        }

    @model_validator(mode="after")
    def deserialize_dataframe(self, info) -> DataFrame:
        if info.context is None or info.context.get("deserialize") is not True:
            return self

        engine = DataFrameTransformerEngine()
        return loop_manager.run_sync(
            engine.to_python_value,
            literals_pb2.Literal(
                scalar=literals_pb2.Scalar(
                    structured_dataset=literals_pb2.StructuredDataset(
                        metadata=literals_pb2.StructuredDatasetMetadata(
                            structured_dataset_type=types_pb2.StructuredDatasetType(format=self.format)
                        ),
                        uri=self.uri,
                    )
                )
            ),
            type(self),
        )

    @classmethod
    def columns(cls) -> typing.Dict[str, typing.Type]:
        return {}

    @classmethod
    def column_names(cls) -> typing.List[str]:
        return [k for k, v in cls.columns().items()]

    @classmethod
    def from_df(
        cls,
        val: typing.Optional[typing.Any] = None,
        uri: typing.Optional[str] = None,
    ) -> DataFrame:
        """
        Wrapper to create a DataFrame from a dataframe.
        The reason this is implemented as a wrapper instead of a full translation invoking
        the type engine and the encoders is because there's too much information in the type
        signature of the task that we don't want the user to have to replicate.
        """
        instance = cls(uri=uri)
        instance._raw_df = val
        return instance

    @classmethod
    def from_existing_remote(
        cls,
        remote_path: str,
        format: typing.Optional[str] = None,
        **kwargs,
    ) -> "DataFrame":
        """
        Create a DataFrame reference from an existing remote dataframe.

        Args:
            remote_path: The remote path to the existing dataframe
            format: Format of the stored dataframe

        Example:
            ```python
            df = DataFrame.from_existing_remote("s3://bucket/data.parquet", format="parquet")
            ```
        """
        return cls(uri=remote_path, format=format or GENERIC_FORMAT, **kwargs)

    @property
    def val(self) -> Optional[DF]:
        return self._raw_df

    @property
    def metadata(self) -> Optional[literals_pb2.StructuredDatasetMetadata]:
        return self._metadata

    @property
    def literal(self) -> Optional[literals_pb2.StructuredDataset]:
        return self._literal_sd

    def open(self, dataframe_type: Type[DF]):
        """
        Load the handler if needed. For the use case like:
        @task
        def t1(df: DataFrame):
          import pandas as pd
          df.open(pd.DataFrame).all()

        pandas is imported inside the task, so panda handler won't be loaded during deserialization in type engine.
        """
        from flyte.io._dataframe import lazy_import_dataframe_handler

        lazy_import_dataframe_handler()
        self._dataframe_type = dataframe_type
        return self

    async def all(self) -> DF:  # type: ignore
        if self._dataframe_type is None:
            raise ValueError("No dataframe type set. Use open() to set the local dataframe type you want to use.")

        if self.uri is not None and self.val is None:
            expected = TypeEngine.to_literal_type(DataFrame)
            await self._set_literal(expected)

        return await flyte_dataset_transformer.open_as(self.literal, self._dataframe_type, self.metadata)

    async def _set_literal(self, expected: types_pb2.LiteralType) -> None:
        """
        Explicitly set the DataFrame Literal to handle the following cases:

        1. Read the content from a DataFrame with an uri, for example:

        @task
        def return_df() -> DataFrame:
            df = DataFrame(uri="s3://my-s3-bucket/s3_flyte_dir/df.parquet", format="parquet")
            df = df.open(pd.DataFrame).all()
            return df

        For details, please refer to this issue: https://github.com/flyteorg/flyte/issues/5954.

        2. Need access to self._literal_sd when converting task output LiteralMap back to flyteidl, please see:
        https://github.com/flyteorg/flytekit/blob/f938661ff8413219d1bea77f6914a58c302d5c6c/flytekit/bin/entrypoint.py#L326

        For details, please refer to this issue: https://github.com/flyteorg/flyte/issues/5956.
        """
        to_literal = await flyte_dataset_transformer.to_literal(self, DataFrame, expected)
        self._literal_sd = to_literal.scalar.structured_dataset
        if self.metadata is None:
            self._metadata = self._literal_sd.metadata

    async def set_literal(self, expected: types_pb2.LiteralType) -> None:
        """
        A public wrapper method to set the DataFrame Literal.

        This method provides external access to the internal _set_literal method.
        """
        return await self._set_literal(expected)

    async def iter(self) -> typing.AsyncIterator[DF]:
        if self._dataframe_type is None:
            raise ValueError("No dataframe type set. Use open() to set the local dataframe type you want to use.")
        return await flyte_dataset_transformer.iter_as(
            self.literal, self._dataframe_type, updated_metadata=self.metadata
        )


# flat the nested column map recursively
def flatten_dict(sub_dict: dict, parent_key: str = "") -> typing.Dict:
    result = {}
    for key, value in sub_dict.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            result.update(flatten_dict(sub_dict=value, parent_key=current_key))
        elif is_dataclass(value):
            fields = getattr(value, "__dataclass_fields__")
            d = {k: v.type for k, v in fields.items()}
            result.update(flatten_dict(sub_dict=d, parent_key=current_key))
        elif hasattr(value, "model_fields"):  # Pydantic model
            d = {k: v.annotation for k, v in value.model_fields.items()}
            result.update(flatten_dict(sub_dict=d, parent_key=current_key))
        else:
            result[current_key] = value
    return result


def extract_cols_and_format(
    t: typing.Any,
) -> typing.Tuple[Type[T], Optional[typing.OrderedDict[str, Type]], Optional[str], Optional["pa.lib.Schema"]]:
    """
    Helper function, just used to iterate through Annotations and extract out the following information:
      - base type, if not Annotated, it will just be the type that was passed in.
      - column information, as a collections.OrderedDict,
      - the storage format, as a ``DataFrameFormat`` (str),
      - pa.lib.Schema

    If more than one of any type of thing is found, an error will be raised.
    If no instances of a given type are found, then None will be returned.

    If we add more things, we should put all the returned items in a dataclass instead of just a tuple.

    :param t: The incoming type which may or may not be Annotated
    :return: Tuple representing
        the original type,
        optional OrderedDict of columns,
        optional str for the format,
        optional pyarrow Schema
    """
    fmt = ""
    ordered_dict_cols = None
    pa_schema = None
    if get_origin(t) is Annotated:
        base_type, *annotate_args = get_args(t)
        for aa in annotate_args:
            if hasattr(aa, "__annotations__"):
                # handle dataclass argument
                d = collections.OrderedDict()
                d.update(aa.__annotations__)
                ordered_dict_cols = d
            elif isinstance(aa, dict):
                d = collections.OrderedDict()
                d.update(aa)
                ordered_dict_cols = d
            elif isinstance(aa, DataFrameFormat):
                if fmt != "":
                    raise ValueError(f"A format was already specified {fmt}, cannot use {aa}")
                fmt = aa
            elif isinstance(aa, collections.OrderedDict):
                if ordered_dict_cols is not None:
                    raise ValueError(f"Column information was already found {ordered_dict_cols}, cannot use {aa}")
                ordered_dict_cols = aa
            elif isinstance(aa, pa.lib.Schema):
                if pa_schema is not None:
                    raise ValueError(f"Arrow schema was already found {pa_schema}, cannot use {aa}")
                pa_schema = aa
        return base_type, ordered_dict_cols, fmt, pa_schema

    # We return None as the format instead of parquet or something because the transformer engine may find
    # a better default for the given dataframe type.
    return t, ordered_dict_cols, fmt, pa_schema


class DataFrameEncoder(ABC, Generic[T]):
    def __init__(
        self,
        python_type: Type[T],
        protocol: Optional[str] = None,
        supported_format: Optional[str] = None,
    ):
        """
        Extend this abstract class, implement the encode function, and register your concrete class with the
        DataFrameTransformerEngine class in order for the core flytekit type engine to handle
        dataframe libraries. This is the encoding interface, meaning it is used when there is a Python value that the
        flytekit type engine is trying to convert into a Flyte Literal. For the other way, see
        the DataFrameEncoder

        :param python_type: The dataframe class in question that you want to register this encoder with
        :param protocol: A prefix representing the storage driver (e.g. 's3, 'gs', 'bq', etc.). You can use either
          "s3" or "s3://". They are the same since the "://" will just be stripped by the constructor.
          If None, this encoder will be registered with all protocols that flytekit's data persistence layer
          is capable of handling.
        :param supported_format: Arbitrary string representing the format. If not supplied then an empty string
          will be used. An empty string implies that the encoder works with any format. If the format being asked
          for does not exist, the transformer engine will look for the "" encoder instead and write a warning.
        """
        self._python_type = python_type
        self._protocol = protocol.replace("://", "") if protocol else None
        self._supported_format = supported_format or ""

    @property
    def python_type(self) -> Type[T]:
        return self._python_type

    @property
    def protocol(self) -> Optional[str]:
        return self._protocol

    @property
    def supported_format(self) -> str:
        return self._supported_format

    @abstractmethod
    async def encode(
        self,
        dataframe: DataFrame,
        structured_dataset_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.StructuredDataset:
        """
        Even if the user code returns a plain dataframe instance, the dataset transformer engine will wrap the
        incoming dataframe with defaults set for that dataframe
        type. This simplifies this function's interface as a lot of data that could be specified by the user using
        the
        # TODO: Do we need to add a flag to indicate if it was wrapped by the transformer or by the user?

        :param dataframe: This is a DataFrame wrapper object. See more info above.
        :param structured_dataset_type: This the DataFrameType, as found in the LiteralType of the interface
          of the task that invoked this encoding call. It is passed along to encoders so that authors of encoders
          can include it in the returned literals.DataFrame. See the IDL for more information on why this
          literal in particular carries the type information along with it. If the encoder doesn't supply it, it will
          also be filled in after the encoder runs by the transformer engine.
        :return: This function should return a DataFrame literal object. Do not confuse this with the
          DataFrame wrapper class used as input to this function - that is the user facing Python class.
          This function needs to return the IDL DataFrame.
        """
        raise NotImplementedError


class DataFrameDecoder(ABC, Generic[DF]):
    def __init__(
        self,
        python_type: Type[DF],
        protocol: Optional[str] = None,
        supported_format: Optional[str] = None,
        additional_protocols: Optional[List[str]] = None,
    ):
        """
        Extend this abstract class, implement the decode function, and register your concrete class with the
        DataFrameTransformerEngine class in order for the core flytekit type engine to handle
        dataframe libraries. This is the decoder interface, meaning it is used when there is a Flyte Literal value,
        and we have to get a Python value out of it. For the other way, see the DataFrameEncoder

        :param python_type: The dataframe class in question that you want to register this decoder with
        :param protocol: A prefix representing the storage driver (e.g. 's3, 'gs', 'bq', etc.). You can use either
          "s3" or "s3://". They are the same since the "://" will just be stripped by the constructor.
          If None, this decoder will be registered with all protocols that flytekit's data persistence layer
          is capable of handling.
        :param supported_format: Arbitrary string representing the format. If not supplied then an empty string
          will be used. An empty string implies that the decoder works with any format. If the format being asked
          for does not exist, the transformer enginer will look for the "" decoder instead and write a warning.
        """
        self._python_type = python_type
        self._protocol = protocol.replace("://", "") if protocol else None
        self._supported_format = supported_format or ""

    @property
    def python_type(self) -> Type[DF]:
        return self._python_type

    @property
    def protocol(self) -> Optional[str]:
        return self._protocol

    @property
    def supported_format(self) -> str:
        return self._supported_format

    @abstractmethod
    async def decode(
        self,
        flyte_value: literals_pb2.StructuredDataset,
        current_task_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> Union[DF, typing.AsyncIterator[DF]]:
        """
        This is code that will be called by the dataset transformer engine to ultimately translate from a Flyte Literal
        value into a Python instance.

        :param flyte_value: This will be a Flyte IDL DataFrame Literal - do not confuse this with the
          DataFrame class defined also in this module.
        :param current_task_metadata: Metadata object containing the type (and columns if any) for the currently
           executing task. This type may have more or less information than the type information bundled
           inside the incoming flyte_value.
        :return: This function can either return an instance of the dataframe that this decoder handles, or an iterator
            of those dataframes.
        """
        raise NotImplementedError


def get_supported_types():
    import numpy as _np

    _SUPPORTED_TYPES: typing.Dict[Type, types_pb2.LiteralType] = {  # type: ignore
        _np.int32: types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER),
        _np.int64: types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER),
        _np.uint32: types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER),
        _np.uint64: types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER),
        int: types_pb2.LiteralType(simple=types_pb2.SimpleType.INTEGER),
        _np.float32: types_pb2.LiteralType(simple=types_pb2.SimpleType.FLOAT),
        _np.float64: types_pb2.LiteralType(simple=types_pb2.SimpleType.FLOAT),
        float: types_pb2.LiteralType(simple=types_pb2.SimpleType.FLOAT),
        _np.bool_: types_pb2.LiteralType(simple=types_pb2.SimpleType.BOOLEAN),  # type: ignore
        bool: types_pb2.LiteralType(simple=types_pb2.SimpleType.BOOLEAN),
        _np.datetime64: types_pb2.LiteralType(simple=types_pb2.SimpleType.DATETIME),
        _datetime.datetime: types_pb2.LiteralType(simple=types_pb2.SimpleType.DATETIME),
        _np.timedelta64: types_pb2.LiteralType(simple=types_pb2.SimpleType.DURATION),
        _datetime.timedelta: types_pb2.LiteralType(simple=types_pb2.SimpleType.DURATION),
        _np.bytes_: types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING),
        _np.str_: types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING),
        _np.object_: types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING),
        str: types_pb2.LiteralType(simple=types_pb2.SimpleType.STRING),
    }
    return _SUPPORTED_TYPES


class DuplicateHandlerError(ValueError): ...


class DataFrameTransformerEngine(TypeTransformer[DataFrame]):
    """
    Think of this transformer as a higher-level meta transformer that is used for all the dataframe types.
    If you are bringing a custom data frame type, or any data frame type, to flytekit, instead of
    registering with the main type engine, you should register with this transformer instead.
    """

    ENCODERS: ClassVar[Dict[Type, Dict[str, Dict[str, DataFrameEncoder]]]] = {}
    DECODERS: ClassVar[Dict[Type, Dict[str, Dict[str, DataFrameDecoder]]]] = {}
    DEFAULT_PROTOCOLS: ClassVar[Dict[Type, str]] = {}
    DEFAULT_FORMATS: ClassVar[Dict[Type, str]] = {}

    Handlers = Union[DataFrameEncoder, DataFrameDecoder]
    Renderers: ClassVar[Dict[Type, Renderable]] = {}

    @classmethod
    def _finder(cls, handler_map, df_type: Type, protocol: str, format: str):
        # If there's an exact match, then we should use it.
        try:
            return handler_map[df_type][protocol][format]
        except KeyError:
            ...

        fsspec_handler = None
        protocol_specific_handler = None
        single_handler = None
        default_format = cls.DEFAULT_FORMATS.get(df_type, None)

        try:
            fss_handlers = handler_map[df_type]["fsspec"]
            if format in fss_handlers:
                fsspec_handler = fss_handlers[format]
            elif GENERIC_FORMAT in fss_handlers:
                fsspec_handler = fss_handlers[GENERIC_FORMAT]
            else:
                if default_format and default_format in fss_handlers and format == GENERIC_FORMAT:
                    fsspec_handler = fss_handlers[default_format]
                else:
                    if len(fss_handlers) == 1 and format == GENERIC_FORMAT:
                        single_handler = next(iter(fss_handlers.values()))
                    else:
                        ...
        except KeyError:
            ...

        try:
            protocol_handlers = handler_map[df_type][protocol]
            if GENERIC_FORMAT in protocol_handlers:
                protocol_specific_handler = protocol_handlers[GENERIC_FORMAT]
            else:
                if default_format and default_format in protocol_handlers:
                    protocol_specific_handler = protocol_handlers[default_format]
                else:
                    if len(protocol_handlers) == 1:
                        single_handler = next(iter(protocol_handlers.values()))
                    else:
                        ...

        except KeyError:
            ...

        if protocol_specific_handler or fsspec_handler or single_handler:
            return protocol_specific_handler or fsspec_handler or single_handler
        else:
            raise ValueError(f"Failed to find a handler for {df_type}, protocol [{protocol}], fmt ['{format}']")

    @classmethod
    def get_encoder(cls, df_type: Type, protocol: str, format: str):
        return cls._finder(DataFrameTransformerEngine.ENCODERS, df_type, protocol, format)

    @classmethod
    def get_decoder(cls, df_type: Type, protocol: str, format: str) -> DataFrameDecoder:
        return cls._finder(DataFrameTransformerEngine.DECODERS, df_type, protocol, format)

    @classmethod
    def _handler_finder(cls, h: Handlers, protocol: str) -> Dict[str, Handlers]:
        if isinstance(h, DataFrameEncoder):
            top_level = cls.ENCODERS
        elif isinstance(h, DataFrameDecoder):
            top_level = cls.DECODERS  # type: ignore
        else:
            raise TypeError(f"We don't support this type of handler {h}")
        if h.python_type not in top_level:
            top_level[h.python_type] = {}
        if protocol not in top_level[h.python_type]:
            top_level[h.python_type][protocol] = {}
        return top_level[h.python_type][protocol]  # type: ignore

    def __init__(self):
        super().__init__("DataFrame Transformer", DataFrame)
        self._type_assertions_enabled = False

    @classmethod
    def register_renderer(cls, python_type: Type, renderer: Renderable):
        cls.Renderers[python_type] = renderer

    @classmethod
    def register(
        cls,
        h: Handlers,
        default_for_type: bool = False,
        override: bool = False,
        default_format_for_type: bool = False,
        default_storage_for_type: bool = False,
    ):
        """
        Call this with any Encoder or Decoder to register it with the flytekit type system. If your handler does not
        specify a protocol (e.g. s3, gs, etc.) field, then

        :param h: The DataFrameEncoder or DataFrameDecoder you wish to register with this transformer.
        :param default_for_type: If set, when a user returns from a task an instance of the dataframe the handler
          handles, e.g. ``return pd.DataFrame(...)``, not wrapped around the ``StructuredDataset`` object, we will
          use this handler's protocol and format as the default, effectively saying that this handler will be called.
          Note that this shouldn't be set if your handler's protocol is None, because that implies that your handler
          is capable of handling all the different storage protocols that flytekit's data persistence layer is aware of.
          In these cases, the protocol is determined by the raw output data prefix set in the active context.
        :param override: Override any previous registrations. If default_for_type is also set, this will also override
          the default.
        :param default_format_for_type: Unlike the default_for_type arg that will set this handler's format and storage
          as the default, this will only set the format. Error if already set, unless override is specified.
        :param default_storage_for_type: Same as above but only for the storage format. Error if already set,
          unless override is specified.
        """
        if not (isinstance(h, DataFrameEncoder) or isinstance(h, DataFrameDecoder)):
            raise TypeError(f"We don't support this type of handler {h}")

        if h.protocol is None:
            if default_for_type:
                raise ValueError(f"Registering SD handler {h} with all protocols should never have default specified.")
            try:
                cls.register_for_protocol(
                    h, "fsspec", False, override, default_format_for_type, default_storage_for_type
                )
            except DuplicateHandlerError:
                ...

        elif h.protocol == "":
            raise ValueError(f"Use None instead of empty string for registering handler {h}")
        else:
            cls.register_for_protocol(
                h, h.protocol, default_for_type, override, default_format_for_type, default_storage_for_type
            )

    @classmethod
    def register_for_protocol(
        cls,
        h: Handlers,
        protocol: str,
        default_for_type: bool,
        override: bool,
        default_format_for_type: bool,
        default_storage_for_type: bool,
    ):
        """
        See the main register function instead.
        """
        if protocol == "/":
            protocol = "file"
        lowest_level = cls._handler_finder(h, protocol)
        if h.supported_format in lowest_level and override is False:
            raise DuplicateHandlerError(
                f"Already registered a handler for {(h.python_type, protocol, h.supported_format)}"
            )
        lowest_level[h.supported_format] = h
        logger.debug(
            f"Registered {h.__class__.__name__} as handler for {h.python_type.__class__.__name__},"
            f" protocol {protocol}, fmt {h.supported_format}"
        )

        if (default_format_for_type or default_for_type) and h.supported_format != GENERIC_FORMAT:
            if h.python_type in cls.DEFAULT_FORMATS and not override:
                if cls.DEFAULT_FORMATS[h.python_type] != h.supported_format:
                    logger.info(
                        f"Not using handler {h.__class__.__name__} with format {h.supported_format}"
                        f" as default for {h.python_type.__class__.__name__},"
                        f" {cls.DEFAULT_FORMATS[h.python_type]} already specified."
                    )
            else:
                logger.debug(f"Use {type(h).__name__} as default handler for {h.python_type.__class__.__name__}.")
                cls.DEFAULT_FORMATS[h.python_type] = h.supported_format
        if default_storage_for_type or default_for_type:
            if h.protocol in cls.DEFAULT_PROTOCOLS and not override:
                logger.debug(
                    f"Not using handler {h} with storage protocol {h.protocol}"
                    f" as default for {h.python_type}, {cls.DEFAULT_PROTOCOLS[h.python_type]} already specified."
                )
            else:
                logger.debug(f"Using storage {protocol} for dataframes of type {h.python_type} from handler {h}")
                cls.DEFAULT_PROTOCOLS[h.python_type] = protocol

        # Register with the type engine as well
        # The semantics as of now are such that it doesn't matter which order these transformers are loaded in, as
        # long as the older Pandas/FlyteSchema transformer do not also specify the override
        engine = DataFrameTransformerEngine()
        TypeEngine.register_additional_type(engine, h.python_type, override=True)

    def assert_type(self, t: Type[DataFrame], v: typing.Any):
        return

    async def to_literal(
        self,
        python_val: Union[DataFrame, typing.Any],
        python_type: Union[Type[DataFrame], Type],
        expected: types_pb2.LiteralType,
    ) -> literals_pb2.Literal:
        # Make a copy in case we need to hand off to encoders, since we can't be sure of mutations.
        python_type, *_attrs = extract_cols_and_format(python_type)
        sdt = types_pb2.StructuredDatasetType(format=self.DEFAULT_FORMATS.get(python_type, GENERIC_FORMAT))

        if issubclass(python_type, DataFrame) and not isinstance(python_val, DataFrame):
            # Catch a common mistake
            raise TypeTransformerFailedError(
                f"Expected a DataFrame instance, but got {type(python_val)} instead."
                f" Did you forget to wrap your dataframe in a DataFrame instance?"
            )

        if expected and expected.structured_dataset_type:
            sdt = types_pb2.StructuredDatasetType(
                columns=expected.structured_dataset_type.columns,
                format=expected.structured_dataset_type.format,
                external_schema_type=expected.structured_dataset_type.external_schema_type,
                external_schema_bytes=expected.structured_dataset_type.external_schema_bytes,
            )

        # If the type signature has the DataFrame class, it will, or at least should, also be a
        # DataFrame instance.
        if isinstance(python_val, DataFrame):
            # There are three cases that we need to take care of here.

            # 1. A task returns a DataFrame that was just a passthrough input. If this happens
            # then return the original literals.DataFrame without invoking any encoder
            #
            # Ex.
            #   def t1(dataset: Annotated[DataFrame, my_cols]) -> Annotated[DataFrame, my_cols]:
            #       return dataset
            if python_val._literal_sd is not None:
                if python_val._already_uploaded:
                    return literals_pb2.Literal(scalar=literals_pb2.Scalar(structured_dataset=python_val._literal_sd))
                if python_val.val is not None:
                    raise ValueError(
                        f"Shouldn't have specified both literal {python_val._literal_sd} and dataframe {python_val.val}"
                    )
                return literals_pb2.Literal(scalar=literals_pb2.Scalar(structured_dataset=python_val._literal_sd))

            # 2. A task returns a python DataFrame with an uri.
            # Note: this case is also what happens we start a local execution of a task with a python DataFrame.
            #  It gets converted into a literal first, then back into a python DataFrame.
            #
            # Ex.
            #   def t2(uri: str) -> Annotated[DataFrame, my_cols]
            #       return DataFrame(uri=uri)
            if python_val.val is None:
                uri = python_val.uri
                format_val = python_val.format

                # Check the user-specified uri
                if not uri:
                    raise ValueError(f"If dataframe is not specified, then the uri should be specified. {python_val}")
                if not storage.is_remote(uri):
                    uri = await storage.put(uri, recursive=True)

                # Check the user-specified format
                # When users specify format for a DataFrame, the format should be retained
                # conditionally. For details, please refer to https://github.com/flyteorg/flyte/issues/6096.
                # Following illustrates why we can't always copy the user-specified file_format over:
                #
                # @task
                # def modify_format(df: Annotated[DataFrame, {}, "task-format"]) -> DataFrame:
                #     return df
                #
                # df = DataFrame(uri="s3://my-s3-bucket/df.parquet", format="user-format")
                # df2 = modify_format(df=df)
                #
                # In this case, we expect the df2.format to be task-format (as shown in Annotated),
                # not user-format. If we directly copy the user-specified format over,
                # the type hint information will be missing.
                if sdt.format == GENERIC_FORMAT and format_val != GENERIC_FORMAT:
                    sdt.format = format_val

                sd_model = literals_pb2.StructuredDataset(
                    uri=uri,
                    metadata=literals_pb2.StructuredDatasetMetadata(structured_dataset_type=sdt),
                )
                return literals_pb2.Literal(scalar=literals_pb2.Scalar(structured_dataset=sd_model))

            # 3. This is the third and probably most common case. The python DataFrame object wraps a dataframe
            # that we will need to invoke an encoder for. Figure out which encoder to call and invoke it.
            df_type = type(python_val.val)
            protocol = self._protocol_from_type_or_prefix(df_type, python_val.uri)

            return await self.encode(
                python_val,
                df_type,
                protocol,
                sdt.format,
                sdt,
            )

        # Otherwise assume it's a dataframe instance. Wrap it with some defaults
        fmt = self.DEFAULT_FORMATS.get(python_type, "")
        protocol = self._protocol_from_type_or_prefix(python_type)
        meta = literals_pb2.StructuredDatasetMetadata(
            structured_dataset_type=expected.structured_dataset_type if expected else None
        )

        fdf = DataFrame.from_df(val=python_val)
        fdf._metadata = meta
        return await self.encode(fdf, python_type, protocol, fmt, sdt)

    def _protocol_from_type_or_prefix(self, df_type: Type, uri: Optional[str] = None) -> str:
        """
        Get the protocol from the default, if missing, then look it up from the uri if provided, if not then look
        up from the provided context's file access.
        """
        if df_type in self.DEFAULT_PROTOCOLS:
            return self.DEFAULT_PROTOCOLS[df_type]
        else:
            from flyte._context import internal_ctx

            ctx = internal_ctx()
            protocol = get_protocol(uri or ctx.raw_data.path)
            logger.debug(
                f"No default protocol for type {df_type} found, using {protocol} from output prefix {ctx.raw_data.path}"
            )
            return protocol

    async def encode(
        self,
        df: DataFrame,
        df_type: Type,
        protocol: str,
        format: str,
        structured_literal_type: types_pb2.StructuredDatasetType,
    ) -> literals_pb2.Literal:
        handler: DataFrameEncoder
        handler = self.get_encoder(df_type, protocol, format)

        sd_model = await handler.encode(df, structured_literal_type)
        # This block is here in case the encoder did not set the type information in the metadata. Since this literal
        # is special in that it carries around the type itself, we want to make sure the type info therein is at
        # least as good as the type of the interface.

        if sd_model.metadata is None:
            sd_model.metadata = literals_pb2.StructuredDatasetMetadata(structured_dataset_type=structured_literal_type)
        if sd_model.metadata and sd_model.metadata.structured_dataset_type is None:
            sd_model.metadata.structured_dataset_type = structured_literal_type
        # Always set the format here to the format of the handler.
        # Note that this will always be the same as the incoming format except for when the fallback handler
        # with a format of "" is used.
        sd_model.metadata.structured_dataset_type.format = handler.supported_format
        lit = literals_pb2.Literal(scalar=literals_pb2.Scalar(structured_dataset=sd_model))

        # Because the handler.encode may have uploaded something, and because the sd may end up living inside a
        # dataclass, we need to modify any uploaded flyte:// urls here. Needed here even though the Type engine
        # already does this because the DataframeTransformerEngine may be called directly.
        modify_literal_uris(lit)
        df._literal_sd = sd_model
        df._already_uploaded = True
        return lit

    async def to_python_value(
        self, lv: literals_pb2.Literal, expected_python_type: Type[T] | DataFrame
    ) -> T | DataFrame:
        """
        The only tricky thing with converting a Literal (say the output of an earlier task), to a Python value at
        the start of a task execution, is the column subsetting behavior. For example, if you have,

        def t1() -> Annotated[StructuredDataset, kwtypes(col_a=int, col_b=float)]: ...
        def t2(in_a: Annotated[StructuredDataset, kwtypes(col_b=float)]): ...

        where t2(in_a=t1()), when t2 does in_a.open(pd.DataFrame).all(), it should get a DataFrame
        with only one column.

        +-----------------------------+-----------------------------------------+--------------------------------------+
        |                             |          StructuredDatasetType of the incoming Literal                         |
        +-----------------------------+-----------------------------------------+--------------------------------------+
        | StructuredDatasetType       | Has columns defined                     |  [] columns or None                  |
        | of currently running task   |                                         |                                      |
        +=============================+=========================================+======================================+
        |    Has columns              | The StructuredDatasetType passed to the decoder will have the columns          |
        |    defined                  | as defined by the type annotation of the currently running task.               |
        |                             |                                                                                |
        |                             | Decoders **should** then subset the incoming data to the columns requested.    |
        |                             |                                                                                |
        +-----------------------------+-----------------------------------------+--------------------------------------+
        |   [] columns or None        | StructuredDatasetType passed to decoder | StructuredDatasetType passed to the  |
        |                             | will have the columns from the incoming | decoder will have an empty list of   |
        |                             | Literal. This is the scenario where     | columns.                             |
        |                             | the Literal returned by the running     |                                      |
        |                             | task will have more information than    |                                      |
        |                             | the running task's signature.           |                                      |
        +-----------------------------+-----------------------------------------+--------------------------------------+
        """
        if lv.HasField("scalar") and lv.scalar.HasField("binary"):
            raise TypeTransformerFailedError("Attribute access unsupported.")

        # Detect annotations and extract out all the relevant information that the user might supply
        expected_python_type, column_dict, _storage_fmt, _pa_schema = extract_cols_and_format(expected_python_type)

        # Start handling for DataFrame scalars, first look at the columns
        incoming_columns = lv.scalar.structured_dataset.metadata.structured_dataset_type.columns

        # If the incoming literal, also doesn't have columns, then we just have an empty list, so initialize here
        final_dataset_columns = []
        # If the current running task's input does not have columns defined, or has an empty list of columns
        if column_dict is None or len(column_dict) == 0:
            # but if it does, then we just copy it over
            if incoming_columns is not None and incoming_columns != []:
                final_dataset_columns = incoming_columns[:]
        # If the current running task's input does have columns defined
        else:
            final_dataset_columns = self._convert_ordered_dict_of_columns_to_list(column_dict)

        new_sdt = types_pb2.StructuredDatasetType(
            columns=final_dataset_columns,
            format=lv.scalar.structured_dataset.metadata.structured_dataset_type.format,
            external_schema_type=lv.scalar.structured_dataset.metadata.structured_dataset_type.external_schema_type,
            external_schema_bytes=lv.scalar.structured_dataset.metadata.structured_dataset_type.external_schema_bytes,
        )
        metad = literals_pb2.StructuredDatasetMetadata(structured_dataset_type=new_sdt)

        # A DataFrame type, for example
        #   t1(input_a: DataFrame)  # or
        #   t1(input_a: Annotated[DataFrame, my_cols])
        if issubclass(expected_python_type, DataFrame):
            fdf = DataFrame(format=metad.structured_dataset_type.format, uri=lv.scalar.structured_dataset.uri)
            fdf._already_uploaded = True
            fdf._literal_sd = lv.scalar.structured_dataset
            fdf._metadata = metad
            return fdf

        # If the requested type was not a flyte.DataFrame, then it means it was a raw dataframe type, which means
        # we should do the opening/downloading and whatever else it might entail right now. No iteration option here.
        return await self.open_as(lv.scalar.structured_dataset, df_type=expected_python_type, updated_metadata=metad)

    def to_html(self, python_val: typing.Any, expected_python_type: Type[T]) -> str:
        if isinstance(python_val, DataFrame):
            if python_val.val is not None:
                df = python_val.val
            else:
                # Here we only render column information by default instead of opening the structured dataset.
                col = typing.cast(DataFrame, python_val).columns()
                dataframe = pd.DataFrame(col, ["column type"])
                return dataframe.to_html()  # type: ignore
        else:
            df = python_val

        if type(df) in self.Renderers:
            return self.Renderers[type(df)].to_html(df)
        else:
            raise NotImplementedError(f"Could not find a renderer for {type(df)} in {self.Renderers}")

    async def open_as(
        self,
        sd: literals_pb2.StructuredDataset,
        df_type: Type[DF],
        updated_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> DF:
        """
        :param sd:
        :param df_type:
        :param updated_metadata: New metadata type, since it might be different from the metadata in the literal.
        :return: dataframe. It could be pandas dataframe or arrow table, etc.
        """
        protocol = get_protocol(sd.uri)
        decoder = self.get_decoder(df_type, protocol, sd.metadata.structured_dataset_type.format)
        result = await decoder.decode(sd, updated_metadata)
        return typing.cast(DF, result)

    async def iter_as(
        self,
        sd: literals_pb2.StructuredDataset,
        df_type: Type[DF],
        updated_metadata: literals_pb2.StructuredDatasetMetadata,
    ) -> typing.AsyncIterator[DF]:
        protocol = get_protocol(sd.uri)
        decoder = self.DECODERS[df_type][protocol][sd.metadata.structured_dataset_type.format]
        result: Union[Coroutine[Any, Any, DF], Coroutine[Any, Any, typing.AsyncIterator[DF]]] = decoder.decode(
            sd, updated_metadata
        )
        if not isinstance(result, types.AsyncGeneratorType):
            raise ValueError(f"Decoder {decoder} didn't return an async iterator {result} but should have from {sd}")
        return result

    def _get_dataset_column_literal_type(self, t: Type) -> types_pb2.LiteralType:
        if t in get_supported_types():
            return get_supported_types()[t]
        origin = getattr(t, "__origin__", None)
        if origin is list:
            return types_pb2.LiteralType(collection_type=self._get_dataset_column_literal_type(t.__args__[0]))
        if origin is dict:
            return types_pb2.LiteralType(map_value_type=self._get_dataset_column_literal_type(t.__args__[1]))
        raise AssertionError(f"type {t} is currently not supported by DataFrame")

    def _convert_ordered_dict_of_columns_to_list(
        self, column_map: typing.Optional[typing.OrderedDict[str, Type]]
    ) -> typing.List[types_pb2.StructuredDatasetType.DatasetColumn]:
        converted_cols: typing.List[types_pb2.StructuredDatasetType.DatasetColumn] = []
        if column_map is None or len(column_map) == 0:
            return converted_cols
        flat_column_map = flatten_dict(column_map)
        for k, v in flat_column_map.items():
            lt = self._get_dataset_column_literal_type(v)
            converted_cols.append(types_pb2.StructuredDatasetType.DatasetColumn(name=k, literal_type=lt))
        return converted_cols

    def _get_dataset_type(self, t: typing.Union[Type[DataFrame], typing.Any]) -> types_pb2.StructuredDatasetType:
        _original_python_type, column_map, storage_format, pa_schema = extract_cols_and_format(t)  # type: ignore

        # Get the column information
        converted_cols: typing.List[types_pb2.StructuredDatasetType.DatasetColumn] = (
            self._convert_ordered_dict_of_columns_to_list(column_map)
        )

        return types_pb2.StructuredDatasetType(
            columns=converted_cols,
            format=storage_format,
            external_schema_type="arrow" if pa_schema else None,
            external_schema_bytes=typing.cast(pa.lib.Schema, pa_schema).to_string().encode() if pa_schema else None,
        )

    def get_literal_type(self, t: typing.Union[Type[DataFrame], typing.Any]) -> types_pb2.LiteralType:
        """
        Provide a concrete implementation so that writers of custom dataframe handlers since there's nothing that
        special about the literal type. Any dataframe type will always be associated with the structured dataset type.
        The other aspects of it - columns, external schema type, etc. can be read from associated metadata.

        :param t: The python dataframe type, which is mostly ignored.
        """
        return types_pb2.LiteralType(structured_dataset_type=self._get_dataset_type(t))

    def guess_python_type(self, literal_type: types_pb2.LiteralType) -> Type[DataFrame]:
        # todo: technically we should return the dataframe type specified in the constructor, but to do that,
        #   we'd have to store that, which we don't do today. See possibly #1363
        if literal_type.HasField("structured_dataset_type"):
            return DataFrame
        raise ValueError(f"DataFrameTransformerEngine cannot reverse {literal_type}")


flyte_dataset_transformer = DataFrameTransformerEngine()
TypeEngine.register(flyte_dataset_transformer)
