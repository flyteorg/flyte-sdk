import hashlib
import os
import typing
from typing import Type

import aiofiles
import cloudpickle
from flyteidl.core import literals_pb2, types_pb2

import flyte.storage as storage

from ._type_engine import TypeEngine, TypeTransformer

T = typing.TypeVar("T")


class FlytePickle(typing.Generic[T]):
    """
    This type is only used by flytekit internally. User should not use this type.
    Any type that flyte can't recognize will become FlytePickle
    """

    @classmethod
    def python_type(cls) -> typing.Type:
        return type(None)

    @classmethod
    def __class_getitem__(cls, python_type: typing.Type) -> typing.Type:
        if python_type is None:
            return cls

        class _SpecificFormatClass(FlytePickle):
            # Get the type engine to see this as kind of a generic
            __origin__ = FlytePickle

            @classmethod
            def python_type(cls) -> typing.Type:
                return python_type

        return _SpecificFormatClass

    @classmethod
    async def to_pickle(cls, python_val: typing.Any) -> str:
        h = hashlib.md5()
        str_bytes = cloudpickle.dumps(python_val)
        h.update(str_bytes)

        uri = storage.get_random_local_path(file_path_or_file_name=h.hexdigest())
        os.makedirs(os.path.dirname(uri), exist_ok=True)
        async with aiofiles.open(uri, "w+b") as outfile:
            await outfile.write(str_bytes)

        return await storage.put(str(uri))

    @classmethod
    async def from_pickle(cls, uri: str) -> typing.Any:
        # Deserialize the pickle, and return data in the pickle,
        # and download pickle file to local first if file is not in the local file systems.
        if storage.is_remote(uri):
            local_path = storage.get_random_local_path()
            await storage.get(uri, str(local_path), False)
            uri = str(local_path)
        async with aiofiles.open(uri, "rb") as infile:
            data = cloudpickle.loads(await infile.read())
        return data


class FlytePickleTransformer(TypeTransformer[FlytePickle]):
    PYTHON_PICKLE_FORMAT = "PythonPickle"

    def __init__(self):
        super().__init__(name="FlytePickle", t=FlytePickle)

    def assert_type(self, t: Type[T], v: T):
        # Every type can serialize to pickle, so we don't need to check the type here.
        ...

    async def to_python_value(self, lv: literals_pb2.Literal, expected_python_type: Type[T]) -> T:
        uri = lv.scalar.blob.uri
        return await FlytePickle.from_pickle(uri)

    async def to_literal(
        self,
        python_val: T,
        python_type: Type[T],
        expected: types_pb2.LiteralType,
    ) -> literals_pb2.Literal:
        if python_val is None:
            raise AssertionError("Cannot pickle None Value.")
        meta = literals_pb2.BlobMetadata(
            type=types_pb2.BlobType(
                format=self.PYTHON_PICKLE_FORMAT, dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE
            )
        )
        remote_path = await FlytePickle.to_pickle(python_val)
        return literals_pb2.Literal(scalar=literals_pb2.Scalar(blob=literals_pb2.Blob(metadata=meta, uri=remote_path)))

    def guess_python_type(self, literal_type: types_pb2.LiteralType) -> typing.Type[FlytePickle[typing.Any]]:
        if (
            literal_type.blob is not None
            and literal_type.blob.dimensionality == types_pb2.BlobType.BlobDimensionality.SINGLE
            and literal_type.blob.format == FlytePickleTransformer.PYTHON_PICKLE_FORMAT
        ):
            return FlytePickle

        raise ValueError(f"Transformer {self} cannot reverse {literal_type}")

    def get_literal_type(self, t: Type[T]) -> types_pb2.LiteralType:
        lt = types_pb2.LiteralType(
            blob=types_pb2.BlobType(
                format=self.PYTHON_PICKLE_FORMAT, dimensionality=types_pb2.BlobType.BlobDimensionality.SINGLE
            )
        )
        lt.metadata = {"python_class_name": str(t)}
        return lt


TypeEngine.register(FlytePickleTransformer())
