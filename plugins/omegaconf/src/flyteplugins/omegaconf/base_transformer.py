from __future__ import annotations

from typing import Generic, Type, TypeVar

import msgpack
from flyte.types._type_engine import (
    CACHE_KEY_METADATA,
    MESSAGEPACK,
    SERIALIZATION_FORMAT,
    TypeTransformer,
)
from flyteidl2.core.literals_pb2 import Binary, Literal, Scalar
from flyteidl2.core.types_pb2 import LiteralType, SimpleType, TypeAnnotation
from google.protobuf import struct_pb2

from omegaconf import DictConfig, ListConfig

from .codec import deserialize_omegaconf, serialize_omegaconf

T = TypeVar("T", DictConfig, ListConfig)


class OmegaConfTransformerBase(TypeTransformer[T], Generic[T]):
    def __init__(self, name: str, container_type: Type[T]):
        super().__init__(name, container_type)
        self._container_type = container_type

    def get_literal_type(self, t: Type[T]) -> LiteralType:
        meta_struct = struct_pb2.Struct()
        meta_struct.update({CACHE_KEY_METADATA: {SERIALIZATION_FORMAT: MESSAGEPACK}})
        return LiteralType(
            simple=SimpleType.STRUCT,
            annotation=TypeAnnotation(annotations=meta_struct),
        )

    async def to_literal(
        self,
        python_val: T,
        python_type: Type[T],
        expected: LiteralType,
    ) -> Literal:
        payload = serialize_omegaconf(python_val)
        msgpack_bytes = msgpack.dumps(payload)
        return Literal(scalar=Scalar(binary=Binary(value=msgpack_bytes, tag=MESSAGEPACK)))

    def from_binary_idl(self, binary_idl_object: Binary, expected_python_type: Type[T]) -> T:
        if binary_idl_object.tag != MESSAGEPACK:
            raise TypeError(f"Unsupported binary format: `{binary_idl_object.tag}`")
        payload = msgpack.loads(binary_idl_object.value, strict_map_key=False)
        config = deserialize_omegaconf(payload)
        if not isinstance(config, self._container_type):
            raise TypeError(f"Expected {self._container_type.__name__} payload, got {type(config).__name__}")
        return config

    async def to_python_value(self, lv: Literal, expected_python_type: Type[T]) -> T:
        if lv and lv.HasField("scalar") and lv.scalar.HasField("binary"):
            return self.from_binary_idl(lv.scalar.binary, expected_python_type)
        raise TypeError(f"Cannot convert literal to {self._container_type.__name__}: {lv}")


class DictConfigTransformer(OmegaConfTransformerBase[DictConfig]):
    def __init__(self):
        super().__init__("OmegaConf DictConfig Transformer", DictConfig)


class ListConfigTransformer(OmegaConfTransformerBase[ListConfig]):
    def __init__(self):
        super().__init__("OmegaConf ListConfig Transformer", ListConfig)
