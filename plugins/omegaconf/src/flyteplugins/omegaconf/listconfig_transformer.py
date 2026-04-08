from __future__ import annotations

from typing import Type

import msgpack
from flyteidl2.core.literals_pb2 import Binary, Literal, Scalar
from flyteidl2.core.types_pb2 import LiteralType, SimpleType, TypeAnnotation
from google.protobuf import struct_pb2
from omegaconf import ListConfig, OmegaConf

from flyte.types._type_engine import (
    CACHE_KEY_METADATA,
    MESSAGEPACK,
    SERIALIZATION_FORMAT,
    TypeTransformer,
)


class ListConfigTransformer(TypeTransformer[ListConfig]):
    def __init__(self):
        super().__init__("OmegaConf ListConfig Transformer", ListConfig)

    def get_literal_type(self, t: Type[ListConfig]) -> LiteralType:
        meta_struct = struct_pb2.Struct()
        meta_struct.update({CACHE_KEY_METADATA: {SERIALIZATION_FORMAT: MESSAGEPACK}})
        return LiteralType(
            simple=SimpleType.STRUCT,
            annotation=TypeAnnotation(annotations=meta_struct),
        )

    async def to_literal(
        self,
        python_val: ListConfig,
        python_type: Type[ListConfig],
        expected: LiteralType,
    ) -> Literal:
        values = OmegaConf.to_container(python_val, resolve=True)
        msgpack_bytes = msgpack.dumps(values)
        return Literal(
            scalar=Scalar(binary=Binary(value=msgpack_bytes, tag=MESSAGEPACK))
        )

    def from_binary_idl(
        self, binary_idl_object: Binary, expected_python_type: Type[ListConfig]
    ) -> ListConfig:
        if binary_idl_object.tag != MESSAGEPACK:
            raise TypeError(f"Unsupported binary format: `{binary_idl_object.tag}`")
        values = msgpack.loads(binary_idl_object.value, strict_map_key=False)
        return OmegaConf.create(values)

    async def to_python_value(
        self, lv: Literal, expected_python_type: Type[ListConfig]
    ) -> ListConfig:
        if lv and lv.HasField("scalar") and lv.scalar.HasField("binary"):
            return self.from_binary_idl(lv.scalar.binary, expected_python_type)
        raise TypeError(f"Cannot convert literal to ListConfig: {lv}")
