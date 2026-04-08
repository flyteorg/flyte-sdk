from __future__ import annotations

import importlib
from typing import Type

import msgpack
from flyteidl2.core.literals_pb2 import Binary, Literal, Scalar
from flyteidl2.core.types_pb2 import LiteralType, SimpleType, TypeAnnotation
from google.protobuf import struct_pb2
from omegaconf import DictConfig, OmegaConf

from flyte.types._type_engine import (
    CACHE_KEY_METADATA,
    MESSAGEPACK,
    SERIALIZATION_FORMAT,
    TypeTransformer,
)


class DictConfigTransformer(TypeTransformer[DictConfig]):
    def __init__(self):
        super().__init__("OmegaConf DictConfig Transformer", DictConfig)

    def get_literal_type(self, t: Type[DictConfig]) -> LiteralType:
        meta_struct = struct_pb2.Struct()
        meta_struct.update({CACHE_KEY_METADATA: {SERIALIZATION_FORMAT: MESSAGEPACK}})
        return LiteralType(
            simple=SimpleType.STRUCT,
            annotation=TypeAnnotation(annotations=meta_struct),
        )

    async def to_literal(
        self,
        python_val: DictConfig,
        python_type: Type[DictConfig],
        expected: LiteralType,
    ) -> Literal:
        base_type = OmegaConf.get_type(python_val)
        base_dataclass = (
            f"{base_type.__module__}.{base_type.__qualname__}"
            if base_type is not dict
            else "builtins.dict"
        )
        payload = {
            "base_dataclass": base_dataclass,
            "values": OmegaConf.to_container(python_val, resolve=True),
        }
        msgpack_bytes = msgpack.dumps(payload)
        return Literal(
            scalar=Scalar(binary=Binary(value=msgpack_bytes, tag=MESSAGEPACK))
        )

    def from_binary_idl(
        self, binary_idl_object: Binary, expected_python_type: Type[DictConfig]
    ) -> DictConfig:
        if binary_idl_object.tag != MESSAGEPACK:
            raise TypeError(f"Unsupported binary format: `{binary_idl_object.tag}`")
        payload = msgpack.loads(binary_idl_object.value, strict_map_key=False)
        return _reconstruct_dictconfig(payload)

    async def to_python_value(
        self, lv: Literal, expected_python_type: Type[DictConfig]
    ) -> DictConfig:
        if lv and lv.HasField("scalar") and lv.scalar.HasField("binary"):
            return self.from_binary_idl(lv.scalar.binary, expected_python_type)
        raise TypeError(f"Cannot convert literal to DictConfig: {lv}")


def _reconstruct_dictconfig(payload: dict) -> DictConfig:
    """Reconstruct a DictConfig from the deserialized msgpack payload.

    Always uses Auto mode: tries to reconstruct the original dataclass-backed
    DictConfig if the class is importable, falls back to plain DictConfig otherwise.
    """
    base_dataclass_name = payload["base_dataclass"]
    values = payload["values"]

    if base_dataclass_name == "builtins.dict":
        return OmegaConf.create(values)

    try:
        base_cls = _import_class(base_dataclass_name)
        return OmegaConf.merge(OmegaConf.structured(base_cls), values)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return OmegaConf.create(values)


def _import_class(fully_qualified_name: str) -> type:
    """Import a class from its fully qualified name (module.path.ClassName)."""
    module_name, class_name = fully_qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
