from __future__ import annotations

import importlib
from enum import Enum
from pathlib import Path, PurePath
from typing import Any

from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

PAYLOAD_MARKER = "__flyte_omegaconf__"
PAYLOAD_KIND = "kind"
PAYLOAD_VALUES = "values"
PAYLOAD_SCHEMA = "schema"
PAYLOAD_TYPE = "type"
PAYLOAD_NAME = "name"
PAYLOAD_VALUE = "value"

KIND_DICT = "dict"
KIND_LIST = "list"
KIND_MISSING = "missing"
KIND_ENUM = "enum"
KIND_PATH = "path"
KIND_TUPLE = "tuple"


def serialize_omegaconf(node: DictConfig | ListConfig) -> dict[str, Any]:
    return _serialize_value(node)


def deserialize_omegaconf(payload: dict[str, Any]) -> DictConfig | ListConfig:
    config = _deserialize_value(payload)
    if isinstance(config, (DictConfig, ListConfig)):
        return config
    raise TypeError(f"Expected OmegaConf container payload, got {type(config).__name__}")


def _serialize_value(value: Any) -> Any:
    if isinstance(value, DictConfig):
        payload: dict[str, Any] = {
            PAYLOAD_MARKER: True,
            PAYLOAD_KIND: KIND_DICT,
            PAYLOAD_VALUES: {},
        }
        schema_name = _schema_name(value)
        if schema_name is not None:
            payload[PAYLOAD_SCHEMA] = schema_name

        for key in value.keys():
            child_node = value._get_node(key)
            payload[PAYLOAD_VALUES][key] = _serialize_child(value, key, child_node)
        return payload

    if isinstance(value, ListConfig):
        payload = {
            PAYLOAD_MARKER: True,
            PAYLOAD_KIND: KIND_LIST,
            PAYLOAD_VALUES: [],
        }
        for index in range(len(value)):
            child_node = value._get_node(index)
            payload[PAYLOAD_VALUES].append(_serialize_child(value, index, child_node))
        return payload

    if isinstance(value, Enum):
        return {
            PAYLOAD_MARKER: True,
            PAYLOAD_KIND: KIND_ENUM,
            PAYLOAD_TYPE: _qualified_name(type(value)),
            PAYLOAD_NAME: value.name,
            PAYLOAD_VALUE: value.value,
        }

    if isinstance(value, PurePath):
        return {
            PAYLOAD_MARKER: True,
            PAYLOAD_KIND: KIND_PATH,
            PAYLOAD_VALUE: str(value),
        }

    if isinstance(value, tuple):
        return {
            PAYLOAD_MARKER: True,
            PAYLOAD_KIND: KIND_TUPLE,
            PAYLOAD_VALUES: [_serialize_value(item) for item in value],
        }

    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    return value


def _serialize_child(parent: DictConfig | ListConfig, key: str | int, child_node: Any) -> Any:
    if child_node._is_missing():
        return {
            PAYLOAD_MARKER: True,
            PAYLOAD_KIND: KIND_MISSING,
        }
    return _serialize_value(parent[key])


def _deserialize_value(payload: Any) -> Any:
    if isinstance(payload, list):
        return [_deserialize_value(item) for item in payload]

    if not isinstance(payload, dict):
        return payload

    if payload.get(PAYLOAD_MARKER) is not True:
        return {key: _deserialize_value(value) for key, value in payload.items()}

    kind = payload[PAYLOAD_KIND]

    if kind == KIND_MISSING:
        return MISSING

    if kind == KIND_ENUM:
        enum_value = payload[PAYLOAD_VALUE]
        enum_name = payload[PAYLOAD_NAME]
        enum_type_name = payload[PAYLOAD_TYPE]
        try:
            enum_type = _import_class(enum_type_name)
            return enum_type[enum_name]
        except (ImportError, AttributeError, ModuleNotFoundError, KeyError, ValueError):
            return enum_value

    if kind == KIND_PATH:
        return Path(payload[PAYLOAD_VALUE])

    if kind == KIND_TUPLE:
        return tuple(_deserialize_value(item) for item in payload[PAYLOAD_VALUES])

    if kind == KIND_LIST:
        values = [_deserialize_value(item) for item in payload[PAYLOAD_VALUES]]
        return OmegaConf.create(values)

    if kind == KIND_DICT:
        values = {key: _deserialize_value(value) for key, value in payload[PAYLOAD_VALUES].items()}
        schema_name = payload.get(PAYLOAD_SCHEMA)
        if schema_name is None:
            return OmegaConf.create(values)

        try:
            schema_type = _import_class(schema_name)
            return OmegaConf.merge(OmegaConf.structured(schema_type), values)
        except (ImportError, AttributeError, ModuleNotFoundError):
            return OmegaConf.create(values)

    raise TypeError(f"Unsupported OmegaConf payload kind: {kind}")


def _schema_name(config: DictConfig) -> str | None:
    schema_type = OmegaConf.get_type(config)
    if schema_type in (None, dict):
        return None
    return _qualified_name(schema_type)


def _qualified_name(value_type: type) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _import_class(fully_qualified_name: str) -> type:
    module_name, class_name = fully_qualified_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
