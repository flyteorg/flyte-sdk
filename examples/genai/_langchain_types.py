"""Type transformers for LangChain types."""

from __future__ import annotations

import json
from typing import Type

import msgpack
from flyteidl2.core.literals_pb2 import Binary, Literal, Scalar
from flyteidl2.core.types_pb2 import LiteralType, SimpleType, TypeAnnotation
from google.protobuf import json_format as _json_format
from google.protobuf import struct_pb2

from flyte.types._type_engine import (
    CACHE_KEY_METADATA,
    MESSAGEPACK,
    SERIALIZATION_FORMAT,
    TypeEngine,
    TypeTransformer,
    TypeTransformerFailedError,
)

try:
    from langchain_core.load import dumps, loads
    from langchain_core.messages.base import BaseMessage

    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    BaseMessage = None  # type: ignore


class BaseMessageTransformer(TypeTransformer["BaseMessage"]):
    """Type transformer for LangChain BaseMessage and its subclasses.

    This transformer handles serialization and deserialization of LangChain
    messages (HumanMessage, AIMessage, SystemMessage, etc.) using LangChain's
    native serialization format.

    The transformer uses LangChain's `dumps()` and `loads()` functions which:
    - Preserve the exact message type (polymorphic serialization)
    - Handle all message metadata and additional kwargs
    - Support the full LangChain message hierarchy

    Example:
        >>> from langchain_core.messages import HumanMessage, AIMessage
        >>>
        >>> @task
        >>> def process_message(msg: BaseMessage) -> BaseMessage:
        ...     # msg could be HumanMessage, AIMessage, etc.
        ...     return AIMessage(content=f"Processed: {msg.content}")
    """

    def __init__(self):
        if not _LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required for BaseMessageTransformer. "
                "Install it with: pip install langchain-core"
            )
        super().__init__("BaseMessage Transformer", BaseMessage, enable_type_assertions=False)

    def get_literal_type(self, t: Type[BaseMessage]) -> LiteralType:
        """Get the Flyte LiteralType for a BaseMessage type.

        Args:
            t: The BaseMessage type or subclass.

        Returns:
            A LiteralType with STRUCT simple type and appropriate metadata.
        """
        # Get JSON schema from the message type
        schema = t.model_json_schema()

        meta_struct = struct_pb2.Struct()
        meta_struct.update(
            {
                CACHE_KEY_METADATA: {
                    SERIALIZATION_FORMAT: MESSAGEPACK,
                }
            }
        )

        return LiteralType(
            simple=SimpleType.STRUCT,
            metadata=schema,
            annotation=TypeAnnotation(annotations=meta_struct),
        )

    async def to_literal(
        self,
        python_val: BaseMessage,
        python_type: Type[BaseMessage],
        expected: LiteralType,
    ) -> Literal:
        """Convert a BaseMessage to a Flyte Literal.

        Uses LangChain's `dumps()` function for serialization which preserves
        the exact message type and all metadata.

        Args:
            python_val: The BaseMessage instance to convert.
            python_type: The expected Python type.
            expected: The expected Flyte LiteralType.

        Returns:
            A Flyte Literal containing the serialized message.
        """
        # Use LangChain's serialization which preserves the message type
        json_str = dumps(python_val)
        dict_obj = json.loads(json_str)
        msgpack_bytes = msgpack.dumps(dict_obj)
        return Literal(scalar=Scalar(binary=Binary(value=msgpack_bytes, tag=MESSAGEPACK)))

    def from_binary_idl(self, binary_idl_object: Binary, expected_python_type: Type[BaseMessage]) -> BaseMessage:
        """Deserialize a BaseMessage from binary IDL format.

        Args:
            binary_idl_object: The binary IDL object containing msgpack data.
            expected_python_type: The expected Python type (BaseMessage or subclass).

        Returns:
            The deserialized BaseMessage instance.

        Raises:
            TypeTransformerFailedError: If the binary format is unsupported.
        """
        if binary_idl_object.tag == MESSAGEPACK:
            dict_obj = msgpack.loads(binary_idl_object.value, strict_map_key=False)
            json_str = json.dumps(dict_obj)
            # Use LangChain's loads() which handles polymorphic deserialization
            # This will correctly restore the original message type (AIMessage, HumanMessage, etc.)
            python_val = loads(json_str, allowed_objects="core")
            return python_val
        else:
            raise TypeTransformerFailedError(f"Unsupported binary format: `{binary_idl_object.tag}`")

    async def to_python_value(self, lv: Literal, expected_python_type: Type[BaseMessage]) -> BaseMessage:
        """Convert a Flyte Literal to a BaseMessage.

        Handles two kinds of literal values:
        1. Binary scalars (from serialization via to_literal)
        2. Protobuf Structs (from the Flyte UI)

        Args:
            lv: The Flyte Literal to convert.
            expected_python_type: The expected Python type (BaseMessage or subclass).

        Returns:
            The deserialized BaseMessage instance.
        """
        if lv and lv.HasField("scalar") and lv.scalar.HasField("binary"):
            return self.from_binary_idl(lv.scalar.binary, expected_python_type)

        # Handle protobuf struct (from UI input)
        json_str = _json_format.MessageToJson(lv.scalar.generic)
        python_val = loads(json_str, allowed_objects="core")
        return python_val


def register_langchain_transformers():
    """Register LangChain type transformers with the Flyte TypeEngine.

    This function should be called to enable LangChain type support in Flyte.
    It registers transformers for:
    - BaseMessage (and all subclasses like AIMessage, HumanMessage, etc.)

    Example:
        >>> from _langchain_types import register_langchain_transformers
        >>> register_langchain_transformers()
    """
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required for LangChain type transformers. "
            "Install it with: pip install langchain-core"
        )

    TypeEngine.register(BaseMessageTransformer())
