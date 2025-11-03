"""
Custom type transformer for PositiveInt type.
This demonstrates how to create a type transformer for a custom integer wrapper.
"""

from typing import Type

from flyteidl2.core import literals_pb2, types_pb2

from flyte import logger
from flyte.types import TypeEngine, TypeTransformer, TypeTransformerFailedError
from my_type.custom_type import PositiveInt


class PositiveIntTransformer(TypeTransformer[PositiveInt]):
    """
    Type transformer for PositiveInt that validates and transforms positive integers.

    This transformer:
    - Converts PositiveInt to Flyte's INTEGER literal type
    - Validates that values are positive integers
    - Handles serialization/deserialization between Python and Flyte types
    """

    def __init__(self):
        super().__init__(name="PositiveInt", t=PositiveInt)

    def get_literal_type(self, t: Type[PositiveInt]) -> types_pb2.LiteralType:
        """
        Returns the Flyte literal type for PositiveInt.
        We use SimpleType.INTEGER since PositiveInt is a wrapper around int.
        """
        return types_pb2.LiteralType(
            simple=types_pb2.SimpleType.INTEGER,
            structure=types_pb2.TypeStructure(tag="PositiveInt"),
        )

    async def to_literal(
        self,
        python_val: PositiveInt,
        python_type: Type[PositiveInt],
        expected: types_pb2.LiteralType,
    ) -> literals_pb2.Literal:
        """
        Converts a PositiveInt instance to a Flyte Literal.

        Args:
            python_val: The PositiveInt value to convert
            python_type: The expected Python type
            expected: The expected Flyte literal type

        Returns:
            A Flyte Literal containing the integer value

        Raises:
            TypeTransformerFailedError: If the value is not a PositiveInt
        """
        if not isinstance(python_val, PositiveInt):
            raise TypeTransformerFailedError(f"Expected PositiveInt, got {type(python_val).__name__}")

        return literals_pb2.Literal(
            scalar=literals_pb2.Scalar(primitive=literals_pb2.Primitive(integer=python_val.value))
        )

    async def to_python_value(self, lv: literals_pb2.Literal, expected_python_type: Type[PositiveInt]) -> PositiveInt:
        """
        Converts a Flyte Literal back to a PositiveInt instance.

        Args:
            lv: The Flyte Literal to convert
            expected_python_type: The expected Python type (PositiveInt)

        Returns:
            A PositiveInt instance

        Raises:
            TypeTransformerFailedError: If the literal doesn't contain a valid integer
            ValueError: If the integer value is not positive
        """
        if not lv.scalar or not lv.scalar.primitive:
            raise TypeTransformerFailedError(f"Cannot convert literal {lv} to PositiveInt: missing scalar primitive")

        value = lv.scalar.primitive.integer

        # PositiveInt constructor will validate that the value is positive
        try:
            return PositiveInt(value)
        except (TypeError, ValueError) as e:
            raise TypeTransformerFailedError(f"Cannot convert value {value} to PositiveInt: {e}")

    def guess_python_type(self, literal_type: types_pb2.LiteralType) -> Type[PositiveInt]:
        """
        Guesses the Python type from a Flyte literal type.
        This is used for reverse type inference.
        """
        if (
            literal_type.simple == types_pb2.SimpleType.INTEGER
            and literal_type.structure
            and literal_type.structure.tag == "PositiveInt"
        ):
            return PositiveInt
        raise ValueError(f"Cannot guess PositiveInt from literal type {literal_type}")


def register_positive_int_transformer():
    """Register the PositiveIntTransformer in the TypeEngine."""
    TypeEngine.register(PositiveIntTransformer())
    logger.info("Registered PositiveIntTransformer in TypeEngine.")
