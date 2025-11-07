import inspect

import pytest

from flyte._docstring import Docstring
from flyte._internal.runtime.types_serde import transform_native_to_typed_interface
from flyte.models import NativeInterface


def test_transform_native_to_typed_interface_with_docstring():
    """Test that interface with docstring correctly extracts descriptions"""
    docstring_text = """
    A function that processes user data.

    Args:
        name: The user's name
        age: The user's age in years

    Returns:
        result: A formatted string with user info
    """

    interface = NativeInterface(
        inputs={"name": (str, inspect.Parameter.empty), "age": (int, 42)},
        outputs={"result": str},
        docstring=Docstring(docstring=docstring_text),
    )

    result = transform_native_to_typed_interface(interface)

    assert result is not None
    assert "name" in result.inputs.variables
    assert "age" in result.inputs.variables
    assert "result" in result.outputs.variables

    # Check that descriptions are correctly extracted
    assert result.inputs.variables["name"].description == "The user's name"
    assert result.inputs.variables["age"].description == "The user's age in years"
    assert result.outputs.variables["result"].description == "A formatted string with user info"


def test_transform_native_to_typed_interface_empty_interface():
    """Test that empty interface (no inputs/outputs) works correctly"""
    interface = NativeInterface(
        inputs={},
        outputs={},
        docstring=None,
    )

    result = transform_native_to_typed_interface(interface)

    assert result is not None
    assert len(result.inputs.variables) == 0
    assert len(result.outputs.variables) == 0
