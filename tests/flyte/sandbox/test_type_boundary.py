"""Tests for sandboxed type boundary validation."""

import inspect
from typing import Dict, List, Optional, Set, Tuple

import pytest

from flyte.io import DataFrame, Dir, File
from flyte.sandbox._type_boundary import validate_sandboxed_interface


def _make_inputs(**types_and_defaults) -> Dict:
    """Helper to create an inputs dict matching NativeInterface format."""
    return {name: (tp, inspect.Parameter.empty) for name, tp in types_and_defaults.items()}


def _make_outputs(**types) -> Dict:
    return types


class TestAllowedTypes:
    def test_primitives(self):
        inputs = _make_inputs(a=int, b=float, c=str, d=bool, e=bytes)
        outputs = _make_outputs(o0=int)
        validate_sandboxed_interface(inputs, outputs)

    def test_none_type(self):
        inputs = _make_inputs(a=type(None))
        outputs = _make_outputs(o0=type(None))
        validate_sandboxed_interface(inputs, outputs)

    def test_collections(self):
        inputs = _make_inputs(a=List[int], b=Dict[str, float], c=Tuple[int, str], d=Set[bool])
        outputs = _make_outputs(o0=List[str])
        validate_sandboxed_interface(inputs, outputs)

    def test_nested_collections(self):
        inputs = _make_inputs(a=List[Dict[str, int]], b=Dict[str, List[float]])
        outputs = _make_outputs(o0=List[List[int]])
        validate_sandboxed_interface(inputs, outputs)

    def test_optional(self):
        inputs = _make_inputs(a=Optional[int], b=Optional[str])
        outputs = _make_outputs(o0=Optional[int])
        validate_sandboxed_interface(inputs, outputs)

    def test_flyte_io_types(self):
        inputs = _make_inputs(a=File, b=Dir, c=DataFrame)
        outputs = _make_outputs(o0=File, o1=Dir)
        validate_sandboxed_interface(inputs, outputs)

    def test_bare_collections(self):
        inputs = _make_inputs(a=list, b=dict, c=tuple, d=set)
        outputs = _make_outputs(o0=list)
        validate_sandboxed_interface(inputs, outputs)

    def test_empty_interface(self):
        validate_sandboxed_interface({}, {})

    def test_untyped_parameter(self):
        """Untyped parameters (empty annotation) should be allowed through."""
        inputs = {"a": (inspect.Parameter.empty, inspect.Parameter.empty)}
        outputs = {}
        validate_sandboxed_interface(inputs, outputs)


class TestBlockedTypes:
    def test_custom_class(self):
        class MyClass:
            pass

        inputs = _make_inputs(a=MyClass)
        with pytest.raises(TypeError, match="unsupported type"):
            validate_sandboxed_interface(inputs, {})

    def test_custom_class_in_output(self):
        class MyClass:
            pass

        outputs = _make_outputs(o0=MyClass)
        with pytest.raises(TypeError, match="unsupported type"):
            validate_sandboxed_interface({}, outputs)

    def test_nested_custom_class(self):
        class MyClass:
            pass

        inputs = _make_inputs(a=List[MyClass])
        with pytest.raises(TypeError, match="unsupported type"):
            validate_sandboxed_interface(inputs, {})

    def test_error_message_includes_param_name(self):
        class Bad:
            pass

        inputs = _make_inputs(my_param=Bad)
        with pytest.raises(TypeError, match="my_param"):
            validate_sandboxed_interface(inputs, {})
