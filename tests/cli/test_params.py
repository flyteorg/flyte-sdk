import enum

import click
import pytest
from flyteidl2.core.interface_pb2 import Variable
from flyteidl2.core.types_pb2 import LiteralType, SimpleType

from flyte.cli._params import EnumParamType, to_click_option


class Color(str, enum.Enum):
    RED = "red-value"
    GREEN = "green-value"
    BLUE = "blue-value"


class Size(str, enum.Enum):
    SMALL = "sm-value"
    MEDIUM = "md-value"
    LARGE = "lg-value"


def test_enum_param_type_choices_are_names():
    """EnumParamType should expose enum names as CLI choices, not values."""
    param_type = EnumParamType(Color)
    assert list(param_type.choices) == ["RED", "GREEN", "BLUE"]


def test_enum_param_type_convert_name():
    """Passing an enum name (e.g. GREEN) should return the corresponding enum instance."""
    param_type = EnumParamType(Color)
    assert param_type.convert("GREEN", param=None, ctx=None) is Color.GREEN


def test_enum_param_type_convert_str_enum():
    """EnumParamType should work with StrEnum subclasses."""
    param_type = EnumParamType(Size)
    assert param_type.convert("SMALL", param=None, ctx=None) is Size.SMALL


def test_enum_param_type_rejects_value():
    """Passing an enum value (e.g. 'red-value') should be rejected — only names are valid."""
    param_type = EnumParamType(Color)
    with pytest.raises((click.exceptions.BadParameter, SystemExit)):
        param_type.convert("red-value", param=None, ctx=None)


def test_enum_param_type_passthrough_instance():
    """Passing an already-converted enum instance should be returned as-is."""
    param_type = EnumParamType(Color)
    assert param_type.convert(Color.BLUE, param=None, ctx=None) is Color.BLUE


def test_boolean_flag_false_default():
    """Test boolean parameter with False default creates a simple flag."""
    literal_var = Variable(type=LiteralType(simple=SimpleType.BOOLEAN), description="A boolean flag")

    option = to_click_option(input_name="flag", literal_var=literal_var, python_type=bool, default_val=False)

    assert option.opts == ["--flag"]
    assert not option.secondary_opts
    assert option.is_flag is True
    assert option.default is False
    assert option.required is False


def test_boolean_flag_true_default():
    """Test boolean parameter with True default creates a --flag/--no-flag pattern."""
    literal_var = Variable(type=LiteralType(simple=SimpleType.BOOLEAN), description="A boolean flag with True default")

    option = to_click_option(input_name="enabled", literal_var=literal_var, python_type=bool, default_val=True)

    assert option.opts == ["--enabled"]
    assert option.secondary_opts == ["--no-enabled"]
    assert option.is_flag is True
    assert option.default is True
    assert option.required is False


def test_boolean_flag_no_default():
    """Test boolean parameter with no default creates a simple flag with False default."""
    literal_var = Variable(type=LiteralType(simple=SimpleType.BOOLEAN), description="A boolean flag with no default")

    option = to_click_option(input_name="debug", literal_var=literal_var, python_type=bool, default_val=None)

    assert option.opts == ["--debug"]
    assert option.is_flag is True
    assert option.default is False  # Should default to False for boolean flags
    assert option.required is False
