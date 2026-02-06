from flyteidl2.core.interface_pb2 import Variable
from flyteidl2.core.types_pb2 import LiteralType, SimpleType

from flyte.cli._params import to_click_option


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
