import pytest

from flyte._internal.runtime.convert import convert_from_native_to_error
from flyte.errors import CustomError


@pytest.mark.parametrize(
    "exc,expected",
    [
        (KeyError("price"), "KeyError: 'price'"),
        (TypeError("'float' object is not subscriptable"), "TypeError: 'float' object is not subscriptable"),
        (ValueError(), "ValueError"),
    ],
)
def test_custom_error_from_exception_keeps_class_name(exc, expected):
    err = CustomError.from_exception(exc)
    assert err.code == type(exc).__name__
    assert str(err) == expected
    assert err.__cause__ is exc


def test_error_file_message_keeps_class_name():
    err = convert_from_native_to_error(CustomError.from_exception(KeyError("price"))).err
    assert err.code == "KeyError"
    assert err.message == "KeyError: 'price'"
