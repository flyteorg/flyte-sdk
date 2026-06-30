import typing

import pytest
from pydantic import BaseModel

from flyte.types import TypeEngine


class MyTaskInput(BaseModel):
    name: str = "default"
    tags: dict[str, str] | None = None  # <-- triggered KeyError: 'title' pre-2.3.6


@pytest.mark.asyncio
async def test_optional_dict_field_round_trips():
    lit = TypeEngine.to_literal_type(MyTaskInput)
    assert lit

    # This is the call that used to raise KeyError: 'title'.
    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    instance = MyTaskInput(name="hello", tags={"a": "b"})
    lv = await TypeEngine.to_literal(instance, MyTaskInput, lit)
    assert lv

    pv = await TypeEngine.to_python_value(lv, MyTaskInput)
    assert pv == instance


@pytest.mark.asyncio
async def test_optional_dict_field_none_value():
    lit = TypeEngine.to_literal_type(MyTaskInput)
    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    instance = MyTaskInput()  # tags defaults to None
    lv = await TypeEngine.to_literal(instance, MyTaskInput, lit)
    pv = await TypeEngine.to_python_value(lv, MyTaskInput)
    assert pv == instance
    assert pv.tags is None


def test_guess_python_type_handles_optional_dict_schema():
    """Directly exercise the schema path that produced the crash."""
    lit = TypeEngine.to_literal_type(MyTaskInput)
    guessed = TypeEngine.guess_python_type(lit)

    # The reconstructed type must expose the optional-dict attribute.
    hints = typing.get_type_hints(guessed)
    assert "tags" in hints
