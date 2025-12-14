import pytest
from pydantic import BaseModel

from flyte.types import TypeEngine


class MyNestedModel(BaseModel):
    x: int


class MyInput(BaseModel):
    x: int
    y: int
    m: MyNestedModel


@pytest.mark.asyncio
async def test_simple_pydantic():
    input = MyNestedModel(x=1)
    lit = TypeEngine.to_literal_type(MyNestedModel)
    lv = await TypeEngine.to_literal(input, MyNestedModel, lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed
    v = guessed(x=1)
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv
    pv = await TypeEngine.to_python_value(new_lv, MyNestedModel)
    assert pv
    assert pv == input


@pytest.mark.asyncio
async def test_nested_pydantic():
    input = MyInput(x=1, y=2, m=MyNestedModel(x=1))
    lit = TypeEngine.to_literal_type(MyInput)
    lv = await TypeEngine.to_literal(input, python_type=MyInput, expected=lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed
    print(guessed)
    v = guessed(x=1, y=2, m={"x": 1})
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv
    pv = await TypeEngine.to_python_value(new_lv, MyInput)
    assert pv
    assert pv == input
