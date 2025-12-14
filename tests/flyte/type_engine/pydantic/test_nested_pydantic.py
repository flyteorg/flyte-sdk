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
    lv = TypeEngine.to_literal(input, MyNestedModel, lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed
    v = guessed(x=1)
    assert v.__dict__ == input.model_dump()


@pytest.mark.asyncio
async def test_nested_pydantic():
    input = MyInput(x=1, y=2, m=MyNestedModel(x=1))
    lit = TypeEngine.to_literal_type(MyInput)
    lv = TypeEngine.to_literal(input, python_type=MyInput, expected=lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed
    print(guessed)
