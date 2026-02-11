from enum import Enum

import pytest

from flyte.types import TypeEngine


class Foo(Enum):
    A = "AAA"
    B = "BBB"
    C = "CCC"


@pytest.mark.asyncio
async def test_enums():
    lit = TypeEngine.to_literal_type(Foo)
    lv = await TypeEngine.to_literal(Foo.B, Foo, lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed
    v = guessed["B"]
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv
    pv = await TypeEngine.to_python_value(new_lv, Foo)
    assert pv
    assert pv == Foo.B
