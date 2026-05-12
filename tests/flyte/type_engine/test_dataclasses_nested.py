from dataclasses import dataclass

import pytest

from flyte.types import TypeEngine


@dataclass(eq=True)
class MyNestedModel:
    x: int


@dataclass(eq=True)
class MyInput:
    x: int
    y: int
    m: MyNestedModel


# Models for multi-level nesting test
@dataclass(eq=True)
class DeeplyNestedModel:
    value: str


@dataclass(eq=True)
class MiddleNestedModel:
    number: int
    deep: DeeplyNestedModel


@dataclass(eq=True)
class TopLevelModel:
    name: str
    middle: MiddleNestedModel


# Models for multiple nested classes test
@dataclass(eq=True)
class Address:
    street: str
    city: str


@dataclass(eq=True)
class Contact:
    email: str
    phone: str


@dataclass(eq=True)
class Person:
    name: str
    age: int
    address: Address
    contact: Contact


@pytest.mark.asyncio
async def test_nested_dataclass():
    input = MyInput(x=1, y=2, m=MyNestedModel(x=1))
    lit = TypeEngine.to_literal_type(MyInput)
    lv = await TypeEngine.to_literal(input, python_type=MyInput, expected=lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    v = guessed(x=1, y=2, m={"x": 1})
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    pv = await TypeEngine.to_python_value(new_lv, MyInput)
    assert pv
    assert pv == input


@pytest.mark.asyncio
async def test_multi_level_nested_dataclass():
    input = TopLevelModel(
        name="test",
        middle=MiddleNestedModel(
            number=42,
            deep=DeeplyNestedModel(value="deep_value"),
        ),
    )
    lit = TypeEngine.to_literal_type(TopLevelModel)
    lv = await TypeEngine.to_literal(input, python_type=TopLevelModel, expected=lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    v = guessed(
        name="test",
        middle={"number": 42, "deep": {"value": "deep_value"}},
    )
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    pv = await TypeEngine.to_python_value(new_lv, TopLevelModel)
    assert pv
    assert pv == input


@pytest.mark.asyncio
async def test_multiple_nested_dataclasses():
    input = Person(
        name="John Doe",
        age=30,
        address=Address(street="123 Main St", city="Springfield"),
        contact=Contact(email="john@example.com", phone="555-1234"),
    )
    lit = TypeEngine.to_literal_type(Person)
    lv = await TypeEngine.to_literal(input, python_type=Person, expected=lit)

    assert lit
    assert lv

    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    v = guessed(
        name="John Doe",
        age=30,
        address={"street": "123 Main St", "city": "Springfield"},
        contact={"email": "john@example.com", "phone": "555-1234"},
    )
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    pv = await TypeEngine.to_python_value(new_lv, Person)
    assert pv
    assert pv == input
