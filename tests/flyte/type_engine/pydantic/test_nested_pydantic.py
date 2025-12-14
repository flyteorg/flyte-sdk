import pytest
from pydantic import BaseModel

from flyte.types import TypeEngine


class MyNestedModel(BaseModel):
    x: int


class MyInput(BaseModel):
    x: int
    y: int
    m: MyNestedModel


# Models for multi-level nesting test
class DeeplyNestedModel(BaseModel):
    value: str


class MiddleNestedModel(BaseModel):
    number: int
    deep: DeeplyNestedModel


class TopLevelModel(BaseModel):
    name: str
    middle: MiddleNestedModel


# Models for multiple nested classes test
class Address(BaseModel):
    street: str
    city: str


class Contact(BaseModel):
    email: str
    phone: str


class Person(BaseModel):
    name: str
    age: int
    address: Address
    contact: Contact


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


@pytest.mark.asyncio
async def test_multi_level_nested_pydantic():
    """Test with 3 levels of nesting: TopLevelModel -> MiddleNestedModel -> DeeplyNestedModel"""
    input = TopLevelModel(
        name="test",
        middle=MiddleNestedModel(number=42, deep=DeeplyNestedModel(value="deep_value")),
    )
    lit = TypeEngine.to_literal_type(TopLevelModel)
    lv = await TypeEngine.to_literal(input, python_type=TopLevelModel, expected=lit)

    assert lit
    assert lv

    # Test guessing and reconstructing from dict
    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    # Initialize with nested dicts
    v = guessed(name="test", middle={"number": 42, "deep": {"value": "deep_value"}})
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    # Verify roundtrip
    pv = await TypeEngine.to_python_value(new_lv, TopLevelModel)
    assert pv
    assert pv == input


@pytest.mark.asyncio
async def test_multiple_nested_classes():
    """Test with multiple nested classes at the same level"""
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

    # Test guessing and reconstructing from dict
    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    # Initialize with nested dicts
    v = guessed(
        name="John Doe",
        age=30,
        address={"street": "123 Main St", "city": "Springfield"},
        contact={"email": "john@example.com", "phone": "555-1234"},
    )
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    # Verify roundtrip
    pv = await TypeEngine.to_python_value(new_lv, Person)
    assert pv
    assert pv == input


@pytest.mark.asyncio
async def test_nested_with_mixed_initialization():
    """Test initializing with a mix of dict and object instances"""
    guessed = TypeEngine.guess_python_type(TypeEngine.to_literal_type(Person))

    # Mix dict and actual nested objects
    address_obj = Address(street="456 Elm St", city="Shelbyville")
    v = guessed(
        name="Jane Doe",
        age=25,
        address=address_obj,  # Use actual object
        contact={"email": "jane@example.com", "phone": "555-5678"},  # Use dict
    )

    # Should be able to serialize it
    lit = TypeEngine.to_literal_type(Person)
    lv = await TypeEngine.to_literal(v, guessed, lit)
    assert lv

    # Verify the values are correct
    pv = await TypeEngine.to_python_value(lv, Person)
    assert pv.name == "Jane Doe"
    assert pv.age == 25
    assert pv.address.street == "456 Elm St"
    assert pv.address.city == "Shelbyville"
    assert pv.contact.email == "jane@example.com"
    assert pv.contact.phone == "555-5678"


@pytest.mark.asyncio
async def test_deeply_nested_chain():
    """Test a chain of nested models to ensure deep nesting works correctly"""

    class Level4(BaseModel):
        value: int

    class Level3(BaseModel):
        l4: Level4

    class Level2(BaseModel):
        l3: Level3

    class Level1(BaseModel):
        name: str
        l2: Level2

    input = Level1(name="root", l2=Level2(l3=Level3(l4=Level4(value=100))))
    lit = TypeEngine.to_literal_type(Level1)
    lv = await TypeEngine.to_literal(input, python_type=Level1, expected=lit)

    assert lit
    assert lv

    # Test guessing and reconstructing from deeply nested dict
    guessed = TypeEngine.guess_python_type(lit)
    assert guessed

    v = guessed(name="root", l2={"l3": {"l4": {"value": 100}}})
    new_lv = await TypeEngine.to_literal(v, guessed, lit)
    assert new_lv == lv

    # Verify roundtrip
    pv = await TypeEngine.to_python_value(new_lv, Level1)
    assert pv
    assert pv == input
    assert pv.l2.l3.l4.value == 100
