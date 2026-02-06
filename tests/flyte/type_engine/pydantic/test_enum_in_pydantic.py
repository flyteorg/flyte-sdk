import dataclasses
from enum import Enum

import pytest
from pydantic import BaseModel

from flyte.types import TypeEngine
from flyte.types._type_engine import convert_mashumaro_json_schema_to_python_class


class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Job(BaseModel):
    name: str
    status: Status


@pytest.mark.asyncio
async def test_pydantic_model_with_enum_ref():
    """Test that a Pydantic model with an enum field (which produces a $ref in
    the JSON schema) can be round-tripped through the type engine and that
    guess_python_type reconstructs a valid dataclass."""
    input = Job(name="test-job", status=Status.PENDING)

    lit = TypeEngine.to_literal_type(Job)
    lv = await TypeEngine.to_literal(input, python_type=Job, expected=lit)

    assert lit
    assert lv

    # Roundtrip via the real Pydantic model
    pv = await TypeEngine.to_python_value(lv, Job)
    assert pv == input

    # Guess python type from the schema (simulates pyflyte run behaviour)
    guessed = TypeEngine.guess_python_type(lit)
    assert dataclasses.is_dataclass(guessed)

    # The enum field should be reconstructed as str (enum $ref resolved)
    v = guessed(name="test-job", status="pending")
    assert v.name == "test-job"
    assert v.status == "pending"
