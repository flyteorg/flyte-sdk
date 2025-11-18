from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import flyte

env = flyte.TaskEnvironment(name="inputs_dataclass_types")


@dataclass
class NestedData:
    """A nested dataclass with basic types"""

    name: str
    value: int
    score: float


@dataclass
class ComplexData:
    """A complex dataclass covering all supported subtypes"""

    # Basic types
    str_field: str
    int_field: int
    float_field: float
    bool_field: bool

    # Date/time types
    start_time: datetime
    duration: timedelta

    # Lists of various types
    string_list: List[str]
    int_list: List[int]
    float_list: List[float]
    bool_list: List[bool]

    # Nested dataclass
    nested: NestedData

    # List of nested dataclasses
    nested_list: List[NestedData]

    # Optional fields
    optional_str: Optional[str] = None
    optional_int: Optional[int] = None


@env.task
def main(data: ComplexData) -> str:
    """Process complex dataclass with all supported subtypes"""
    result = f"""Hello, simple types: str={data.str_field}, int={data.int_field}, \
float={data.float_field}, bool={data.bool_field}"""
    result += f"\nTime: {data.start_time}, Duration: {data.duration}"
    result += f"\nComplex data: {data.str_field}, {data.int_field}, {data.float_field}, {data.bool_field}"
    result += f"\nNested: {data.nested.name} = {data.nested.value}"
    result += f"\nTotal nested items: {len(data.nested_list)}"
    return result


if __name__ == "__main__":
    flyte.init_from_config()

    # Create nested data
    nested1 = NestedData(name="first", value=10, score=95.5)
    nested2 = NestedData(name="second", value=20, score=87.3)
    nested3 = NestedData(name="third", value=30, score=92.1)

    # Create complex data instance
    complex_data = ComplexData(
        str_field="Hello World!",
        int_field=42,
        float_field=3.14,
        bool_field=True,
        start_time=datetime.now(),
        duration=timedelta(hours=1),
        string_list=["hello", "world", "flyte"],
        int_list=[1, 2, 3, 4, 5],
        float_list=[1.1, 2.2, 3.3],
        bool_list=[True, False, True],
        nested=nested1,
        nested_list=[nested1, nested2, nested3],
        optional_str="optional value",
        optional_int=99,
    )

    r = flyte.run(main, data=complex_data)
    print(r.name)
    print(r.url)
    r.wait()
