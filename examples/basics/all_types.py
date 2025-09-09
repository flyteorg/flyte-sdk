"""
This example demonstrates all supported Flyte types as inputs to tasks.
It shows how to use primitive types, complex types, and special Flyte types.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
from pydantic import BaseModel

import flyte
from flyte.io import Dir, File, DataFrame
from flyte.types import FlytePickle

# Set up environment with required dependencies for different types
env = flyte.TaskEnvironment(
    name="all_types",
    image=flyte.Image.from_debian_base().with_pip_packages("pandas", "pyarrow"),
    resources=flyte.Resources(cpu="1", memory="1Gi"),
)

# Define custom types for demonstration


class Status(Enum):
    """Example enum with string values (required by Flyte)"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Person:
    """Example dataclass"""

    name: str
    age: int
    email: Optional[str] = None


class PersonModel(BaseModel):
    """Example Pydantic model"""

    name: str
    age: int
    is_active: bool = True


# Main task that takes all types as inputs
@env.task
async def all_types_task(
    # Primitive types
    my_int: int,
    my_float: float,
    my_str: str,
    my_bool: bool,
    # Date and time types
    my_datetime: datetime,
    my_timedelta: timedelta,
    # Collection types
    my_list: List[str],
    my_dict: Dict[str, int],
    my_tuple: Tuple[str, int, bool],
    # Optional types
    my_optional_str: Optional[str],
    my_optional_int: Optional[int],
    # Union types
    my_union: Union[str, int],
    # Complex nested types
    nested_list: List[List[int]],
    nested_dict: Dict[str, List[float]],
    # Custom types
    my_dataclass: Person,
    my_pydantic: PersonModel,
    my_enum: Status,
    # Flyte special types
    my_file: File,
    my_dir: Dir,
    my_dataframe: pd.DataFrame,
    my_flyte_dataframe: DataFrame,
    # Any type (serialized with pickle)
    my_any: Any,
    # FlytePickle for complex objects that can't be serialized otherwise
    my_pickle: FlytePickle,
) -> Dict[str, str]:
    """
    Main task that processes all different Flyte-supported input types.
    Returns a summary of what was received.
    """
    summary = {}

    # Process primitive types
    primitive_results = await process_primitives(my_int, my_float, my_str, my_bool)
    summary.update(primitive_results)

    # Process datetime types
    datetime_results = await process_datetimes(my_datetime, my_timedelta)
    summary.update(datetime_results)

    # Process collections
    collection_results = await process_collections(my_list, my_dict, my_tuple)
    summary.update(collection_results)

    # Process optional types
    optional_results = await process_optionals(my_optional_str, my_optional_int)
    summary.update(optional_results)

    # Process union types
    union_results = await process_unions(my_union)
    summary.update(union_results)

    # Process nested types
    nested_results = await process_nested_types(nested_list, nested_dict)
    summary.update(nested_results)

    # Process custom types
    custom_results = await process_custom_types(my_dataclass, my_pydantic, my_enum)
    summary.update(custom_results)

    # Process Flyte types
    flyte_results = await process_flyte_types(
        my_file, my_dir, my_dataframe, my_flyte_dataframe
    )
    summary.update(flyte_results)

    # Process any and pickle types
    any_pickle_results = await process_any_pickle_types(my_any, my_pickle)
    summary.update(any_pickle_results)

    return summary


# Individual tasks for each type category
@env.task
async def process_primitives(
    my_int: int, my_float: float, my_str: str, my_bool: bool
) -> Dict[str, str]:
    """Process primitive types"""
    return {
        "int": f"Received integer: {my_int} (type: {type(my_int).__name__})",
        "float": f"Received float: {my_float} (type: {type(my_float).__name__})",
        "str": f"Received string: '{my_str}' (length: {len(my_str)})",
        "bool": f"Received boolean: {my_bool} (type: {type(my_bool).__name__})",
    }


@env.task
async def process_datetimes(
    my_datetime: datetime, my_timedelta: timedelta
) -> Dict[str, str]:
    """Process datetime and timedelta types"""
    return {
        "datetime": f"Received datetime: {my_datetime} (weekday: {my_datetime.strftime('%A')})",
        "timedelta": f"Received timedelta: {my_timedelta} (total seconds: {my_timedelta.total_seconds()})",
    }


@env.task
async def process_collections(
    my_list: List[str], my_dict: Dict[str, int], my_tuple: Tuple[str, int, bool]
) -> Dict[str, str]:
    """Process collection types"""
    return {
        "list": f"Received list with {len(my_list)} items: {my_list}",
        "dict": f"Received dict with {len(my_dict)} keys: {list(my_dict.keys())}",
        "tuple": f"Received tuple: {my_tuple} (types: {[type(x).__name__ for x in my_tuple]})",
    }


@env.task
async def process_optionals(
    my_optional_str: Optional[str], my_optional_int: Optional[int]
) -> Dict[str, str]:
    """Process optional types"""
    return {
        "optional_str": f"Optional string: {'None' if my_optional_str is None else repr(my_optional_str)}",
        "optional_int": f"Optional int: {'None' if my_optional_int is None else my_optional_int}",
    }


@env.task
async def process_unions(my_union: Union[str, int]) -> Dict[str, str]:
    """Process union types"""
    return {
        "union": f"Union value: {my_union} (actual type: {type(my_union).__name__})",
    }


@env.task
async def process_nested_types(
    nested_list: List[List[int]], nested_dict: Dict[str, List[float]]
) -> Dict[str, str]:
    """Process nested collection types"""
    total_nested_items = sum(len(sublist) for sublist in nested_list)
    total_dict_items = sum(len(values) for values in nested_dict.values())

    return {
        "nested_list": f"Nested list with {len(nested_list)} sublists, {total_nested_items} total items",
        "nested_dict": f"Nested dict with {len(nested_dict)} keys, {total_dict_items} total values",
    }


@env.task
async def process_custom_types(
    my_dataclass: Person, my_pydantic: PersonModel, my_enum: Status
) -> Dict[str, str]:
    """Process custom dataclass, Pydantic model, and enum types"""
    return {
        "dataclass": f"Person: {my_dataclass.name}, age {my_dataclass.age}, email: {my_dataclass.email}",
        "pydantic": f"PersonModel: {my_pydantic.name}, age {my_pydantic.age}, active: {my_pydantic.is_active}",
        "enum": f"Status: {my_enum.value} (type: {type(my_enum).__name__})",
    }


@env.task
async def process_flyte_types(
    my_file: File,
    my_dir: Dir,
    my_dataframe: pd.DataFrame,
    my_flyte_dataframe: DataFrame,
) -> Dict[str, str]:
    """Process Flyte-specific types"""
    results = {}

    # Process File
    async with my_file.open("rb") as f:
        content = await f.read()
        results["file"] = f"File: {my_file.path} ({len(content)} bytes)"

    # Process Dir
    file_count = 0
    async for file_obj in my_dir.walk():
        file_count += 1
    results["dir"] = f"Directory: {my_dir.path} ({file_count} files)"

    # Process pandas DataFrame
    results["pandas_df"] = (
        f"Pandas DataFrame: {my_dataframe.shape} shape, columns: {list(my_dataframe.columns)}"
    )

    # Process Flyte DataFrame
    df_data = await my_flyte_dataframe.open(pd.DataFrame).all()
    results["flyte_df"] = (
        f"Flyte DataFrame: {df_data.shape} shape, columns: {list(df_data.columns)}"
    )

    return results


@env.task
async def process_any_pickle_types(
    my_any: Any, my_pickle: FlytePickle
) -> Dict[str, str]:
    """Process Any and FlytePickle types"""
    return {
        "any": f"Any type: {my_any} (actual type: {type(my_any).__name__})",
        "pickle": f"FlytePickle: {my_pickle} (type: {type(my_pickle).__name__})",
    }


# Example workflow that creates sample data and runs the main task
@env.task
async def create_sample_data() -> (
    Tuple[File, Dir, pd.DataFrame, DataFrame, Any, FlytePickle]
):
    """Create sample data for testing"""
    # Create a sample file
    sample_file = File.new_remote()
    async with sample_file.open("wb") as f:
        await f.write(b"Hello from Flyte! This is sample file content.")

    # Create a sample directory with files
    import tempfile
    import os

    temp_dir = tempfile.mkdtemp()
    for i in range(3):
        with open(os.path.join(temp_dir, f"file_{i}.txt"), "w") as f:
            f.write(f"Content of file {i}")

    sample_dir = await Dir.from_local(temp_dir)

    # Create sample DataFrames
    sample_pd_df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [95.5, 87.2, 91.8],
        }
    )

    sample_flyte_df = DataFrame.from_df(sample_pd_df)

    # Create sample Any type (complex object)
    sample_any = {"nested": {"data": [1, 2, 3]}, "metadata": {"version": "1.0"}}

    # Create sample FlytePickle
    class CustomObject:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return f"CustomObject(value={self.value})"

    sample_pickle = FlytePickle(CustomObject("Hello from pickle!"))

    return (
        sample_file,
        sample_dir,
        sample_pd_df,
        sample_flyte_df,
        sample_any,
        sample_pickle,
    )


@env.task
async def demo_workflow() -> Dict[str, str]:
    """
    Demo workflow that creates sample data and processes all types.
    """
    # Create sample data
    (
        sample_file,
        sample_dir,
        sample_pd_df,
        sample_flyte_df,
        sample_any,
        sample_pickle,
    ) = await create_sample_data()

    # Run the main all_types_task with sample data
    result = await all_types_task(
        # Primitive types
        my_int=42,
        my_float=3.14159,
        my_str="Hello, Flyte!",
        my_bool=True,
        # Date and time types
        my_datetime=datetime(2024, 1, 15, 10, 30, 0),
        my_timedelta=timedelta(days=7, hours=3, minutes=45),
        # Collection types
        my_list=["apple", "banana", "cherry"],
        my_dict={"a": 1, "b": 2, "c": 3},
        my_tuple=("hello", 123, False),
        # Optional types
        my_optional_str="optional value",
        my_optional_int=None,
        # Union types
        my_union="I'm a string in a union",
        # Complex nested types
        nested_list=[[1, 2, 3], [4, 5], [6, 7, 8, 9]],
        nested_dict={"group1": [1.1, 2.2], "group2": [3.3, 4.4, 5.5]},
        # Custom types
        my_dataclass=Person(name="John Doe", age=30, email="john@example.com"),
        my_pydantic=PersonModel(name="Jane Smith", age=28, is_active=True),
        my_enum=Status.RUNNING,
        # Flyte special types
        my_file=sample_file,
        my_dir=sample_dir,
        my_dataframe=sample_pd_df,
        my_flyte_dataframe=sample_flyte_df,
        # Any and pickle types
        my_any=sample_any,
        my_pickle=sample_pickle,
    )

    return result


if __name__ == "__main__":
    flyte.init_from_config("../../config.yaml")

    print("Running all types demo...")
    run = flyte.run(demo_workflow)
    print(f"Run URL: {run.url}")

    # Wait for completion and print results
    # run.wait(run)
    # outputs = run.outputs()

    # print("\n=== All Types Processing Results ===")
    # for key, value in outputs.items():
    #     print(f"{key}: {value}")
