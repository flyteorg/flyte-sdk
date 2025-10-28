from __future__ import annotations

import dataclasses
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import pytest
from pydantic import BaseModel, Field

import flyte
from flyte.io import Dir, File


def test_none_type():
    """
    Test that a task with a None type can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_none_type")

    @env.task
    async def foo(incoming: None = None) -> None:
        print(f"Hello, world! - {flyte.ctx().action} with incoming: {incoming}")
        return None

    flyte.init()
    result = flyte.run(foo)
    assert result.outputs() is None


def test_basic_types():
    """
    Test that a task with basic Python types can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_basic_types")

    @env.task
    async def basic_types(int_val: int = 42, float_val: float = 3.14, str_val: str = "hello") -> Tuple[int, float, str]:
        return int_val * 2, float_val * 2, str_val + " world"

    flyte.init()
    result = flyte.run(basic_types)
    outputs = result.outputs()
    assert outputs == (84, 6.28, "hello world")


def test_collection_types():
    """
    Test that a task with collection types can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_collection_types")

    @env.task
    async def collection_types(
        list_val: List[int] = [1, 2, 3], dict_val: Dict[str, int] = {"a": 1, "b": 2}
    ) -> Dict[str, List[int]]:
        # Modify the list and dict
        result_list = [x * 2 for x in list_val]
        result_dict = {k: v * 2 for k, v in dict_val.items()}
        return {"list_result": result_list, "dict_result": sorted(result_dict.values())}

    flyte.init()
    result = flyte.run(collection_types)
    outputs = result.outputs()
    assert outputs == {"list_result": [2, 4, 6], "dict_result": [2, 4]}


def test_optional_types():
    """
    Test that a task with Optional types can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_optional_types")

    @env.task
    async def optional_types(
        maybe_int: Optional[int] = 10, maybe_str: Optional[str] = None
    ) -> Dict[str, Union[str, int, bool, None]]:
        return {
            "int_provided": maybe_int is not None,
            "int_value": maybe_int,
            "str_provided": maybe_str is not None,
            "str_value": maybe_str,
        }

    flyte.init()
    result = flyte.run(optional_types)
    outputs = result.outputs()
    assert outputs == {"int_provided": True, "int_value": 10, "str_provided": False, "str_value": None}


def test_union_types():
    """
    Test that a task with Union types can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_union_types")

    @env.task
    async def union_types(value: Union[int, str] = "hello") -> str:
        if isinstance(value, int):
            return f"Received integer: {value}"
        else:
            return f"Received string: {value}"

    flyte.init()
    result = flyte.run(union_types)
    assert result.outputs() == "Received string: hello"

    # Try with an integer value
    result = flyte.run(union_types, value=42)
    assert result.outputs() == "Received integer: 42"


class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age


def test_custom_class():
    """
    Test that a task with custom class types can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_custom_class")

    @env.task
    async def process_person(person: Person) -> Dict[str, Any]:
        return {"name": person.name, "age": person.age, "is_adult": person.age >= 18}

    flyte.init()
    result = flyte.run(process_person, person=Person("Alice", 25))
    outputs = result.outputs()
    assert outputs["name"] == "Alice"
    assert outputs["age"] == 25
    assert outputs["is_adult"] is True


@pytest.mark.parametrize(
    "input_value,expected_output",
    [
        (5, 25),
        (10, 100),
        (0, 0),
        (-5, 25),
    ],
)
def test_parametrized_input(input_value, expected_output):
    """
    Parametrized test for multiple input-output combinations.
    """
    env = flyte.TaskEnvironment(name="test_parametrized")

    @env.task
    async def square_number(num: int) -> int:
        return num * num

    flyte.init()
    result = flyte.run(square_number, num=input_value)
    assert result.outputs() == expected_output


@dataclass
class UserProfile:
    id: int
    name: str
    email: str
    active: bool = True
    tags: List[str] = dataclasses.field(default_factory=list)


def test_dataclass():
    """
    Test that a task with dataclass types can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_dataclass")

    @env.task
    async def process_user_profile(user: UserProfile) -> Dict[str, Any]:
        return {
            "id": user.id,
            "name": user.name.upper(),
            "email": user.email,
            "is_active": user.active,
            "tag_count": len(user.tags),
        }

    flyte.init()
    user = UserProfile(id=123, name="John Doe", email="john@example.com", tags=["customer", "premium"])
    result = flyte.run(process_user_profile, user=user)
    outputs = result.outputs()

    assert outputs["id"] == 123
    assert outputs["name"] == "JOHN DOE"
    assert outputs["email"] == "john@example.com"
    assert outputs["is_active"] is True
    assert outputs["tag_count"] == 2


class Product(BaseModel):
    id: int
    name: str
    price: float
    in_stock: bool = True
    categories: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def test_pydantic_model():
    """
    Test that a task with Pydantic model types can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_pydantic_model")

    @env.task
    async def process_product(product: Product) -> Product:
        product.price = product.price * 1.2  # Apply a tax of 20%
        return product

    flyte.init()
    product = Product(
        id=42,
        name="Laptop",
        price=999.99,
        categories=["electronics", "computers"],
        metadata={"brand": "TechBrand", "weight": "1.5kg"},
    )
    result = flyte.run(process_product, product=product)
    outputs = result.outputs()

    # Check that the output is a Pydantic model with the expected values
    assert isinstance(outputs, Product)
    # Check the values after processing
    assert outputs.id == 42
    assert outputs.name == "Laptop"
    assert outputs.price == pytest.approx(1199.988)  # 20% tax applied
    assert outputs.in_stock is True
    assert outputs.categories == ["electronics", "computers"]
    assert outputs.metadata == {"brand": "TechBrand", "weight": "1.5kg"}


def test_pandas_dataframe():
    """
    Test that a task with pandas DataFrame can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_pandas_dataframe")

    @env.task
    async def process_dataframe(df: pd.DataFrame) -> int:
        return len(df)

    flyte.init()

    # Create a test dataframe
    data = {
        "name": ["Alice", "Bob", "Charlie", "David"],
        "age": [25, 30, 35, 40],
        "category": ["A", "B", "A", "C"],
        "active": [True, False, True, True],
    }
    df = pd.DataFrame(data)

    result = flyte.run(process_dataframe, df=df)
    outputs = result.outputs()

    assert outputs == 4  # The number of rows in the DataFrame


def test_file_io():
    """
    Test that a task with File IO can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_file_io")

    @env.task
    async def process_text_file(input_file: File) -> int:
        text_content = ""
        with input_file.open_sync("rt", "r") as f:
            text_content = f.read()

        # Process the content
        lines = text_content.strip().split("\n")
        word_count = sum(len(line.split()) for line in lines)

        return word_count

    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp:
        temp.write("Hello world!\nThis is a test file.\nIt contains three lines.")
        temp_path = temp.name

    try:
        flyte.init()
        input_file = File[str](path=temp_path)
        result = flyte.run(process_text_file, input_file=input_file)
        outputs = result.outputs()

        assert outputs == 11
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_file_transformation():
    """
    Test that a task that transforms a file can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_file_transformation")

    @env.task
    async def transform_csv_to_json(csv_file: File) -> File:
        # Read the CSV file into a DataFrame
        temp_df = pd.read_csv(csv_file.path)

        # Create a new output file
        output_file = File.new_remote()

        # Convert DataFrame to JSON and write to the output file
        with output_file.open_sync("w") as f:
            temp_df.to_json(f, orient="records")

        return output_file

    # Create a temporary CSV file for testing
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as temp:
        temp.write("id,name,value\n1,alpha,100\n2,beta,200\n3,gamma,300")
        temp_path = temp.name

    try:
        flyte.init()
        csv_file = File[pd.DataFrame](path=temp_path)
        result = flyte.run(transform_csv_to_json, csv_file=csv_file)
        output_file = result.outputs()

        # Verify the output file exists and contains valid JSON
        assert output_file.exists_sync()
        with output_file.open_sync("r") as f:
            content = f.read()
            assert "alpha" in content
            assert "beta" in content
            assert "gamma" in content
    finally:
        # Clean up the temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_directory_operations():
    """
    Test that a task with Dir operations can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_directory_operations")

    @env.task
    async def analyze_text_directory(text_dir: Dir[str]) -> Dict[str, Any]:
        # Analyze text files in the directory
        total_files = 0
        total_lines = 0
        total_words = 0
        file_info = []

        for file in text_dir.walk_sync():
            if file.path.endswith(".txt"):
                total_files += 1
                lines = 0
                words = 0

                with file.open_sync("rt") as f:
                    for line in f:
                        lines += 1
                        words += len(line.split())

                total_lines += lines
                total_words += words
                file_info.append({"name": file.name, "lines": lines, "words": words})

        return {
            "file_count": total_files,
            "total_lines": total_lines,
            "total_words": total_words,
            "file_details": file_info,
        }

    # Create a temporary directory with multiple text files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create 3 text files in the directory
        file_contents = [
            "This is file one.\nIt has two lines.",
            "This is file two.\nIt has\nthree lines.",
            "This is file three.\nWith some\nmore\nlines here.",
        ]

        for i, content in enumerate(file_contents, 1):
            file_path = os.path.join(temp_dir, f"file{i}.txt")
            with open(file_path, "w") as f:
                f.write(content)

        # Add a non-text file that should be ignored
        with open(os.path.join(temp_dir, "data.csv"), "w") as f:
            f.write("id,value\n1,test")

        flyte.init()
        dir_input = Dir[str](path=temp_dir)
        result = flyte.run(analyze_text_directory, text_dir=dir_input)
        outputs = result.outputs()

        assert outputs["file_count"] == 3
        assert outputs["total_lines"] == 9  # Sum of all lines in text files
        assert outputs["total_words"] == 25  # Sum of all words in text files
        assert len(outputs["file_details"]) == 3
    finally:
        # Clean up the temp directory
        import shutil

        shutil.rmtree(temp_dir)


@dataclass
class NestedData:
    name: str
    value: int


@dataclass
class ComplexDataClass:
    id: int
    nested: NestedData
    items: List[str]
    mapping: Dict[str, float]


def test_complex_dataclass():
    """
    Test that a task with complex nested dataclass can be run successfully.
    """
    env = flyte.TaskEnvironment(name="test_complex_dataclass")

    @env.task
    async def process_complex_data(data: ComplexDataClass) -> Dict[str, Any]:
        return {
            "id": data.id,
            "nested_name": data.nested.name,
            "nested_value": data.nested.value,
            "item_count": len(data.items),
            "mapping_keys": sorted(data.mapping.keys()),
            "sum_values": sum(data.mapping.values()),
        }

    flyte.init()
    complex_data = ComplexDataClass(
        id=42,
        nested=NestedData(name="test", value=100),
        items=["a", "b", "c", "d"],
        mapping={"x": 1.0, "y": 2.5, "z": 3.5},
    )

    result = flyte.run(process_complex_data, data=complex_data)
    outputs = result.outputs()

    assert outputs["id"] == 42
    assert outputs["nested_name"] == "test"
    assert outputs["nested_value"] == 100
    assert outputs["item_count"] == 4
    assert outputs["mapping_keys"] == ["x", "y", "z"]
    assert outputs["sum_values"] == 7.0
