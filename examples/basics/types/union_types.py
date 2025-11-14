from dataclasses import dataclass

import flyte

env = flyte.TaskEnvironment(name="inputs_union_types")


@dataclass
class UserData:
    """A simple dataclass for demonstration"""

    name: str
    age: int
    email: str


@env.task
def main(value: str | int | UserData) -> str:
    """Process union type input (string | int | dataclass)"""
    if isinstance(value, str):
        return f"Received string: {value}"
    elif isinstance(value, int):
        return f"Received integer: {value}"
    elif isinstance(value, UserData):
        return f"Received dataclass: name={value.name}, age={value.age}, email={value.email}"
    else:
        return f"Received unknown type: {type(value)}"


if __name__ == "__main__":
    flyte.init_from_config()

    # Test with string
    print("Testing with string:")
    r1 = flyte.run(main, value="Hello, Flyte!")
    print(r1.name)
    print(r1.url)
    r1.wait()

    # Test with int
    print("\nTesting with int:")
    r2 = flyte.run(main, value=42)
    print(r2.name)
    print(r2.url)
    r2.wait()

    # Test with dataclass
    print("\nTesting with dataclass:")
    user = UserData(name="Alice", age=30, email="alice@example.com")
    r3 = flyte.run(main, value=user)
    print(r3.name)
    print(r3.url)
    r3.wait()
