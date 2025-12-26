from datetime import datetime, timedelta

import flyte

env = flyte.TaskEnvironment(name="inputs_simple_type_defaults")


@env.task
def main(
    str: str = "World",
    int: int = 42,
    float: float = 3.14,
    bool: bool = True,
    start_time: datetime = datetime(2024, 1, 1, 12, 0, 0),
    duration: timedelta = timedelta(hours=1),
) -> str:
    """Process simple types with default values"""
    result = f"Hello, simple types: str={str}, int={int}, float={float}, bool={bool}"
    result += f"\nTime: {start_time}, Duration: {duration}"
    return result


if __name__ == "__main__":
    flyte.init_from_config()

    # Test with all defaults
    print("Testing with all defaults:")
    r1 = flyte.run(main)
    print(r1.name)
    print(r1.url)
    r1.wait()

    # Test with some custom values
    print("\nTesting with some custom values:")
    r2 = flyte.run(main, str="Flyte", int=100, float=2.71)
    print(r2.name)
    print(r2.url)
    r2.wait()
