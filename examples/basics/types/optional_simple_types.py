import flyte
from datetime import datetime, timedelta
from typing import Optional

env = flyte.TaskEnvironment(name="inputs_optional_simple_types")


@env.task
def main(
    maybe_int: Optional[int] = None,
    maybe_float: Optional[float] = None,
    maybe_str: Optional[str] = None,
    maybe_bool: Optional[bool] = None,
    maybe_datetime: Optional[datetime] = None,
    maybe_duration: Optional[timedelta] = None,
) -> str:
    """Process optional simple types (int, float, string, bool, datetime, timedelta) or None"""
    result = "Optional types received:\n"
    
    result += f"  maybe_int: {maybe_int} (provided: {maybe_int is not None})\n"
    result += f"  maybe_float: {maybe_float} (provided: {maybe_float is not None})\n"
    result += f"  maybe_str: {maybe_str} (provided: {maybe_str is not None})\n"
    result += f"  maybe_bool: {maybe_bool} (provided: {maybe_bool is not None})\n"
    result += f"  maybe_datetime: {maybe_datetime} (provided: {maybe_datetime is not None})\n"
    result += f"  maybe_duration: {maybe_duration} (provided: {maybe_duration is not None})\n"
    
    return result


if __name__ == "__main__":
    flyte.init_from_config()

    # Test with all values provided
    print("Testing with all values provided:")
    r1 = flyte.run(
        main,
        maybe_int=42,
        maybe_float=3.14,
        maybe_str="Hello",
        maybe_bool=True,
        maybe_datetime=datetime.now(),
        maybe_duration=timedelta(hours=1),
    )
    print(r1.name)
    print(r1.url)
    r1.wait()

    # Test with some values as None
    print("\nTesting with some values as None:")
    r2 = flyte.run(
        main,
        maybe_int=None,
        maybe_float=2.71,
        maybe_str=None,
        maybe_bool=False,
        maybe_datetime=None,
        maybe_duration=timedelta(minutes=30),
    )
    print(r2.name)
    print(r2.url)
    r2.wait()

    # Test with all None (using defaults)
    print("\nTesting with all None (using defaults):")
    r3 = flyte.run(main)
    print(r3.name)
    print(r3.url)
    r3.wait()

