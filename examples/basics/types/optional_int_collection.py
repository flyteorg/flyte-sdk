from typing import Optional

import flyte

env = flyte.TaskEnvironment(name="optional_int_collection")


@env.task
def main(numbers: Optional[list[int]] = None) -> str:
    """Process an optional collection of integers"""
    if numbers is None:
        return "No numbers provided (None)"

    if not numbers:
        return "Empty list provided"

    result = f"Received {len(numbers)} integers:\n"
    result += f"  Numbers: {numbers}\n"
    result += f"  Sum: {sum(numbers)}\n"
    result += f"  Average: {sum(numbers) / len(numbers):.2f}\n"
    result += f"  Min: {min(numbers)}\n"
    result += f"  Max: {max(numbers)}"

    return result


if __name__ == "__main__":
    flyte.init_from_config()

    # Test with a list of integers
    print("Testing with a list of integers:")
    r1 = flyte.run(main, numbers=[1, 2, 3, 4, 5, 10, 20, 30])
    print(r1.name)
    print(r1.url)
    r1.wait()

    # Test with None (using default)
    print("\nTesting with None (using default):")
    r2 = flyte.run(main)
    print(r2.name)
    print(r2.url)
    r2.wait()

    # Test with empty list
    print("\nTesting with empty list:")
    r3 = flyte.run(main, numbers=[])
    print(r3.name)
    print(r3.url)
    r3.wait()

    # Test with another list
    print("\nTesting with another list:")
    r4 = flyte.run(main, numbers=[100, 200, 300])
    print(r4.name)
    print(r4.url)
    r4.wait()
