import flyte

env = flyte.TaskEnvironment(name="inputs_int_collection")


@env.task
def main(numbers: list[int]) -> str:
    """Process a collection of integers"""
    result = f"Received {len(numbers)} integers:\n"
    result += f"  Numbers: {numbers}\n"
    result += f"  Sum: {sum(numbers)}\n"
    result += f"  Average: {sum(numbers) / len(numbers) if numbers else 0:.2f}\n"
    result += f"  Min: {min(numbers) if numbers else 'N/A'}\n"
    result += f"  Max: {max(numbers) if numbers else 'N/A'}"
    return result


if __name__ == "__main__":
    flyte.init_from_config()

    test_numbers = [1, 2, 3, 4, 5, 10, 20, 30]
    r = flyte.run(main, numbers=test_numbers)
    print(r.name)
    print(r.url)
    r.wait()

