from lib.workflows import env
from lib.workflows import utils

from flyte import task


@env.task
def process_task(value: int) -> str:
    """Example task using imported modules."""
    if not utils.validate_input(value):
        raise ValueError("Invalid input")

    environment = env.ENV_CONFIG["environment"]
    result = utils.process_data(f"Value: {value} in {environment}")
    return result


@env.task
def example_workflow(input_value: int = 42) -> str:
    """Example workflow."""
    return process_task(value=input_value)


if __name__ == "__main__":
    print(example_workflow(input_value=10))
