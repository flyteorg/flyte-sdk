from my_module.envs import env
from my_module.tasks.a import process_value


@env.task
def process_list(x_list: list[int]) -> float:
    """Process a list of values and return their mean."""
    if len(x_list) < 2:
        raise ValueError(f"x_list must have at least 2 elements, got {len(x_list)}")

    results = list(map(process_value, x_list))
    return sum(results) / len(results)
