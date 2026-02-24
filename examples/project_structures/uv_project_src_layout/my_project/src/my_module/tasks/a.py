from my_module.envs import env


@env.task
def process_value(x: int) -> int:
    """Process a single value."""
    return x * 2 + 5