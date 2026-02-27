import flyte

# TaskEnvironments provide a simple way of grouping configuration used by tasks
env = flyte.TaskEnvironment(name="hello_world")


# use TaskEnvironments to define tasks, which are regular Python functions.
@env.task
def fn(x: int) -> int:  # type annotations are recommended.
    slope, intercept = 2, 5
    return slope * x + intercept


# tasks can also call other tasks, which will be manifested in different containers.
@env.task
def main(x_list: list[int]) -> float:
    x_len = len(x_list)
    if x_len < 10:
        raise ValueError(f"x_list doesn't have a larger enough sample size, found: {x_len}")
    y_list = list(flyte.map(fn, x_list))  # flyte.map is like Python map, but runs in parallel.
    y_mean = sum(y_list) / len(y_list)
    return y_mean
