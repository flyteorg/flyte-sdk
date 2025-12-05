import flyte

# TaskEnvironments provide a simple way of grouping configuration used by tasks (more later).
env = flyte.TaskEnvironment(
    name="hello_world",
    resources=flyte.Resources(memory="250Mi"),
    image=flyte.Image.from_debian_base().with_source_file("file.py")
)


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


if __name__ == "__main__":
    flyte.init_from_config()  # establish remote connection from within your script.
    run = flyte.run(main, x_list=list(range(10)))  # run remotely inline and pass data.

    # print various attributes of the run.
    print(run.name)
    print(run.url)

    run.wait()  # stream the logs from the root action to the terminal.
